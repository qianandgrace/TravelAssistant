"""旅游行程规划 — LangGraph Workflow 骨架 CLI 入口。

用法：
    python main.py 杭州 3

流程：目的地 + 天数 -> 高德 MCP（geocode/天气/POI 搜索）-> qwen 合成逐日行程。
"""
import argparse
import asyncio
import os
import sys

# Windows GBK 控制台无法编码部分字符（如 emoji），遇到时用 ? 替代而不是抛异常
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from langgraph.types import Command

from utils.config import config
from utils.memory import MemoryManager
from utils.tools import get_map_tools
from utils.workflow import build_workflow, save_workflow_graph


def _check_llm_key() -> None:
    """回退链上至少要有一个可用 API key，全部缺失才报错并提示。"""
    key_env = {
        "openai": "LAOZHANG_API_KEY",
        "qwen": "QWEN_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    chain = config.LLM_FALLBACK_CHAIN
    present = [t for t in chain if os.getenv(key_env.get(t, ""))]
    if not present:
        raise SystemExit(
            f"[错误] 缺少所有 LLM 的 API key（回退链 {chain} 均无 key）。\n"
            f"   请在 .env 中配置其中至少一个：{', '.join(key_env.get(t, '') for t in chain)}。"
        )


def _print_node(node_name: str, data: dict) -> None:
    """按节点打印中间状态（stream_mode="updates" 的每个 update）。"""
    if node_name == "geocode":
        print(f"  [OK] geocode             : 坐标={data.get('location')}  adcode={data.get('adcode')}")
    elif node_name == "get_weather":
        print(f"  [OK] get_weather         : 天气=\n{data.get('weather')}")
    elif node_name == "search_pois":
        print(f"  [OK] search_pois         : 搜到 {len(data.get('pois', []))} 个 POI（景点/美食/酒店）")
    elif node_name == "retrieve_memory":
        has_mem = bool(data.get("memories"))
        print(f"  [OK] retrieve_memory     : {'检索到相关记忆' if has_mem else '暂无相关记忆'}")
    elif node_name == "summarize_conversation":
        n = data.get("summarized_count", 0)
        if n:
            print(f"  [OK] summarize_conversation: 压缩对话，裁掉 {n} 条旧消息并并入累计摘要")
        else:
            print("  [OK] summarize_conversation: 对话在窗口内，无需压缩")
    elif node_name == "plan_itinerary":
        print("  [OK] plan_itinerary      : 行程生成完成")
    elif node_name == "review_itinerary":
        action = data.get("user_action")
        note = "行程已确认" if action == "accept" else "已收到修改意见，重新规划"
        print(f"  [OK] review_itinerary    : {note}")
    elif node_name == "extract_memory":
        print(f"  [OK] extract_memory      : 提炼出 {len(data.get('knowledge') or [])} 条旅游知识")
    elif node_name == "save_memory":
        print(f"  [OK] save_memory         : {data.get('memory_saved')}")


def _resolve_interrupt(value) -> dict:
    """根据 interrupt 的 payload 类型，阻塞读取用户选择并返回 resume 值。"""
    kind = value.get("kind") if isinstance(value, dict) else None

    if kind == "review_itinerary":
        print("\n" + "=" * 50)
        print("[行程预览] 请审阅以下生成的行程：")
        print(value.get("itinerary", ""))
        print("=" * 50)
        print("请选择：")
        print("  1 - 接受（直接采用）")
        print("  2 - 编辑（粘贴修改后的完整行程，以单独一行 END 结束）")
        print("  3 - 拒绝（输入修改意见，将据此重新规划）")
        while True:
            choice = input("> ").strip()
            if choice == "2":
                print("[编辑模式] 请粘贴修改后的完整行程，最后单独一行输入 END 结束：")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                return {"action": "edit", "text": "\n".join(lines)}
            if choice == "3":
                text = input("请输入修改意见（将据此重新规划）：").strip()
                return {"action": "reject", "text": text}
            if choice in ("1", "accept", ""):  # 直接回车默认接受
                return {"action": "accept"}
            print("  无法识别，请输入 1 / 2 / 3")

    if kind == "confirm_memory":
        print("\n" + "=" * 50)
        knowledge = value.get("knowledge") or []
        if knowledge:
            print("将把以下旅游知识保存进长期记忆：")
            for i, k in enumerate(knowledge, 1):
                print(f"  {i}. {k}")
        else:
            print("本次没有提炼出新的旅游知识。")
        print("=" * 50)
        choice = input("确认保存到长期记忆？[y/N] ").strip().lower()
        return {"action": "save" if choice in ("y", "yes") else "skip"}

    # 未知类型：默认按接受处理
    return {"action": "accept"}


async def run(destination: str, days: int, preference: str = "") -> dict:
    print(f"[*] 正在为「{destination}」规划 {days} 天行程，连接高德 MCP...")
    map_tools = await get_map_tools()

    thread_id = os.getenv("TRAVEL_THREAD_ID", f"travel-{destination}")
    async with MemoryManager() as memory:
        graph = build_workflow(map_tools, memory=memory)

        # 每次运行都把 workflow 结构图保存下来（graph/*.mmd / .txt / .png），便于查看与排查
        saved_graph = save_workflow_graph(graph)
        print(f"  [OK] graph 已保存：{saved_graph['png'] or saved_graph['mmd']}")

        result = {}
        run_config = {"configurable": {"thread_id": thread_id}}
        graph_input = {"destination": destination, "days": days, "preference": preference}
        # HITL 中断-续跑循环：每次 astream 跑到下一个 interrupt 就停下，用户裁决后 resume
        while True:
            interrupts = []
            async for update in graph.astream(graph_input, run_config, stream_mode="updates"):
                for node_name, data in update.items():
                    if data is None:  # 结束标记 {END: None}，跳过
                        continue
                    if node_name == "__interrupt__":
                        interrupts.extend(data)  # tuple[Interrupt, ...]
                        continue
                    _print_node(node_name, data)
                    result.update(data)
            if not interrupts:
                break
            # 线性流程同一时刻只有一个活跃中断，一次处理一个
            resume = _resolve_interrupt(interrupts[0].value)
            graph_input = Command(resume=resume)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="旅游行程规划（LangGraph Workflow）")
    parser.add_argument("destination", help="目的地，如：杭州")
    parser.add_argument("days", type=int, help="旅行天数，如：3")
    parser.add_argument("--preference", default="", help="用户偏好，如：喜欢历史、不吃辣")
    args = parser.parse_args()

    if args.days <= 0:
        raise SystemExit("[错误] 旅行天数必须大于 0")

    _check_llm_key()

    # Windows 默认 ProactorEventLoop 与 psycopg 异步不兼容，需切换到 Selector 事件循环
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    result = asyncio.run(run(args.destination, args.days, args.preference))
    print("\n" + "=" * 50)
    print(result.get("itinerary", "（未生成行程）"))


if __name__ == "__main__":
    main()

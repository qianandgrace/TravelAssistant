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

from utils.config import config
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


async def run(destination: str, days: int) -> dict:
    print(f"[*] 正在为「{destination}」规划 {days} 天行程，连接高德 MCP...")
    map_tools = await get_map_tools()

    graph = build_workflow(map_tools)

    # 每次运行都把 workflow 结构图保存下来（graph/*.mmd / .txt / .png），便于查看与排查
    saved_graph = save_workflow_graph(graph)
    print(f"  [OK] graph 已保存：{saved_graph['png'] or saved_graph['mmd']}")

    result = {}
    # stream_mode="updates" 逐节点打印中间结果，直观展示 workflow 分步执行
    async for update in graph.astream(
        {"destination": destination, "days": days}, stream_mode="updates"
    ):
        for node_name, data in update.items():
            if node_name == "geocode":
                print(f"  [OK] geocode     : 坐标={data.get('location')}  adcode={data.get('adcode')}")
            elif node_name == "get_weather":
                print(f"  [OK] get_weather : 天气=\n{data.get('weather')}")
            elif node_name == "search_pois":
                print(f"  [OK] search_pois : 搜到 {len(data.get('pois', []))} 个 POI（景点/美食/酒店）")
            elif node_name == "plan_itinerary":
                print(f"  [OK] plan_itinerary: 行程生成完成")
        result.update(data)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="旅游行程规划（LangGraph Workflow）")
    parser.add_argument("destination", help="目的地，如：杭州")
    parser.add_argument("days", type=int, help="旅行天数，如：3")
    args = parser.parse_args()

    if args.days <= 0:
        raise SystemExit("❌ 旅行天数必须大于 0")

    _check_llm_key()

    result = asyncio.run(run(args.destination, args.days))
    print("\n" + "=" * 50)
    print(result.get("itinerary", "（未生成行程）"))


if __name__ == "__main__":
    main()

"""LangGraph Workflow：旅游行程规划骨架。

与 React agent（模型自主决定调用哪个工具）不同，这里是显式控制流的 workflow：
每个节点在代码里确定性地调用某个高德工具，LLM 只在最后的行程合成节点被调用。

图结构（传 memory 时）：
    START ─ geocode ─┬─ search_pois ─┬─ retrieve_memory ─ summarize_conversation ─ plan_itinerary
                     └─ get_weather ─┘                                                    │
                              review_itinerary ◄──(reject，带 feedback 回 plan_itinerary 重规划)──┘
                              │ (accept / edit)
                              ▼
                        extract_memory ─ save_memory ─ END

  geocode                : maps_geo 目的地名 -> 经纬度 + adcode
  get_weather            : maps_weather 查天气（依赖 geocode，保证与 search_pois 同一步汇合）
  search_pois            : maps_around_search 按坐标搜景点/美食/酒店
  retrieve_memory        : 语义检索长期记忆（episodic + semantic）+ 把本轮用户请求追加进 messages
  summarize_conversation : 短期记忆扩容：trim_messages 裁掉过旧对话，被裁部分用 LLM 合并进累计 summary
  plan_itinerary         : LLM 把 POI + 天气 + 天数 + 记忆 + 历史对话 + 修改意见合成逐日行程
  review_itinerary       : interrupt：让用户 接受/编辑/拒绝；拒绝则带 feedback 回 plan_itinerary 重新规划
  extract_memory         : LLM 从行程提炼通用旅游知识（semantic 记忆内容）
  save_memory            : interrupt 确认后落库 episodic/semantic（best-effort）

注意：interrupt 节点在 resume 时会从头重跑，`interrupt()` 前的代码会执行两次，
因此昂贵的 LLM 工作（plan_itinerary、extract_memory）都拆在中断节点之前、各自独立完成。
注意：get_weather 与 search_pois 共用 geocode 作为前驱，确保两条并行分支
在同一 superstep 完成，否则汇合节点（多入边）会被重复触发。
"""
import datetime
import json
import logging
import os
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from utils.config import config
from utils.llm import acall_with_fallback
from utils.prompts import (
    HISTORY_SUMMARIZER_PROMPT,
    ITINERARY_PLANNER_PROMPT,
    KNOWLEDGE_EXTRACTOR_PROMPT,
)

logger = logging.getLogger(__name__)

# 每类 POI 最多取多少个喂给 LLM（around_search 默认返回 20 个）
MAX_POIS_PER_CATEGORY = 8

# 短期记忆：近期对话保留窗口（近似 token，超出部分的旧消息会被摘要压缩）
RECENT_MAX_TOKENS = 3000

# 高德附近搜索用的 POI 类别
POI_CATEGORIES = ("景点", "美食", "酒店")


class TravelState(TypedDict, total=False):
    destination: str    # 目的地名
    days: int           # 旅行天数
    preference: str     # 用户偏好（--preference，可选）
    location: str       # 经纬度 "lng,lat"
    adcode: str         # 城市编码
    pois: list          # POI 列表 [{category, name, address}]
    weather: str        # 天气文本
    memories: str       # retrieve_memory 检索到的长期记忆文本
    itinerary: str      # 最终行程
    memory_saved: str   # save_memory 的保存结果说明
    # ---- 短期记忆（thread 级，checkpointer 持久化）----
    summary: str         # 累计对话摘要（被压缩掉的旧对话）
    messages: list       # 近期对话历史（BaseMessage 列表）
    summarized_count: int  # summarize_conversation 本次裁掉的旧消息条数
    # ---- HITL ----
    feedback: str        # review_itinerary 拒绝时用户给的修改意见
    user_action: str     # review_itinerary 的裁决结果：accept / reject
    knowledge: list      # extract_memory 提炼的通用旅游知识


def _extract_text(content_blocks) -> str:
    """MCP 工具返回 [{'type':'text','text':'...'}]，取出其中的文本。兼容 dict 与 ContentBlock 对象。"""
    for block in content_blocks:
        if isinstance(block, dict):
            if block.get("type") == "text":
                return block.get("text", "")
        elif getattr(block, "type", None) == "text":
            return getattr(block, "text", "")
    return str(content_blocks)


def _parse_json(content_blocks) -> dict:
    """解析 MCP 工具返回的 JSON 文本。"""
    return json.loads(_extract_text(content_blocks))


def _format_weather(weather_json: dict) -> str:
    """把天气 forecasts 整理成逐日文本，便于 LLM 阅读。"""
    city = weather_json.get("city", "")
    lines = [f"城市：{city}"]
    for f in weather_json.get("forecasts", [])[:5]:
        date = f.get("date", "")
        day_w, night_w = f.get("dayweather", ""), f.get("nightweather", "")
        desc = day_w + (f"转{night_w}" if night_w and night_w != day_w else "")
        temp = f"{f.get('daytemp')}~{f.get('nighttemp')}°C"
        wind = f"{f.get('daywind', '')}风 {f.get('daypower', '')}"
        lines.append(f"{date}：{desc}，{temp}，{wind}")
    return "\n".join(lines)


def _format_pois(pois: list) -> str:
    """把 POI 列表按类别整理成分组文本，便于 LLM 阅读。"""
    grouped: dict[str, list] = {}
    for p in pois:
        grouped.setdefault(p["category"], []).append(p)
    lines = []
    for category, items in grouped.items():
        lines.append(f"【{category}】")
        for i, item in enumerate(items, 1):
            addr = item.get("address") or "地址待查"
            lines.append(f"{i}. {item['name']}（{addr}）")
        lines.append("")
    return "\n".join(lines).strip()


def _format_history(state: TravelState) -> str:
    """把累计摘要 + 近期对话拼成给 plan prompt 的文本（短期记忆上下文）。"""
    parts = []
    if state.get("summary"):
        parts.append("【历史对话摘要】\n" + state["summary"].strip())
    msgs = state.get("messages") or []
    if msgs:
        lines = []
        for m in msgs[-6:]:
            role = "用户" if isinstance(m, HumanMessage) else "助手"
            lines.append(f"{role}：{str(m.content)[:300]}")
        parts.append("【近期对话】\n" + "\n".join(lines))
    return "\n".join(parts)


async def _summarize_history(existing_summary: str, dropped_messages: list) -> str:
    """把被 trim 掉的旧对话合并进累计摘要（短期记忆扩容）。"""
    history = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}：{m.content}"
        for m in dropped_messages
    )
    prompt = HISTORY_SUMMARIZER_PROMPT.format(
        existing_summary=existing_summary.strip() or "（空）",
        history=history.strip() or "（空）",
    )
    res = await acall_with_fallback([HumanMessage(content=prompt)])
    return str(res.content).strip()


def create_nodes(map_tools):
    """根据高德工具集合构造 workflow 的节点函数（闭包持有工具）。"""
    tools = {t.name: t for t in map_tools}
    geo_tool = tools["maps_geo"]
    weather_tool = tools["maps_weather"]
    around_tool = tools["maps_around_search"]

    async def geocode(state: TravelState) -> dict:
        res = await geo_tool.ainvoke({"address": state["destination"]})
        data = _parse_json(res)
        results = data.get("results") or data.get("geocodes") or []
        if not results:
            raise ValueError(f"高德无法解析目的地：{state['destination']}")
        first = results[0]
        return {"location": first["location"], "adcode": first.get("adcode", "")}

    async def get_weather(state: TravelState) -> dict:
        res = await weather_tool.ainvoke({"city": state["destination"]})
        return {"weather": _format_weather(_parse_json(res))}

    async def search_pois(state: TravelState) -> dict:
        location = state["location"]
        pois = []
        for category in POI_CATEGORIES:
            res = await around_tool.ainvoke(
                {"keywords": category, "location": location, "radius": "30000"}
            )
            data = _parse_json(res)
            for p in data.get("pois", [])[:MAX_POIS_PER_CATEGORY]:
                pois.append(
                    {
                        "category": category,
                        "name": p.get("name", ""),
                        "address": p.get("address", ""),
                    }
                )
        return {"pois": pois}

    async def plan_itinerary(state: TravelState) -> dict:
        prompt = ITINERARY_PLANNER_PROMPT.format(
            destination=state["destination"],
            days=state["days"],
            weather=state.get("weather", "（无天气数据）"),
            pois_text=_format_pois(state.get("pois", [])) or "（未获取到 POI 数据）",
            preference=state.get("preference") or "（未提供）",
            memories=state.get("memories") or "（暂无相关记忆）",
            history=_format_history(state) or "（暂无对话历史）",
            feedback=state.get("feedback") or "（无）",
        )
        # 按 config.LLM_FALLBACK_CHAIN 顺序调用：qwen -> deepseek -> openai
        res = await acall_with_fallback([HumanMessage(content=prompt)])
        return {"itinerary": str(res.content)}

    return geocode, get_weather, search_pois, plan_itinerary


async def _extract_knowledge(destination: str, itinerary: str) -> list[str]:
    """用 LLM 把行程提炼成通用旅游知识（semantic 记忆内容）。"""
    if not itinerary:
        return []
    prompt = KNOWLEDGE_EXTRACTOR_PROMPT.format(destination=destination, itinerary=itinerary)
    res = await acall_with_fallback([HumanMessage(content=prompt)])
    text = str(res.content).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:  # noqa: BLE001 - 解析失败则按行回退
        pass
    return [
        line.strip().lstrip("- ").strip()
        for line in text.splitlines()
        if line.strip() and line.strip() not in ("[]",)
    ]


def build_workflow(map_tools, memory=None):
    """构建并编译 LangGraph workflow。

    Args:
        map_tools: 来自 utils.tools.get_map_tools()。
        memory:    utils.memory.MemoryManager，传入后启用长短期记忆节点并以
                   Postgres 作为 thread 级 checkpointer；为 None 时保持原 4 节点流程。
    """
    geocode, get_weather, search_pois, plan_itinerary = create_nodes(map_tools)

    graph = StateGraph(TravelState)
    graph.add_node("geocode", geocode)
    graph.add_node("get_weather", get_weather)
    graph.add_node("search_pois", search_pois)
    graph.add_node("plan_itinerary", plan_itinerary)

    if memory is not None:
        async def retrieve_memory(state: TravelState) -> dict:
            """语义检索长期记忆 + 把本轮用户请求追加进短期对话历史。"""
            dest = state.get("destination", "")
            pref = state.get("preference", "")
            parts = [dest, pref, "旅游 景点 美食 酒店"]
            text = await memory.retrieve(query=" ".join(p for p in parts if p))
            msgs = list(state.get("messages") or [])
            msgs.append(
                HumanMessage(
                    content=f"请规划{dest}的{state.get('days', 0)}天行程，偏好：{pref or '无'}"
                )
            )
            return {"memories": text, "messages": msgs}

        async def summarize_conversation(state: TravelState) -> dict:
            """短期记忆扩容：trim_messages 裁掉过旧对话，被裁部分用 LLM 合并进累计摘要。"""
            messages = list(state.get("messages") or [])
            recent = trim_messages(
                messages,
                max_tokens=RECENT_MAX_TOKENS,
                strategy="last",
                token_counter="approximate",
            )
            # 对象身份比较，避免依赖 message.id（trim_messages 返回原对象引用）
            dropped = [m for m in messages if all(m is not r for r in recent)]
            summary = state.get("summary") or ""
            if dropped:
                summary = await _summarize_history(summary, dropped)
            return {
                "messages": recent,
                "summary": summary,
                "summarized_count": len(dropped),
            }

        async def review_itinerary(state: TravelState) -> dict:
            """HITL：interrupt 让用户接受 / 编辑 / 拒绝生成的行程。"""
            decision = interrupt(
                {"kind": "review_itinerary", "itinerary": state.get("itinerary", "")}
            )
            action = decision.get("action") if isinstance(decision, dict) else "accept"
            itinerary = state.get("itinerary", "")
            msgs = list(state.get("messages") or [])
            if action == "edit":
                edited = str(decision.get("text") or "").strip() or itinerary
                msgs.append(AIMessage(content=edited))
                return {
                    "itinerary": edited,
                    "user_action": "accept",
                    "feedback": "",
                    "messages": msgs,
                }
            if action == "reject":
                fb = str(decision.get("text") or "").strip()
                if fb:
                    msgs.append(HumanMessage(content=f"对行程不满意，修改意见：{fb}"))
                return {"user_action": "reject", "feedback": fb, "messages": msgs}
            # accept：沿用当前 itinerary
            msgs.append(AIMessage(content=itinerary))
            return {"user_action": "accept", "feedback": "", "messages": msgs}

        async def extract_memory(state: TravelState) -> dict:
            """LLM 提炼通用旅游知识（独立节点，避免 interrupt 重跑重复耗 LLM）。"""
            dest = state.get("destination", "")
            knowledge = await _extract_knowledge(dest, state.get("itinerary", ""))
            return {"knowledge": knowledge}

        async def save_memory(state: TravelState) -> dict:
            """HITL 确认后落库 episodic/semantic（best-effort，失败不阻断本次行程）。"""
            decision = interrupt(
                {"kind": "confirm_memory", "knowledge": state.get("knowledge") or []}
            )
            if not (isinstance(decision, dict) and decision.get("action") == "save"):
                return {"memory_saved": "用户放弃保存记忆"}
            status = "记忆保存失败"
            try:
                dest = state.get("destination", "")
                itinerary = state.get("itinerary", "")
                await memory.save_episodic(
                    destination=dest,
                    days=state.get("days", 0),
                    preference=state.get("preference", ""),
                    itinerary=itinerary,
                    weather=state.get("weather", ""),
                )
                await memory.save_semantic(dest, state.get("knowledge") or [])
                status = f"episodic+semantic 已保存（{len(state.get('knowledge') or [])} 条知识）"
            except Exception as e:  # noqa: BLE001
                logger.warning("保存记忆失败（不影响本次行程）：%s", e)
            return {"memory_saved": status}

        graph.add_node("retrieve_memory", retrieve_memory)
        graph.add_node("summarize_conversation", summarize_conversation)
        graph.add_node("review_itinerary", review_itinerary)
        graph.add_node("extract_memory", extract_memory)
        graph.add_node("save_memory", save_memory)

    graph.add_edge(START, "geocode")
    graph.add_edge("geocode", "search_pois")
    graph.add_edge("geocode", "get_weather")  # 与 search_pois 并行，且与汇合点同一步完成

    if memory is not None:
        graph.add_edge("search_pois", "retrieve_memory")
        graph.add_edge("get_weather", "retrieve_memory")  # 汇合点（同一步完成，只触发一次）
        graph.add_edge("retrieve_memory", "summarize_conversation")
        graph.add_edge("summarize_conversation", "plan_itinerary")
        graph.add_edge("plan_itinerary", "review_itinerary")
        graph.add_conditional_edges(
            "review_itinerary",
            lambda s: "plan_itinerary" if s.get("user_action") == "reject" else "extract_memory",
            {"plan_itinerary": "plan_itinerary", "extract_memory": "extract_memory"},
        )
        graph.add_edge("extract_memory", "save_memory")
        graph.add_edge("save_memory", END)
        return graph.compile(checkpointer=memory.checkpointer)

    graph.add_edge("search_pois", "plan_itinerary")
    graph.add_edge("get_weather", "plan_itinerary")  # 汇合点
    graph.add_edge("plan_itinerary", END)
    return graph.compile()


def save_workflow_graph(graph, output_dir: str = "graph") -> dict:
    """把编译好的 workflow 图保存到磁盘，每次运行生成一个带时间戳的新快照。

    产出（同名前缀，时间戳区分，可追溯到某次运行）：
      - {prefix}.mmd  Mermaid 文本：可在 https://mermaid.live 或 VS Code（Mermaid 扩展）打开
      - {prefix}.txt  ASCII 结构：直接看节点与连线
      - {prefix}.png  本地渲染的 PNG 图（依赖 pygraphviz；未安装时自动跳过，不影响其它文件）

    Returns:
        {"mmd": path, "txt": path, "png": path}，未生成的文件为 None。
    """
    gr = graph.get_graph()
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"workflow_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    saved = {"mmd": None, "txt": None, "png": None}

    # 1) Mermaid 文本（零依赖，总是保存）
    mmd_path = os.path.join(output_dir, f"{prefix}.mmd")
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(gr.draw_mermaid())
    saved["mmd"] = mmd_path

    # 2) ASCII 结构（依赖 grandalf；缺失时跳过，不阻断）
    try:
        txt_path = os.path.join(output_dir, f"{prefix}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(gr.draw_ascii())
        saved["txt"] = txt_path
    except ImportError:
        print("  [跳过] 未安装 grandalf，不保存 ASCII（可 pip install grandalf 启用）")
    except Exception as e:  # noqa: BLE001
        print(f"  [警告] ASCII 保存失败：{e}")

    # 3) PNG 本地渲染（best effort，用 pygraphviz，不走外部渲染服务）
    try:
        import pygraphviz as pgv

        a = pgv.AGraph(directed=True)
        for nid, node in gr.nodes.items():
            a.add_node(str(nid), label=str(node.name))
        for edge in gr.edges:
            a.add_edge(str(edge.source), str(edge.target))
        a.layout(prog="dot")
        png_path = os.path.join(output_dir, f"{prefix}.png")
        a.draw(png_path, format="png")
        saved["png"] = png_path
    except ImportError:
        print("  [跳过] 未安装 pygraphviz，不渲染 PNG（可 pip install pygraphviz 启用）")
    except Exception as e:  # noqa: BLE001 - 渲染失败不应阻断主流程
        print(f"  [警告] PNG 渲染失败：{e}")

    return saved

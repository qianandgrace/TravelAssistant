"""LangGraph Workflow：旅游行程规划骨架。

与 React agent（模型自主决定调用哪个工具）不同，这里是显式控制流的 workflow：
每个节点在代码里确定性地调用某个高德工具，LLM 只在最后的行程合成节点被调用。

图结构：
    START ─ geocode ─┬─ search_pois ─┐
                     └─ get_weather ─┴─ plan_itinerary ─ END

  geocode        : maps_geo 目的地名 -> 经纬度 + adcode
  get_weather    : maps_weather 查天气（依赖 geocode，保证与 search_pois 同一步汇合）
  search_pois    : maps_around_search 按坐标搜景点/美食/酒店
  plan_itinerary : LLM 把 POI + 天气 + 天数合成逐日行程

注意：get_weather 与 search_pois 共用 geocode 作为前驱，确保两条并行分支
在同一 superstep 完成，否则 plan_itinerary（多入边节点）会被重复触发。
"""
import datetime
import json
import os
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from utils.config import config
from utils.llm import acall_with_fallback
from utils.prompts import ITINERARY_PLANNER_PROMPT

# 每类 POI 最多取多少个喂给 LLM（around_search 默认返回 20 个）
MAX_POIS_PER_CATEGORY = 8

# 高德附近搜索用的 POI 类别
POI_CATEGORIES = ("景点", "美食", "酒店")


class TravelState(TypedDict, total=False):
    destination: str   # 目的地名
    days: int          # 旅行天数
    location: str      # 经纬度 "lng,lat"
    adcode: str        # 城市编码
    pois: list         # POI 列表 [{category, name, address}]
    weather: str       # 天气文本
    itinerary: str     # 最终行程


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
        )
        # 按 config.LLM_FALLBACK_CHAIN 顺序调用：qwen -> deepseek -> openai
        res = await acall_with_fallback([HumanMessage(content=prompt)])
        return {"itinerary": str(res.content)}

    return geocode, get_weather, search_pois, plan_itinerary


def build_workflow(map_tools):
    """构建并编译 LangGraph workflow。map_tools 来自 utils.tools.get_map_tools()。"""
    geocode, get_weather, search_pois, plan_itinerary = create_nodes(map_tools)

    graph = StateGraph(TravelState)
    graph.add_node("geocode", geocode)
    graph.add_node("get_weather", get_weather)
    graph.add_node("search_pois", search_pois)
    graph.add_node("plan_itinerary", plan_itinerary)

    graph.add_edge(START, "geocode")
    graph.add_edge("geocode", "search_pois")
    graph.add_edge("geocode", "get_weather")  # 与 search_pois 并行，且与汇合点同一步完成
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

"""LangGraph Workflow：旅游行程规划骨架。

与 React agent（模型自主决定调用哪个工具）不同，这里是显式控制流的 workflow：
每个节点在代码里确定性地调用某个高德工具，LLM 只在最后的行程合成节点被调用。

图结构（传 memory 时）：
    START ─ geocode ─┬─ search_pois ─┬─ retrieve_memory ─ summarize_conversation ─ do_research ─ plan_itinerary
                     └─ get_weather ─┘                                                                   │
                              review_itinerary ◄──(reject，带 feedback 回 plan_itinerary 重规划)──┘
                              │ (accept / edit)
                              ▼
                        extract_memory ─ save_memory ─ END

  geocode                : maps_geo 目的地名 -> 经纬度 + adcode
  get_weather            : maps_weather 查天气（依赖 geocode，保证与 search_pois 同一步汇合）
  search_pois            : maps_around_search 按坐标搜景点/美食/酒店
  do_research            : Travel Research：搜索攻略来源（配置了 TAVILY_API_KEY 时）+ 生成紧凑研究摘要
  retrieve_memory        : 语义检索长期记忆（episodic + semantic）+ 把本轮用户请求追加进 messages
  summarize_conversation : 短期记忆扩容：trim_messages 裁掉过旧对话，被裁部分用 LLM 合并进累计 summary
  plan_itinerary         : LLM 把 POI + 天气 + 研究摘要 + 天数 + 记忆 + 历史对话 + 修改意见合成逐日行程（严格 JSON）
  enrich_routes          : 地图服务填充：poi_id/geocode 解析每个 item 真实坐标，每天步行方向算真实距离/时长（LLM 不参与）
  enrich_images          : POI/搜索 API 真实图片填充 item.image，失败回退占位图（LLM 不编造 URL）
  review_itinerary       : interrupt：让用户 接受/编辑/拒绝；拒绝则带 feedback 回 plan_itinerary 重新规划
  extract_memory         : LLM 从行程提炼通用旅游知识（semantic 记忆内容）
  save_memory            : interrupt 确认后落库 episodic/semantic（best-effort）

注意：interrupt 节点在 resume 时会从头重跑，`interrupt()` 前的代码会执行两次，
因此昂贵的 LLM 工作（plan_itinerary、extract_memory）都拆在中断节点之前、各自独立完成。
注意：get_weather 与 search_pois 共用 geocode 作为前驱，确保两条并行分支
在同一 superstep 完成，否则汇合节点（多入边）会被重复触发。
"""
import asyncio
import datetime
import json
import logging
import os
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from utils.config import config
from utils.images import fill_item_images
from utils.itinerary_schema import (
    parse_itinerary_json,
    render_itinerary_md,
    validate_and_normalize,
)
from utils.llm import acall_with_fallback
from utils.prompts import (
    HISTORY_SUMMARIZER_PROMPT,
    ITINERARY_PLANNER_JSON_PROMPT,
    ITINERARY_PLANNER_PROMPT,
    ITINERARY_REPAIR_PROMPT,
    KNOWLEDGE_EXTRACTOR_PROMPT,
)
from utils.research import research_summary, search_guides

logger = logging.getLogger(__name__)

# 每类 POI 最多取多少个喂给 LLM（around_search 默认返回 20 个）
MAX_POIS_PER_CATEGORY = 8

# 短期记忆：近期对话保留窗口（近似 token，超出部分的旧消息会被摘要压缩）
RECENT_MAX_TOKENS = 3000

# 高德附近搜索用的 POI 类别
POI_CATEGORIES = ("景点", "美食", "酒店")

# 参与路线计算/地图 marker 的 item 类型（有固定物理位置的真实地点；
# 交通/自由活动/其他 是移动或非固定点，geocode 会产生无意义坐标）
ROUTE_PLACE_TYPES = {"景点", "美食", "酒店"}


class TravelState(TypedDict, total=False):
    destination: str    # 目的地名
    days: int           # 旅行天数
    query: str          # 用户本轮原始输入（修改行程等场景需原样给 planner）
    preference: str     # 用户偏好（--preference，可选）
    start_date: str     # 出发日期 YYYY-MM-DD（可选，来自实体抽取）
    location: str       # 经纬度 "lng,lat"
    adcode: str         # 城市编码
    pois: list          # POI 列表 [{category, name, address, id, location}]
    weather: str        # 天气文本
    research: dict      # Travel Research 摘要 {area_clusters, common_routes, ..., sources}
    memories: str       # retrieve_memory 检索到的长期记忆文本
    itinerary: str      # 最终行程（Markdown 渲染，供现有前端/HITL 展示）
    itinerary_data: dict  # 结构化行程 JSON（校验后）
    itinerary_note: str   # 行程生成说明（如回退文本版时）
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


def _split_location(loc) -> tuple[float, float] | None:
    """把高德的 'lng,lat' 拆成 (lng, lat) 浮点数；非法则返回 None。"""
    if not loc:
        return None
    try:
        lng, lat = str(loc).split(",")
        return float(lng), float(lat)
    except (TypeError, ValueError):
        return None


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
    """把 POI 列表按类别整理成分组文本，便于 LLM 阅读并引用 poi_id。"""
    grouped: dict[str, list] = {}
    for p in pois:
        grouped.setdefault(p["category"], []).append(p)
    lines = []
    for category, items in grouped.items():
        lines.append(f"【{category}】")
        for i, item in enumerate(items, 1):
            addr = item.get("address") or "地址待查"
            pid = item.get("id") or ""
            lines.append(f"{i}. [{item.get('name')}]（{addr}）  id={pid}")
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


def _format_dates(start_date: str, days: int) -> str:
    """把出发日期 + 天数展开成逐日日期串，供 planner 使用；无日期则给占位说明。"""
    if not start_date:
        return "（未指定，按第 1 天起逐日递增）"
    try:
        base = datetime.date.fromisoformat(start_date)
    except ValueError:
        return "（未指定，按第 1 天起逐日递增）"
    return "、".join((base + datetime.timedelta(days=i)).isoformat() for i in range(days))


def _format_research(research: dict) -> str:
    """把 Travel Research 摘要拼成给 plan prompt 的紧凑文本（非整篇文章）。"""
    if not research:
        return "（无）"
    labels = (
        ("area_clusters", "热门区域"),
        ("common_routes", "常见路线"),
        ("popular_combinations", "热门搭配"),
        ("transportation_tips", "交通提示"),
        ("avoid", "避坑"),
        ("practical_tips", "实用建议"),
    )
    lines = []
    for key, label in labels:
        items = [str(x).strip() for x in (research.get(key) or []) if str(x).strip()]
        if items:
            lines.append(f"{label}：" + "；".join(items))
    return "\n".join(lines).strip() or "（无）"


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
    walking_tool = tools["maps_direction_walking"]
    text_search_tool = tools["maps_text_search"]

    async def geocode(state: TravelState) -> dict:
        res = await geo_tool.ainvoke({"address": state["destination"]})
        data = _parse_json(res)
        results = data.get("results") or data.get("geocodes") or []
        if not results:
            raise ValueError(f"高德无法解析目的地：{state['destination']}")
        first = results[0]
        return {"location": first["location"], "adcode": first.get("adcode", "")}

    async def get_weather(state: TravelState) -> dict:
        """天气是锦上添花的上下文：失败返回空，绝不阻断规划。"""
        try:
            res = await weather_tool.ainvoke({"city": state["destination"]})
            return {"weather": _format_weather(_parse_json(res))}
        except Exception as e:  # noqa: BLE001 - 天气失败降级为空
            logger.warning("天气获取失败（降级为空）：%s", e)
            return {"weather": ""}

    async def search_pois(state: TravelState) -> dict:
        """POI 失败降级为空列表（planner 有『未获取到 POI 数据』兜底），不阻断规划。"""
        location = state["location"]
        pois = []
        try:
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
                            "id": p.get("id", ""),          # 供 LLM 引用 poi_id / Phase 5 匹配坐标
                            "location": p.get("location", ""),  # "lng,lat"，Phase 5 使用
                            "photo": p.get("photo", ""),        # 真实 POI 图，Phase 6 使用
                        }
                    )
        except Exception as e:  # noqa: BLE001 - POI 失败降级为空
            logger.warning("POI 搜索失败（降级为空）：%s", e)
            pois = []
        return {"pois": pois}

    async def do_research(state: TravelState) -> dict:
        """攻略/游记研究：搜索来源（可用时）-> 生成紧凑研究摘要。

        研究是可选上下文：搜索失败本身返回空来源；摘要 LLM 失败降级为空结构，
        绝不阻断后续规划（planner 对『（无）』研究有兜底）。
        """
        dest = state["destination"]
        pref = state.get("preference", "")
        days = state.get("days", 0)
        try:
            sr = await search_guides(dest, pref, days)
            summary = await research_summary(dest, pref, days, sr["sources"])
            summary["search_available"] = sr["search_available"]
            return {"research": summary}
        except Exception as e:  # noqa: BLE001 - 研究失败降级为空
            logger.warning("研究摘要失败（降级为空）：%s", e)
            return {"research": {}}

    def _base_prompt_fields(state: TravelState) -> dict:
        """plan prompt 共用的字段（JSON 与文本兜底都用）。"""
        return {
            "destination": state["destination"],
            "days": state["days"],
            "dates": _format_dates(state.get("start_date"), state["days"]),
            "weather": state.get("weather", "（无天气数据）"),
            "research": _format_research(state.get("research")),
            "pois_text": _format_pois(state.get("pois", [])) or "（未获取到 POI 数据）",
            "preference": state.get("preference") or "（未提供）",
            "memories": state.get("memories") or "（暂无相关记忆）",
            "history": _format_history(state) or "（暂无对话历史）",
            "feedback": state.get("feedback") or "（无）",
        }

    async def _generate_itinerary_json(state: TravelState) -> dict | None:
        """结构化生成行程：LLM 严格 JSON -> 解析 -> 校验/规整 -> 失败修复一次。"""
        fields = _base_prompt_fields(state)
        prompt = ITINERARY_PLANNER_JSON_PROMPT.format(**fields)
        res = await acall_with_fallback([HumanMessage(content=prompt)])
        text = str(res.content)
        data, err = _try_parse_validate(text, state)
        if data is not None:
            return data
        # 修复一次：带原文 + 错误让 LLM 重新输出
        repair_prompt = ITINERARY_REPAIR_PROMPT.format(
            days=state["days"], invalid=text, error=err or "结构不符合要求",
        )
        res2 = await acall_with_fallback([HumanMessage(content=repair_prompt)])
        data2, err2 = _try_parse_validate(str(res2.content), state)
        if data2 is not None:
            return data2
        logger.error("行程 JSON 两次失败，回退文本版：%s | %s", err, err2)
        return None

    def _try_parse_validate(text: str, state: TravelState):
        data, err = parse_itinerary_json(text)
        if data is None:
            return None, err
        norm, errors = validate_and_normalize(
            data, state["days"], state.get("start_date", "")
        )
        if errors:
            return None, "；".join(errors)
        return norm, None

    async def plan_itinerary(state: TravelState) -> dict:
        """结构化行程优先：JSON 校验通过则产出 itinerary_data + Markdown 渲染；
        两次失败则回退原文本 prompt，保证用户一定拿到可读行程。"""
        data = await _generate_itinerary_json(state)
        if data is not None:
            data["sources"] = [
                {"title": s.get("title", ""), "url": s.get("url", "")}
                for s in (state.get("research") or {}).get("sources", [])
            ]
            return {"itinerary_data": data, "itinerary": render_itinerary_md(data)}
        # 兜底：文本版行程
        fields = _base_prompt_fields(state)
        prompt = ITINERARY_PLANNER_PROMPT.format(**fields)
        res = await acall_with_fallback([HumanMessage(content=prompt)])
        return {
            "itinerary": str(res.content),
            "itinerary_data": None,
            "itinerary_note": "结构化行程生成失败，已回退为文本版行程",
        }

    async def _resolve_item_coord(item: dict, pois_by_id: dict, city: str) -> tuple[float, float] | None:
        """解析单个 item 的坐标：优先 poi_id 命中搜索结果坐标，未命中用名称/地址 geocode。

        任何失败都返回 None（不阻塞），由调用方决定跳过该点。
        """
        pid = item.get("poi_id")
        if pid and pid in pois_by_id:
            pt = _split_location(pois_by_id[pid].get("location"))
            if pt:
                return pt
        address = str(item.get("address") or "").strip() or str(item.get("name") or "").strip()
        if not address:
            return None
        try:
            res = await geo_tool.ainvoke({"address": address, "city": city})
            results = (_parse_json(res).get("results") or _parse_json(res).get("geocodes")) or []
            if results:
                pt = _split_location(results[0].get("location"))
                if pt:
                    return pt
        except Exception:  # noqa: BLE001 - 单点 geocode 失败跳过，不崩
            pass
        return None

    async def _walking_leg(origin: str, destination: str) -> tuple[float, float] | None:
        """调高德步行方向取一段真实距离(米)/时长(秒)；失败返回 None。"""
        try:
            res = await walking_tool.ainvoke({"origin": origin, "destination": destination})
            path = _parse_json(res)["route"]["paths"][0]
            return float(path.get("distance", 0) or 0), float(path.get("duration", 0) or 0)
        except Exception:  # noqa: BLE001 - 单段失败跳过
            return None

    async def enrich_routes(state: TravelState) -> dict:
        """Phase 5：用地图服务填充真实坐标 + 每天真实路线距离/时长。

        LLM 在生成时被强制 latitude/longitude=null、route=0（职责分离），
        本节点全部由高德 API 计算：poi_id 命中搜索结果的坐标，未命中用名称/地址
        geocode；每天相邻两个有坐标的 item 之间调 maps_direction_walking 取真实
        距离与时长，并生成 route.points（markers + 连线）。单点失败即跳过该点，
        全部失败则保留空 route —— 永不崩溃。
        """
        data = state.get("itinerary_data")
        if not data or not data.get("days"):
            return {}  # 文本回退版行程无结构化数据，跳过
        city = state["destination"]
        pois_by_id = {p.get("id"): p for p in (state.get("pois") or []) if p.get("id")}
        days = data["days"]

        # 1) 解析每个 item 的真实坐标（只解析有固定位置的地点类型）
        for day in days:
            for item in day.get("items") or []:
                if item.get("type") not in ROUTE_PLACE_TYPES:
                    continue
                pt = await _resolve_item_coord(item, pois_by_id, city)
                if pt:
                    item["longitude"], item["latitude"] = pt[0], pt[1]

        # 2) 计算每天路线：相邻地点类 item 之间步行方向，并收集 markers
        for day in days:
            pts = [
                {"name": it.get("name", ""), "longitude": it["longitude"], "latitude": it["latitude"]}
                for it in day.get("items") or []
                if it.get("type") in ROUTE_PLACE_TYPES
                and it.get("longitude") is not None and it.get("latitude") is not None
            ]
            if len(pts) < 2:
                continue
            legs = [
                (f"{a['longitude']},{a['latitude']}", f"{b['longitude']},{b['latitude']}")
                for a, b in zip(pts, pts[1:])
            ]
            results = await asyncio.gather(
                *[_walking_leg(o, d) for o, d in legs],
                return_exceptions=True,
            )
            distance_m = 0.0
            duration_s = 0.0
            for r in results:
                if isinstance(r, Exception) or r is None:
                    continue
                distance_m += r[0]
                duration_s += r[1]
            day["route"] = {
                "points": pts,
                "distance_km": round(distance_m / 1000.0, 1),
                "estimated_minutes": int(round(duration_s / 60.0)),
            }

        return {"itinerary_data": data, "itinerary": render_itinerary_md(data)}

    async def enrich_images(state: TravelState) -> dict:
        """Phase 6：用 POI/搜索 API 的真实图片填充每个 item 的 image。

        图片来源严格为高德搜索结果（around_search 的 photo / text_search 兜底），
        绝不由 LLM 编造 URL；任何未命中的 item 回退本地占位图（data-URI SVG）。
        """
        data = state.get("itinerary_data")
        if not data or not data.get("days"):
            return {}
        city = state["destination"]
        pois_by_id = {p.get("id"): p for p in (state.get("pois") or []) if p.get("id")}
        all_items = [it for day in data["days"] for it in (day.get("items") or [])]
        await fill_item_images(all_items, pois_by_id, text_search_tool, city)
        return {"itinerary_data": data, "itinerary": render_itinerary_md(data)}

    return (
        geocode, get_weather, search_pois, plan_itinerary,
        do_research, enrich_routes, enrich_images,
    )


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
    geocode, get_weather, search_pois, plan_itinerary, do_research, enrich_routes, enrich_images = create_nodes(map_tools)

    graph = StateGraph(TravelState)
    graph.add_node("geocode", geocode)
    graph.add_node("get_weather", get_weather)
    graph.add_node("search_pois", search_pois)
    graph.add_node("do_research", do_research)
    graph.add_node("plan_itinerary", plan_itinerary)
    graph.add_node("enrich_routes", enrich_routes)
    graph.add_node("enrich_images", enrich_images)

    if memory is not None:
        async def retrieve_memory(state: TravelState) -> dict:
            """语义检索长期记忆 + 把本轮用户请求原样追加进短期对话历史。

            追加用户原始输入（state.query），使『修改行程』等意图的指令原样到达
            planner；记忆检索失败降级为空，不阻断。
            """
            dest = state.get("destination", "")
            pref = state.get("preference", "")
            try:
                parts = [dest, pref, "旅游 景点 美食 酒店"]
                text = await memory.retrieve(query=" ".join(p for p in parts if p))
            except Exception as e:  # noqa: BLE001 - 记忆检索失败降级为空
                logger.warning("长期记忆检索失败（降级为空）：%s", e)
                text = ""
            msgs = list(state.get("messages") or [])
            need = state.get("query") or (
                f"请规划{dest}的{state.get('days', 0)}天行程，偏好：{pref or '无'}"
            )
            msgs.append(HumanMessage(content=need))
            return {"memories": text, "messages": msgs}

        async def summarize_conversation(state: TravelState) -> dict:
            """短期记忆扩容：trim_messages 裁掉过旧对话，被裁部分用 LLM 合并进累计摘要。

            摘要 LLM 失败时本轮不压缩（保留全部消息），避免丢失对话上下文。
            """
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
            summarized = len(dropped)
            if dropped:
                try:
                    summary = await _summarize_history(summary, dropped)
                except Exception as e:  # noqa: BLE001 - 摘要失败则不压缩，保留上下文
                    logger.warning("对话摘要失败，本轮不压缩：%s", e)
                    recent, summarized = messages, 0
            return {
                "messages": recent,
                "summary": summary,
                "summarized_count": summarized,
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
        # 汇合点：search_pois / get_weather 同一步完成后进入 retrieve_memory，只触发一次
        graph.add_edge("search_pois", "retrieve_memory")
        graph.add_edge("get_weather", "retrieve_memory")
        graph.add_edge("retrieve_memory", "summarize_conversation")
        graph.add_edge("summarize_conversation", "do_research")
        graph.add_edge("do_research", "plan_itinerary")
        # 结构化行程生成后 -> 填充坐标/路线 -> 填充真实图片 -> 再交用户裁决
        graph.add_edge("plan_itinerary", "enrich_routes")
        graph.add_edge("enrich_routes", "enrich_images")
        graph.add_edge("enrich_images", "review_itinerary")
        graph.add_conditional_edges(
            "review_itinerary",
            lambda s: "plan_itinerary" if s.get("user_action") == "reject" else "extract_memory",
            {"plan_itinerary": "plan_itinerary", "extract_memory": "extract_memory"},
        )
        graph.add_edge("extract_memory", "save_memory")
        graph.add_edge("save_memory", END)
        return graph.compile(checkpointer=memory.checkpointer)

    # 无记忆分支：汇合点在 do_research
    graph.add_edge("search_pois", "do_research")
    graph.add_edge("get_weather", "do_research")  # 汇合点（同一步完成，只触发一次）
    graph.add_edge("do_research", "plan_itinerary")
    graph.add_edge("plan_itinerary", "enrich_routes")
    graph.add_edge("enrich_routes", "enrich_images")
    graph.add_edge("enrich_images", END)
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

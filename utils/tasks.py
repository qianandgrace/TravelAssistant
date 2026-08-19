"""Celery 异步任务：在 worker 进程里运行 LangGraph travel workflow。

两个任务：
  invoke_agent_task   : 对话式输入 -> 实体抽取 -> 跑 workflow（HITL 中断则存状态等待）
  resume_agent_task   : 用 Command(resume=...) 续跑被中断的 workflow

会话状态（idle/pending/running/interrupted/completed/error）写回 Redis，
图状态由 Postgres checkpointer 持久化 —— 客户端/服务端故障后都可按 task_id 查询与恢复。
"""
import asyncio
import logging
import sys

from celery import Celery
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from utils.config import Config
from utils.entity_extractor import extract_travel_entities
from utils.images import fill_destination_images
from utils.intent_router import NON_TRAVEL_REPLY, classify_intent
from utils.llm import acall_with_fallback
from utils.memory import MemoryManager
from utils.prompts import TRAVEL_QA_PROMPT
from utils.recommendation import recommend_destinations
from utils.session_manager import get_session_manager
from utils.tools import get_map_tools
from utils.workflow import build_workflow

logger = logging.getLogger(__name__)

celery_app = Celery(
    "travel_assistant",
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_track_started=True,
    result_expires=int(Config.TASK_TTL),
    broker_connection_retry_on_startup=True,
    timezone="Asia/Shanghai",
    # 不让 celery 劫持根 logger：应用日志级别由 LOG_LEVEL 环境变量统一控制，
    # 各流程节点的 DEBUG 日志才能透出到控制台/文件。
    worker_hijack_root_logger=False,
)


def _prepare_loop() -> None:
    """Windows 下 psycopg 异步需要 Selector 事件循环（默认 Proactor 不兼容）。"""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _safe_result(state: dict, parsed: dict) -> dict:
    """只保留 JSON 安全的关键字段，避免 messages(BaseMessage)/pois/weather 等序列化失败。"""
    keys = ("destination", "days", "preference", "start_date", "itinerary",
            "itinerary_data", "itinerary_note", "research",
            "memory_saved", "knowledge", "summary")
    result = {k: state.get(k) for k in keys if state.get(k) is not None}
    result["parsed"] = parsed
    return result


def _merge_modification_context(parsed: dict, prior: dict, query: str) -> dict:
    """修改行程：用户通常只说改动、不重复目的地。

    当抽取结果里没有「明确的新目的地」（为空、或兜底成整句原文 query）时，
    复用上一轮会话的 destination/days/preference/start_date，避免 geocode
    把整句原文当地址而失败；用户的改动指令由 state.query 原样交给 planner。
    若抽到了明确新目的地（如『改去青岛3天』），则保留抽取结果。
    """
    extracted_dest = (parsed.get("destination") or "").strip()
    if extracted_dest and extracted_dest != query:
        parsed["destination"] = extracted_dest
        return parsed
    if prior.get("destination"):
        parsed["destination"] = prior["destination"]
    if prior.get("days"):
        parsed["days"] = prior["days"]
    if prior.get("preference"):
        parsed["preference"] = prior["preference"]
    if prior.get("start_date") and not parsed.get("start_date"):
        parsed["start_date"] = prior["start_date"]
    return parsed


async def _prior_graph_state(session_id: str) -> dict:
    """读回同一会话上一轮 checkpointer 里持久化的行程上下文，供修改行程复用。

    返回 {"destination", "days", "preference", "start_date"}；读不到则返回空 dict
    （按新规划处理，不阻断）。checkpointer 按 thread_id=session_id 存短期记忆，
    与用户命名空间无关。
    """
    try:
        async with MemoryManager() as memory:
            tup = await memory.checkpointer.aget_tuple(
                {"configurable": {"thread_id": session_id}}
            )
            if tup is None:
                return {}
            cv = (tup.checkpoint or {}).get("channel_values") or {}
            return {
                "destination": cv.get("destination") or "",
                "days": cv.get("days") or 0,
                "preference": cv.get("preference") or "",
                "start_date": cv.get("start_date") or "",
            }
    except Exception as e:  # noqa: BLE001 - 读不到上一轮也不阻断
        logger.warning("读取上一轮行程状态失败（按新规划处理）：%s", e)
        return {}


async def _run_graph(graph_input, user_id: str, session_id: str, task_id: str, parsed: dict) -> None:
    """运行或续跑 workflow，并把中断/完成/错误状态写回 Redis。"""
    sm = get_session_manager()
    config = {"configurable": {"thread_id": session_id}}
    # 续跑时 graph_input 是 Command(resume=...) 而非 dict，这里只做防御性展示，避免日志反噬主流程
    if isinstance(graph_input, dict):
        _input_display = {k: v for k, v in graph_input.items() if k != "query"}
    else:
        _input_display = f"<{type(graph_input).__name__} resume>"
    logger.debug("[graph] 启动 workflow：thread_id=%s 输入=%s", session_id, _input_display)
    try:
        map_tools = await get_map_tools()
        async with MemoryManager(user_id=user_id) as memory:
            graph = build_workflow(map_tools, memory=memory)
            interrupts = []
            async for update in graph.astream(graph_input, config, stream_mode="updates"):
                for node_name, data in update.items():
                    if data is None:
                        continue
                    if node_name == "__interrupt__":
                        interrupts.extend(data)
                    elif isinstance(data, dict):
                        logger.debug("[graph] 节点 %s 输出：%s", node_name,
                                     {k: (v if not isinstance(v, list) or len(v) < 6 else f"<{len(v)} 项>")
                                      for k, v in data.items()})
            state_values = dict((await graph.aget_state(config)).values or {})
            if interrupts:
                logger.debug("[graph] 收到中断：kind=%s", (interrupts[0].value or {}).get("kind"))
                await sm.update_session(
                    user_id, session_id, task_id,
                    status="interrupted",
                    last_response={
                        "interrupt_data": interrupts[0].value,
                        "partial": _safe_result(state_values, parsed),
                    },
                )
                logger.info("任务 %s 中断，等待用户裁决", task_id)
            else:
                await sm.update_session(
                    user_id, session_id, task_id,
                    status="completed",
                    last_response={"result": _safe_result(state_values, parsed)},
                )
                logger.info("任务 %s 完成", task_id)
    except Exception as e:  # noqa: BLE001 - worker 里任何失败都要落为 error 状态
        logger.exception("agent 任务失败")
        await sm.update_session(
            user_id, session_id, task_id,
            status="error",
            last_response={"message": str(e)},
        )
    finally:
        await sm.close()


async def _run_resume(user_id: str, session_id: str, task_id: str, command_data: dict) -> None:
    """恢复被中断的任务。实体信息从中断时的 partial 里取回，避免最终结果丢 parsed。"""
    parsed = {}
    sm = get_session_manager()
    try:
        session = await sm.get_session_by_task(user_id, session_id, task_id)
        lr = (session or {}).get("last_response") or {}
        parsed = (lr.get("partial") or {}).get("parsed") or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("恢复 parsed 失败，使用空：%s", e)
    await sm.close()
    await _run_graph(Command(resume=command_data), user_id, session_id, task_id, parsed)


async def _travel_reply(query: str) -> str:
    """TRAVEL_QA / DESTINATION_RECOMMENDATION 分支：LLM 直接回答（不跑 workflow）。"""
    prompt = TRAVEL_QA_PROMPT.format(query=query)
    res = await acall_with_fallback([HumanMessage(content=prompt)])
    return str(res.content).strip()


async def _run_invoke(user_id: str, session_id: str, task_id: str, query: str) -> None:
    """意图识别 -> 路由分发：拒绝/直接回答/抽取实体后跑图。"""
    sm = get_session_manager()

    # 1) 意图识别
    try:
        intent_info = await classify_intent(query)
    except Exception as e:  # noqa: BLE001
        logger.warning("意图识别失败，默认按规划处理：%s", e)
        intent_info = {"intent": "TRAVEL_PLANNING", "reason": "识别失败默认规划", "route": "graph"}
    intent = intent_info.get("intent", "TRAVEL_PLANNING")
    logger.debug("[intent] query=%s -> intent=%s route=%s 原因=%s",
                 query[:40], intent, intent_info.get("route"), intent_info.get("reason", ""))

    # 2) 非旅游：直接礼貌拒绝，不走 workflow
    if intent_info.get("route") == "refuse":
        await sm.update_session(
            user_id, session_id, task_id,
            status="completed", last_query=query,
            last_response={
                "result": {
                    "intent": intent,
                    "reply": NON_TRAVEL_REPLY,
                    "is_non_travel": True,
                }
            },
        )
        await sm.close()
        logger.info("NON_TRAVEL 直接拒绝，task=%s", task_id)
        return

    # 3) 目的地推荐：LLM 输出结构化 JSON 卡片；失败回退纯文本回答
    if intent_info.get("route") == "recommend":
        try:
            rec = await recommend_destinations(query)
            cards = rec.get("destinations") or []
            # Phase 6：推荐卡片图片用真实 POI 图（text_search），失败回退占位图，不影响推荐主体
            try:
                map_tools = await get_map_tools()
                text_tool = next(
                    (t for t in map_tools if t.name == "maps_text_search"), None
                )
                if text_tool is not None:
                    cards = await fill_destination_images(cards, text_tool)
            except Exception as e:  # noqa: BLE001 - 图片失败忽略
                logger.warning("推荐卡片图片填充失败（忽略）：%s", e)
            await sm.update_session(
                user_id, session_id, task_id,
                status="completed", last_query=query,
                last_response={
                    "result": {
                        "intent": intent,
                        "recommendations": cards,
                        "reply": f"已为你推荐 {len(cards)} 个目的地，点击下方目的地按钮即可开始规划。",
                    }
                },
            )
        except Exception as e:  # noqa: BLE001 - 结构化失败不崩，回退纯文本回答
            logger.warning("结构化推荐失败，回退纯文本回答：%s", e)
            try:
                reply = await _travel_reply(query)
                await sm.update_session(
                    user_id, session_id, task_id,
                    status="completed", last_query=query,
                    last_response={"result": {"intent": intent, "reply": reply}},
                )
            except Exception as e2:  # noqa: BLE001
                logger.exception("推荐回退也失败")
                await sm.update_session(
                    user_id, session_id, task_id,
                    status="error", last_response={"message": str(e2)},
                )
        await sm.close()
        logger.info("目的地推荐完成，task=%s", task_id)
        return

    # 4) 旅游问答：LLM 直接回答
    if intent_info.get("route") == "reply":
        try:
            reply = await _travel_reply(query)
            await sm.update_session(
                user_id, session_id, task_id,
                status="completed", last_query=query,
                last_response={"result": {"intent": intent, "reply": reply}},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("直接回答失败")
            await sm.update_session(
                user_id, session_id, task_id,
                status="error", last_response={"message": str(e)},
            )
        await sm.close()
        logger.info("reply 意图 %s 已应答，task=%s", intent, task_id)
        return

    # 5) 规划 / 修改行程：抽取实体 -> 复用上一轮上下文（修改时）-> 写里程碑 -> 跑图
    try:
        parsed = await extract_travel_entities(query)
    except Exception as e:  # noqa: BLE001
        logger.warning("实体抽取失败，使用兜底：%s", e)
        parsed = {"destination": query, "days": 3, "preference": "",
                  "start_date": "", "end_date": ""}
    logger.debug("[entity] 抽取结果：目的地=%s 天数=%s 偏好=%s 起始=%s",
                 parsed.get("destination"), parsed.get("days"),
                 parsed.get("preference") or "无", parsed.get("start_date") or "无")

    # 修改行程：用户通常只说改动、不重复目的地。若没抽到明确的新目的地，
    # 就从同一会话上一轮的 checkpointer 复用 destination/days 等，避免
    # geocode 用整句原文当目的地而失败；用户的改动指令由 state.query 原样交给 planner。
    if intent == "ITINERARY_MODIFICATION":
        prior = await _prior_graph_state(session_id)
        logger.debug("[modify] 修改行程，读取上一轮上下文：%s", prior or "（无）")
        parsed = _merge_modification_context(parsed, prior, query)
        logger.debug("[modify] 合并后：目的地=%s 天数=%s", parsed["destination"], parsed["days"])

    parsed["intent"] = intent
    await sm.update_session(
        user_id, session_id, task_id,
        status="running", last_query=query, last_response={"parsed": parsed},
    )
    await sm.close()

    graph_input = {
        "destination": parsed["destination"],
        "days": parsed["days"],
        "preference": parsed.get("preference", ""),
        "start_date": parsed.get("start_date", ""),
        "query": query,
    }
    await _run_graph(graph_input, user_id, session_id, task_id, parsed)


@celery_app.task(name="invoke_agent_task")
def invoke_agent_task(user_id: str, session_id: str, task_id: str, query: str, system_message: str = None):
    _prepare_loop()
    asyncio.run(_run_invoke(user_id, session_id, task_id, query))


@celery_app.task(name="resume_agent_task")
def resume_agent_task(user_id: str, session_id: str, task_id: str, command_data: dict):
    _prepare_loop()
    asyncio.run(_run_resume(user_id, session_id, task_id, command_data))

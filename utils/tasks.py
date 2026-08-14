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
from langgraph.types import Command

from utils.config import Config
from utils.entity_extractor import extract_travel_entities
from utils.memory import MemoryManager
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
)


def _prepare_loop() -> None:
    """Windows 下 psycopg 异步需要 Selector 事件循环（默认 Proactor 不兼容）。"""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _safe_result(state: dict, parsed: dict) -> dict:
    """只保留 JSON 安全的关键字段，避免 messages(BaseMessage)/pois/weather 等序列化失败。"""
    keys = ("destination", "days", "preference", "itinerary",
            "memory_saved", "knowledge", "summary")
    result = {k: state.get(k) for k in keys if state.get(k) is not None}
    result["parsed"] = parsed
    return result


async def _run_graph(graph_input, user_id: str, session_id: str, task_id: str, parsed: dict) -> None:
    """运行或续跑 workflow，并把中断/完成/错误状态写回 Redis。"""
    sm = get_session_manager()
    config = {"configurable": {"thread_id": session_id}}
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
            state_values = dict((await graph.aget_state(config)).values or {})
            if interrupts:
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


async def _run_invoke(user_id: str, session_id: str, task_id: str, query: str) -> None:
    """抽取实体 -> 写里程碑 -> 跑图。"""
    try:
        parsed = await extract_travel_entities(query)
    except Exception as e:  # noqa: BLE001
        logger.warning("实体抽取失败，使用兜底：%s", e)
        parsed = {"destination": query, "days": 3, "preference": "",
                  "start_date": "", "end_date": ""}

    sm = get_session_manager()
    await sm.update_session(
        user_id, session_id, task_id,
        status="running", last_query=query, last_response={"parsed": parsed},
    )
    await sm.close()

    graph_input = {
        "destination": parsed["destination"],
        "days": parsed["days"],
        "preference": parsed.get("preference", ""),
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

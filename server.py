"""FastAPI 后端：Redis 会话状态 + Celery 异步任务调度。

- /agent/invoke : 提交一次对话式请求，立即返回 {user_id, session_id, task_id}
- /agent/resume : 恢复被 HITL 中断的任务（command 直接作为 Command(resume=...)）
- /agent/status/... 与 /agent/tasks/... : 按 task_id 随时查询状态与结果
- /agent/sessionids 与 /agent/active/sessionid : 历史会话 / 最近会话
- /session/ttl : 动态调整会话过期时间
- /agent/write/longterm : 写入用户长期记忆
- DELETE /agent/session|task : 删除会话/任务

会话状态存 Redis，图状态存 Postgres checkpointer -> 客户端/服务端故障后都可恢复。
"""
import asyncio
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Response

from utils import auth
from utils.config import Config
from utils.memory import MemoryManager
from utils.models import (
    ActiveSessionInfoResponse,
    AgentRequest,
    LoginRequest,
    LongMemRequest,
    RegisterRequest,
    ResumeRequest,
    SessionInfoResponse,
    SessionStatusResponse,
    SystemInfoResponse,
    TaskInfoResponse,
    TtlRequest,
)
from utils.render_html import export_filename, render_itinerary_export_html
from utils.session_manager import get_session_manager
from utils.tasks import invoke_agent_task, resume_agent_task

# Windows 下 psycopg 异步需要 Selector 事件循环
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 日志级别：LOG_LEVEL 环境变量（默认 DEBUG）。llm.py 的 basicConfig 可能已先配置根
# logger（第一次生效），因此这里不用 basicConfig，而是显式设级别 + 追加文件 handler。
_level = getattr(logging, Config.LOG_LEVEL, logging.DEBUG)
logging.getLogger().setLevel(_level)
_has_file = any(getattr(h, "baseFilename", None) == str(Config.LOG_FILE) for h in logging.getLogger().handlers)
if not _has_file:
    _fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(_fh)
logger = logging.getLogger("server")
logger.debug("server 日志已配置：级别=%s，文件=%s", Config.LOG_LEVEL, Config.LOG_FILE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session_manager = get_session_manager()
    await auth.init_user_store()
    logger.info("Redis 会话管理器 + 用户表初始化成功")
    yield
    await app.state.session_manager.close()
    await auth.close_user_store()
    logger.info("服务已关闭并清理资源")


app = FastAPI(
    title="TravelAssistant 后端（Redis 会话 + Celery 异步）",
    description="基于 LangGraph 的旅游行程规划 Agent 异步服务",
    lifespan=lifespan,
)


# ---------- 注册 / 登录 ----------
@app.post("/auth/register", response_model=dict)
async def register(request: RegisterRequest):
    logger.info("调用 /auth/register，username=%s", request.username)
    ok, msg, user = await auth.create_user(request.username, request.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "user_id": user.id, "username": user.username}


@app.post("/auth/login", response_model=dict)
async def login(request: LoginRequest):
    logger.info("调用 /auth/login，username=%s", request.username)
    user = await auth.verify_user(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    sm = app.state.session_manager
    active = ""
    if await sm.user_id_exists(user.id):
        active = await sm.get_user_active_session_id(user.id)
    return {
        "status": "success",
        "user_id": user.id,
        "username": user.username,
        "active_session_id": active,
    }


# ---------- 提交 / 恢复异步任务 ----------
@app.post("/agent/invoke", response_model=dict)
async def invoke_agent(request: AgentRequest):
    logger.info("调用 /agent/invoke，user=%s", request.user_id)
    user_id = request.user_id
    session_id = request.session_id or str(uuid.uuid4())
    task_id = request.task_id or str(uuid.uuid4())
    sm = app.state.session_manager

    if not await sm.session_task_id_exists(user_id, session_id, task_id):
        await sm.create_session(
            user_id=user_id, session_id=session_id, task_id=task_id,
            status="idle", last_updated=time.time(), ttl=request.ttl,
        )
    invoke_agent_task.delay(
        user_id=user_id, session_id=session_id, task_id=task_id,
        query=request.query, system_message=request.system_message,
    )
    await sm.set_task_status(task_id, "pending", user_id, session_id)
    return {"user_id": user_id, "session_id": session_id, "task_id": task_id}


@app.post("/agent/resume", response_model=dict)
async def resume_agent(response: ResumeRequest):
    logger.info("调用 /agent/resume，user=%s task=%s", response.user_id, response.task_id)
    sm = app.state.session_manager
    if not await sm.session_task_id_exists(response.user_id, response.session_id, response.task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    session = await sm.get_session_by_task(response.user_id, response.session_id, response.task_id)
    if session.get("status") != "interrupted":
        raise HTTPException(
            status_code=400,
            detail=f"当前状态为 {session.get('status')}，无法恢复非中断状态的会话",
        )
    resume_agent_task.delay(
        user_id=response.user_id, session_id=response.session_id,
        task_id=response.task_id, command_data=response.command,
    )
    await sm.set_task_status(response.task_id, "pending", response.user_id, response.session_id)
    return {"user_id": response.user_id, "session_id": response.session_id, "task_id": response.task_id}


# ---------- 查询 ----------
@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    sm = app.state.session_manager
    return SystemInfoResponse(
        sessions_count=await sm.get_session_count(),
        active_users=await sm.get_all_users_session_ids(),
    )


@app.get("/agent/active/sessionid/{user_id}", response_model=ActiveSessionInfoResponse)
async def get_agent_active_sessionid(user_id: str):
    sm = app.state.session_manager
    if not await sm.user_id_exists(user_id):
        return ActiveSessionInfoResponse(active_session_id="")
    return ActiveSessionInfoResponse(
        active_session_id=await sm.get_user_active_session_id(user_id)
    )


@app.get("/agent/sessionids/{user_id}", response_model=SessionInfoResponse)
async def get_agent_sessionids(user_id: str):
    sm = app.state.session_manager
    if not await sm.user_id_exists(user_id):
        return SessionInfoResponse(session_ids=[])
    return SessionInfoResponse(session_ids=await sm.get_all_session_ids(user_id))


@app.get("/agent/tasks/{user_id}/{session_id}", response_model=TaskInfoResponse)
async def get_agent_task_ids(user_id: str, session_id: str):
    sm = app.state.session_manager
    if not await sm.session_id_exists(user_id, session_id):
        return TaskInfoResponse(task_ids=[])
    return TaskInfoResponse(task_ids=await sm.get_task_status(user_id, session_id))


@app.get("/agent/status/{user_id}/{session_id}/{task_id}", response_model=SessionStatusResponse)
async def get_agent_status(user_id: str, session_id: str, task_id: str):
    sm = app.state.session_manager
    if not await sm.session_task_id_exists(user_id, session_id, task_id):
        return SessionStatusResponse(
            user_id=user_id, session_id=session_id, task_id=task_id,
            status="not_found", message="任务不存在",
        )
    session = await sm.get_session_by_task(user_id, session_id, task_id)
    return SessionStatusResponse(
        user_id=user_id, session_id=session_id, task_id=task_id,
        status=session.get("status"),
        last_query=session.get("last_query"),
        last_updated=session.get("last_updated"),
        last_response=session.get("last_response"),
    )


# ---------- HTML 导出（Phase 8）----------
@app.get("/agent/export/{user_id}/{session_id}/{task_id}")
async def export_agent_html(user_id: str, session_id: str, task_id: str):
    """导出已完成行程为独立 HTML 文件（附件下载）。

    由服务端用 render_html 纯函数渲染，导出文件只含业务字段，
    不含任何服务端 API Key（高德/LLM 等）。
    """
    sm = app.state.session_manager
    if not await sm.session_task_id_exists(user_id, session_id, task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    session = await sm.get_session_by_task(user_id, session_id, task_id)
    if (session or {}).get("status") != "completed":
        raise HTTPException(status_code=400, detail="行程尚未完成，请先完成审阅后再导出")
    result = ((session or {}).get("last_response") or {}).get("result") or {}
    data = result.get("itinerary_data") or {}
    if not (data.get("days") or []):
        raise HTTPException(status_code=400, detail="当前任务没有可导出的行程数据")
    html = render_itinerary_export_html(result)
    filename = export_filename(result.get("destination"))
    headers = {
        "Content-Disposition": (
            f"attachment; filename*=UTF-8''{quote(filename)}; filename=\"itinerary.html\""
        )
    }
    return Response(content=html, media_type="text/html; charset=utf-8", headers=headers)


# ---------- 长期记忆 / 会话管理 ----------
@app.post("/agent/write/longterm")
async def write_long_term(request: LongMemRequest):
    logger.info("调用 /agent/write/longterm，user=%s", request.user_id)
    sm = app.state.session_manager
    if not await sm.user_id_exists(request.user_id):
        raise HTTPException(status_code=404, detail="用户不存在")
    try:
        async with MemoryManager(user_id=request.user_id) as memory:
            await memory.write_user_memory(request.user_id, request.memory_info)
    except Exception as e:  # noqa: BLE001 - embedding/DB 故障要明确报错而非静默 500
        logger.warning("写入长期记忆失败：%s", e)
        raise HTTPException(status_code=400, detail=f"写入长期记忆失败：{e}")
    return {"status": "success", "message": "记忆写入成功"}


@app.post("/session/ttl")
async def set_session_ttl(request: TtlRequest):
    logger.info("调用 /session/ttl，user=%s session=%s ttl=%d",
                request.user_id, request.session_id, request.ttl)
    sm = app.state.session_manager
    affected = await sm.set_session_ttl(request.user_id, request.session_id, request.ttl)
    return {"status": "success", "ttl": request.ttl, "affected_tasks": affected}


@app.delete("/agent/session/{user_id}/{session_id}")
async def delete_agent_session(user_id: str, session_id: str):
    sm = app.state.session_manager
    if not await sm.session_id_exists(user_id, session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    await sm.delete_session(user_id, session_id)
    return {"status": "success", "message": f"会话 {session_id} 已删除"}


@app.delete("/agent/task/{user_id}/{session_id}/{task_id}")
async def delete_agent_task(user_id: str, session_id: str, task_id: str):
    sm = app.state.session_manager
    if not await sm.session_task_id_exists(user_id, session_id, task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    await sm.delete_session(user_id, session_id, task_id)
    return {"status": "success", "message": f"任务 {task_id} 已删除"}


if __name__ == "__main__":
    import uvicorn

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # loop="none"：让 uvicorn 不要重置事件循环策略。
    # 否则 Windows 下 uvicorn 会把策略重置为 Proactor，psycopg 异步连接会报错。
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, loop="none")

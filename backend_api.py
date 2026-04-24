import uuid
from fastapi import FastAPI
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from utils.agents import (
    model,
    embed_model,
    map_assistant,
    railway_assistant,
    flight_assistant,
    get_book_tools,
    SUSPERVISOR_PROMPT,
    Context
)
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from utils.config import config

DB_URI = config.DB_URI

# 全局变量
store = None
supervisor_agent = None


# =========================
# ✅ 生命周期管理（核心）
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, supervisor_agent

    async with (
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
        AsyncPostgresStore.from_conn_string(
            DB_URI,
            index={
                "dims": 768,
                "embed": embed_model,
                "fields": ["text"]
            }
        ) as _store
    ):
        await checkpointer.setup()
        await _store.setup()

        store = _store

        supervisor_agent = create_agent(
            model,
            tools=[map_assistant, railway_assistant, flight_assistant] + get_book_tools(),
            system_prompt=SUSPERVISOR_PROMPT,
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "book_hotel": True,
                        "book_railway": {"allowed_decisions": ["approve", "reject"]},
                        "book_flight": {"allowed_decisions": ["approve", "reject"]},
                    }
                )
            ],
            checkpointer=checkpointer,
            store=store
        )

        print("✅ Agent 初始化完成")

        yield  # 服务运行

        print("🔻 服务关闭")


app = FastAPI(lifespan=lifespan)


# =========================
# ✅ Chat
# =========================
@app.post("/chat")
async def chat(req: dict):
    user_id = req["user_id"]
    message = req["message"]
    thread_id = req.get("thread_id") or f"{user_id}_{uuid.uuid4()}"

    thread = {"configurable": {"thread_id": thread_id}}
    namespace = (user_id, "travel_preferences")

    item = await store.aget(namespace, "travel_preferences")
    memory = item.value["text"] if item else ""

    final_input = f"{message}，用户偏好：{memory}"

    response = ""
    interrupt = None

    async for chunk in supervisor_agent.astream(
        {"messages": [{"role": "user", "content": final_input}]},
        thread,
        context_schema=Context,
        stream_mode=["updates", "messages"],
        version="v2"
    ):
        if chunk["type"] == "messages":
            token, _ = chunk["data"]
            if token.content:
                response += token.content

        elif chunk["type"] == "updates":
            if "__interrupt__" in chunk["data"]:
                interrupt = chunk["data"]["__interrupt__"]

    return {
        "thread_id": thread_id,
        "response": response,
        "interrupt": interrupt
    }


# =========================
# ✅ Resume
# =========================
@app.post("/resume")
async def resume(req: dict):
    thread = {"configurable": {"thread_id": req["thread_id"]}}

    resume_cmd = {
        req["interrupt_id"]: {
            "decisions": req["decisions"]
        }
    }

    response = ""

    async for chunk in supervisor_agent.astream(
        {"resume": resume_cmd},
        thread,
        context_schema=Context,
        stream_mode=["messages"],
        version="v2"
    ):
        if chunk["type"] == "messages":
            token, _ = chunk["data"]
            if token.content:
                response += token.content

    return {"response": response}
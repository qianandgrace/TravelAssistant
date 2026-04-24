import os
from torch import chunk
from typing_extensions import runtime
import uuid
from dataclasses import dataclass
from typing import Any


from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.types import Command, interrupt, interrupt 
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore, logger  
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import HumanInTheLoopMiddleware

from utils.llm import get_llm, get_single_llm
from utils.tools import get_map_tools, get_railway_tools, get_flight_ticket_tools, get_book_tools
from utils.config import config
from utils.prompts import MAP_AGENT_PROMPT, SUSPERVISOR_PROMPT, MAP_AGENT_PROMPT, RAILWAY_AGENT_PROMPT, FLIGT_AGENT_PROMPT

# Initialize LLM and configs
model, embed_model= get_llm()
DB_URI = config.DB_URI

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

@dataclass
class Context:
    user_id: str

async def create_map_agent():
    return create_agent(
        model,
        tools=await get_map_tools(),
        system_prompt=MAP_AGENT_PROMPT
    )

async def create_railway_agent():
    return create_agent(
        model,
        tools=await get_railway_tools(),
        system_prompt=RAILWAY_AGENT_PROMPT
    )

async def create_flight_agent():
    return create_agent(
        model,
        tools=await get_flight_ticket_tools(),
        system_prompt=FLIGT_AGENT_PROMPT
    )


# define subagents as tools
@tool
async def map_assistant(request: str) -> str:
    """
    旅游行程规划助手
    """
    map_agent = await create_map_agent()
    result = await map_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })
    msg = result["messages"][-1]
    return msg.content if hasattr(msg, "content") else str(msg)


@tool
async def railway_assistant(request: str) -> str:
    """
    火车票预定助手
    """
    railway_agent = await create_railway_agent()
    result = await railway_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })
    msg = result["messages"][-1]
    return msg.content if hasattr(msg, "content") else str(msg)


@tool
async def flight_assistant(request: str) -> str:
    """
    机票预定助手
    """
    flight_agent = await create_flight_agent()
    result = await flight_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })
    msg = result["messages"][-1]
    return msg.content if hasattr(msg, "content") else str(msg)

async def   handle_single_interrupt(agent, interrupt, thread):
    actions = interrupt[0].value["action_requests"]

    decisions = []
    resume_cmd = {}

    for i, action in enumerate(actions):
        print(f"\n【待审批 {i+1}/{len(actions)}】")
        print("工具:", action["name"])
        print("参数:", action["args"])

        user_input = input("操作？(approve/edit/reject): ").strip()
        
        if user_input == "approve":
            decisions.append({"type": "approve"})

        elif user_input == "reject":
            decisions.append({"type": "reject"})

        elif user_input == "edit":
            new_val = input("新车次: ")
            decisions.append({
                "type": "edit",
                "args": {"train_number": new_val}
            })
        else:
            decisions.append({"type": "reject"})

    resume_cmd[interrupt[0].id] = {"decisions": decisions}

    # 👉 继续执行（关键）
    async for chunk in agent.astream(
            # {"messages": [{"role": "user", "content": f"{user_request_2}, {item.value['text']}"}]},
            Command(resume=resume_cmd),
            thread,
            context_schema=Context,
            stream_mode=["updates", "messages"],
            version="v2"
        ):
            if chunk["type"] == "messages":
            # LLM token
                token, metadata = chunk["data"]
                if token.content:
                    print(token.content, end="", flush=True)

# ======================
# ✅ 长期记忆提取
# ======================
async def extract_user_preference(text: str) -> str:
    prompt = f"""
请提取用户长期偏好（如时间/交通/景点），没有返回"无"：
{text}
"""
    res = await model.ainvoke([{"role": "user", "content": prompt}])
    return res.content.strip()


async def main():
    # 创建supervisor agent
    async with (AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
                AsyncPostgresStore.from_conn_string(DB_URI,
                                                    index={
                                                            "dims": 768,
                                                            "embed": embed_model,
                                                            "fields": ["text"]  # specify which fields to embed. Default is the whole serialized value
                                                        }) as store):
        await checkpointer.setup() # 确保表结构已创建
        await store.setup()
        supervisor_agent = create_agent(
            model,
            tools=[map_assistant, railway_assistant, flight_assistant] + get_book_tools(),
            system_prompt=SUSPERVISOR_PROMPT,
            middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "book_hotel": True,  # All decisions (approve, edit, reject) allowed
                "book_railway": {"allowed_decisions": ["approve", "reject"]},  # No editing allowed
                "book_flight": {"allowed_decisions": ["approve", "reject"]},  # No editing allowed
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
            # checkpointer=checkpointer,
            checkpointer=InMemorySaver(),  # 使用内存存储作为checkpointer调试用，实际应用中可以选择持久化的checkpointer
            store=store
        )
        # ✅ 用户登录
        user_id = input("请输入用户名：").strip()
        user = Context(user_id="gq")

        namespace = (user_id, "travel preferences")
      
    
        # ✅ 唯一 thread
        thread = {
            "configurable": {
                "thread_id": f"{user_id}_{uuid.uuid4()}"
            }
        }
        print("系统启动成功，开始对话（exit退出）")

        while True:
            user_input = input("\n用户: ").strip()
            if user_input.lower() == "exit":
                print("退出系统")
                break
            # 查询用户记忆
            # ===== 读取记忆 =====
            item = await store.aget(namespace, "travel_preference")
            memory_text = item.value["text"] if item else ""

            final_input = f"{user_input}，用户偏好：{memory_text}"
            async for chunk in supervisor_agent.astream(
                {"messages": [{"role": "user", "content": f"{final_input}"}]},
                thread,
                context_schema=Context,
                stream_mode=["updates", "messages"],
                version="v2"
            ):
                if chunk["type"] == "messages":
                # LLM token
                    token, metadata = chunk["data"]
                    if token.content:
                        print(token.content, end="", flush=True)
                elif chunk["type"] == "updates":
                # Check for interruptcions
                    if "__interrupt__" in chunk["data"]:
                        print(f"\n\nInterrupt: {chunk['data']['__interrupt__']}")
                        interrupts = chunk["data"]["__interrupt__"]
                        await handle_single_interrupt(supervisor_agent, interrupts, thread)
                
                  # ===== 写入长期记忆 =====
                pref = await extract_user_preference(user_input)
                if pref != "无":
                    print(f"写入用户偏好: {pref}")
                    await store.aput(namespace, "user_preference", {"text": pref})

if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
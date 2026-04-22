import os
from typing_extensions import runtime
import uuid
from dataclasses import dataclass


from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.types import Command 
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore  

from utils.llm import get_llm, get_single_llm
from utils.tools import get_map_tools, get_railway_tools, get_flight_ticket_tools
from utils.config import config
from utils.prompts import MAP_AGENT_PROMPT, SUSPERVISOR_PROMPT, MAP_AGENT_PROMPT, RAILWAY_AGENT_PROMPT, FLIGT_AGENT_PROMPT

# Initialize LLM and configs
model, embed_model= get_llm()
DB_URI = config.DB_URI

@dataclass
class Context:
    user_id: str

async def create_map_agent():
    return create_agent(
        model,
        tools=await get_map_tools(),
        system_prompt=MAP_AGENT_PROMPT,
    )

async def create_railway_agent():
    return create_agent(
        model,
        tools=await get_railway_tools(),
        system_prompt=RAILWAY_AGENT_PROMPT,
    )

async def create_flight_agent():
    return create_agent(
        model,
        tools=await get_flight_ticket_tools(),
        system_prompt=FLIGT_AGENT_PROMPT,
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
    return result["messages"][-1].text


@tool
async def railway_assistant(request: str) -> str:
    """
    火车票预定助手
    """
    railway_agent = await create_railway_agent()
    result = await railway_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
async def flight_assistant(request: str) -> str:
    """
    机票预定助手
    """
    flight_agent = await create_flight_agent()
    result = await flight_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

async def main():
    # 创建supervisor agent
    async with (AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
                AsyncPostgresStore.from_conn_string(DB_URI,
                                                    index={
                                                            "dims": 768,
                                                            "embed": embed_model,
                                                            "fields": ["text"]  # specify which fields to embed. Default is the whole serialized value
                                                        }) as store):
         # await checkpointer.setup() # 确保表结构已创建
        await store.setup()
        supervisor_agent = create_agent(
            model,
            tools=[map_assistant, railway_assistant, flight_assistant],
            system_prompt=SUSPERVISOR_PROMPT,
            checkpointer=checkpointer,
            store=store
        )
        user = Context(user_id="gq")
        user_id = user.user_id  
        namespace = (user_id, "travel preferences")
        await store.aput(namespace, "user_preference", {"text": "用户喜欢自然风光,坐车更喜欢上午的车次"})
        thread = {
            "configurable": {
                "thread_id": "2"
            }
        }
        # 查询用户记忆
        item = await store.aget(namespace, "user_preference")
        print("Retrieved user preference:", item)
        # 模拟用户多次请求 
        # user_request_1 = "你好，我叫gq"
        # async for step in supervisor_agent.astream(
        #     {"messages": [{"role": "user", "content": user_request_1}]},
        #     thread,
        #     context_schema=Context,
        # ):
        #     for update in step.values():
        #         for message in update.get("messages", []):
        #             message.pretty_print()

        # user_request_2 = "你好，我叫gq,我想在2026年5月1日上午从武汉出发，去北京旅游，5月4日返程，请帮我规划下行程，行程包含往返的车票"
        # async for step in supervisor_agent.astream(
        #     {"messages": [{"role": "user", "content": f"{user_request_2}, {item.value['text']}"}]},
        #     thread,
        #     context_schema=Context,
        # ):
        #     for update in step.values():
        #         for message in update.get("messages", []):
        #             message.pretty_print()
        
        user_request_3 = "你好，我叫什么名字？刚才问了什么问题？"
        async for step in supervisor_agent.astream(
            {"messages": [{"role": "user", "content": user_request_3}]},
            thread,
            context_schema=Context,
        ):
            for update in step.values():
                for message in update.get("messages", []):
                    message.pretty_print()


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
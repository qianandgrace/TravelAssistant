import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.types import Command 
from langgraph.checkpoint.postgres import PostgresSaver

from utils.llm import get_llm, get_single_llm
from utils.tools import get_map_tools, get_railway_tools, get_flight_ticket_tools
from utils.config import config
from utils.prompts import MAP_AGENT_PROMPT, SUSPERVISOR_PROMPT, MAP_AGENT_PROMPT, RAILWAY_AGENT_PROMPT, FLIGT_AGENT_PROMPT

# Initialize LLM
model = get_single_llm()

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
    supervisor_agent = create_agent(
        model,
        tools=[map_assistant, railway_assistant, flight_assistant],
        system_prompt=SUSPERVISOR_PROMPT)
    # 模拟用户单一意图请求  
    user_request = "帮我预定2026年4月28日北京到深圳的机票"
    async for step in supervisor_agent.astream(
        {"messages": [{"role": "user", "content": user_request}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
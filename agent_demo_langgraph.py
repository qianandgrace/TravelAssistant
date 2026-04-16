from typing_extensions import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain.messages import SystemMessage, HumanMessage, ToolMessage


from utils.llm import get_llm
from utils.tools import get_tools
import asyncio


llm, _ = get_llm("qwen")
tools = asyncio.run(get_tools())
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """你是一个旅游助手，善于利用相关的工具来帮助用户预定机票、酒店和火车票等旅游相关的服务。
你可以根据用户的需求调用相应的工具来完成任务。
如果需要调用工具，请使用正确的工具名称和参数格式进行调用。
"""

# Nodes
async def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    msg = await llm_with_tools.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [msg]}

async def tool_node(state: MessagesState):
    """Performs the tool call (async)"""
    last = state["messages"][-1]
    results = []
    # 这里按顺序执行；如果你想并发执行，我也可以给你 gather 版本
    for tool_call in last.tool_calls:  # type: ignore[attr-defined]
        tool = tools_by_name[tool_call["name"]]
        observation = await tool.ainvoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": results}

# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", END]: # type: ignore
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls: # type: ignore
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    """保存状态图的可视化表示。

    Args:
        graph: 状态图实例。
        filename: 保存文件路径。
    """
    # 尝试执行以下代码块
    try:
        # 以二进制写模式打开文件
        with open(filename, "wb") as f:
            # 将状态图转换为Mermaid格式的PNG并写入文件
            f.write(graph.get_graph().draw_mermaid_png()) # type: ignore
        # 记录保存成功的日志
        print(f"Graph visualization saved as {filename}")
    # 捕获IO错误
    except IOError as e:
        # 记录警告日志
        print.warning(f"Failed to save graph visualization: {e}")


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node) # type: ignore

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
save_graph_visualization(agent, "agent_graph.png")


# Invoke
async def main():
    messages = [HumanMessage(content="我在上海，4月2日要去北京，帮我查下这天上午8点到9点出发的高铁余票，并查询下北京那天的天气!")]
    out = await agent.ainvoke({"messages": messages})
    for m in out["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())
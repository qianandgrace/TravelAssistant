from dataclasses import dataclass

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


from utils.llm import get_llm

# define context structure to support dependency injection
@dataclass
class RuntimeContext:
    db: SQLDatabase

# define prompt
SYSTEM_PROMPT = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
- If the database is offline, ask user to try again later without further comment.
"""

# define tool
@tool
def execute_sql(query: str) -> str:
    """Execute a SQLite command and return results."""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


@tool
def execute_sql(query: str) -> str:
    """Execute a SQLite command and return results."""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

def save_graph_visualization(graph, filename: str = "graph.png") -> None:
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


llm, _ = get_llm("openai") 
db = SQLDatabase.from_uri("sqlite:///simple_demo/Chinook.db")  # 替换为你的数据库URI

### sql agent demo
agent = create_agent(
    model=llm,
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=RuntimeContext,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"execute_sql": {"allowed_decisions": ["approve", "reject"]}},
        ),
    ],
    checkpointer=InMemorySaver(),

)

question = "What are the names of all the employees?"

config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
    context=RuntimeContext(db=db)
)

if "__interrupt__" in result:
    description = result['__interrupt__'][-1].value['action_requests'][-1]['description']
    print(f"\033[1;3;31m{80 * '-'}\033[0m")
    print(
        f"\033[1;3;31m Interrupt:{description}\033[0m"
    )

    result = agent.invoke(
        Command(
            resume={
                "decisions": [{"type": "reject", "message": "the database is offline."}]
            }
        ),
        config=config,  # Same thread ID to resume the paused conversation
        context=RuntimeContext(db=db),
    )
    print(f"\033[1;3;31m{80 * '-'}\033[0m")

print(result["messages"][-1].content)

config = {"configurable": {"thread_id": "2"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
    context=RuntimeContext(db=db)
)

while "__interrupt__" in result:
    description = result['__interrupt__'][-1].value['action_requests'][-1]['description']
    print(f"\033[1;3;31m{80 * '-'}\033[0m")
    print(
        f"\033[1;3;31m Interrupt:{description}\033[0m"
    )
    
    result = agent.invoke(
        Command(
            resume={"decisions": [{"type": "approve"}]}
        ),
        config=config,  # Same thread ID to resume the paused conversation
        context=RuntimeContext(db=db),
    )

for msg in result["messages"]:
    msg.pretty_print()

### simple demo

# agent = create_agent(
#     model=llm,
#     system_prompt="你是一个全栈的戏剧演员",
# )

# result = agent.invoke({"messages": [{"role": "user", "content": "给我讲一个笑话"}]})
# print(result["messages"][1].content)

# for step in agent.stream(
#     {"messages": [{"role": "user", "content": "给我讲一个程序员的笑话"}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()


# for token, metadata in agent.stream(
#     {"messages": [{"role": "user", "content": "给我写一首适合冬天的诗"}]},
#     stream_mode="messages",
# ):
#     print(f"{token.content}", end="")
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool

from utils.utils import add_human_in_the_loop


######### mcp tools ##########
client = MultiServerMCPClient({
        # 12306 mcp key, 查询余票等功能
        "12306-streamableHTTP": {
            "url": "http://127.0.0.1:8166/mcp",
            "transport": "streamable_http",
        }
    })


@tool("book_hotel",description="需要人工审查/批准的预定酒店的工具")
def book_hotel(hotel_name: str):
    # 实际业务场景：处理酒店预定的业务逻辑
    return f"成功预定了在{hotel_name}的住宿"

@tool("book_railway",description="需要人工审查/批准的预定火车票的工具，模拟在线支付的流程，只要传入车次号即可预定成功")
def book_railway(train_number: str):
    # 实际业务场景：处理火车票预定的业务逻辑
    return f"成功预定了{train_number}次列车的车票"


async def get_tools():
    all_tools = await client.get_tools()
    # all_tools.append(book_railway)
    all_tools.append((await add_human_in_the_loop(book_railway)))
    # add_human_tools = [await add_human_in_the_loop(index) for index in all_tools[8:]]
    # tools = all_tools[:8] + add_human_tools
    return all_tools


if __name__ == "__main__":
    import asyncio

    async def main():
        tools = await get_tools()
        for t in tools:
            print(t)

    asyncio.run(main())
    

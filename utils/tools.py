import os

from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool

amap_key=os.getenv("AMAP_MAPS_API_KEY")
var_flight_key=os.getenv("VAR_FLIGHT")
######### mcp tools ##########
# 分别定义不同的工具函数，模拟不同的业务场景，用于构建不同的子agent
map_client = MultiServerMCPClient(
    {
        
        "map-mcp": {
            "transport": "streamable_http",
            "url":f"https://mcp.amap.com/mcp?key={amap_key}",
    
        },
        
    } # type: ignore
)

railway_tools = MultiServerMCPClient(
    {
        "12306-mcp": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "12306-mcp"]
        },
    }

)

flight_ticket_tools = MultiServerMCPClient(
    {
       "flight-ticket-mcp": {
            "transport": "streamable_http",
            "url": f"https://ai.variflight.com/servers/aviation/mcp/?api_key={var_flight_key}"
        } 
    }
)

@tool("book_railway",description="需要人工审查/批准的预定火车票的工具，模拟在线支付的流程，只要传入车次号即可预定成功")
def book_railway(train_number, date, seat_type):
    # 实际业务场景：处理火车票预定的业务逻辑
    return f"成功预定了{date}, {train_number}次列车{seat_type}的车票"

@tool("book_flight",description="需要人工审查/批准的预定机票的工具，模拟在线支付的流程，只要传入航班号即可预定成功")
def book_flight(flight_number: str, date: str, seat_class: str):
    # 实际业务场景：处理机票预定的业务逻辑
    return f"成功预定了{date}, {flight_number}次航班{seat_class}的机票"

@tool("book_hotel", description="需要人工审查/批准的预定酒店的工具，模拟在线支付的流程，只要传入酒店名称即可预定成功")
def book_hotel(hotel_name: str, date: str, nights: int):
    # 实际业务场景：处理酒店预定的业务逻辑
    return f"成功预定了{date}, {hotel_name}的房间{nights}晚"

async def get_map_tools():
    all_tools = await map_client.get_tools()
    return all_tools

async def get_railway_tools():
    all_tools = await railway_tools.get_tools()
    return all_tools

async def get_flight_ticket_tools():
    all_tools = await flight_ticket_tools.get_tools()
    return all_tools

def get_book_tools():
    return [book_railway, book_flight, book_hotel]  


if __name__ == "__main__":
    import asyncio

    async def main():
        map_tools = await get_map_tools()
        for t in map_tools:
            print(t)

        railway_tools = await get_railway_tools()
        for t in railway_tools:
            print(t)
        
        flight_ticket_tools = await get_flight_ticket_tools()
        for t in flight_ticket_tools:
            print(t)


    asyncio.run(main())
    

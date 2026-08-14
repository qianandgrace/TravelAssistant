"""高德（Amap）MCP 工具接入。

通过 MultiServerMCPClient 连接高德官方 MCP 服务（streamable_http 传输），
把高德暴露的 15 个地图能力包装成 LangChain 工具供 workflow 节点调用。
"""
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

# 加载项目根目录的 .env（tools 模块在 import 时就读取 AMAP_MAPS_API_KEY）
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# 高德开放平台 Web 服务 key（需在环境变量 AMAP_MAPS_API_KEY 中配置）
amap_key = os.getenv("AMAP_MAPS_API_KEY")

# 高德地图 MCP 客户端，暴露 geo/POI搜索/天气/路线规划等能力
map_client = MultiServerMCPClient(
    {
        "map-mcp": {
            "transport": "streamable_http",
            "url": f"https://mcp.amap.com/mcp?key={amap_key}",
        }
    }
)

async def get_map_tools():
    """获取高德 MCP 的全部工具（LangChain 工具对象，异步调用）。"""
    if not amap_key:
        raise ValueError("未设置 AMAP_MAPS_API_KEY 环境变量，无法连接高德 MCP。")
    all_tools = await map_client.get_tools()
    return all_tools


if __name__ == "__main__":
    import asyncio
    async def main():
        map_tools = await get_map_tools()
        for t in map_tools:
            print(f"- {t.name}: {t.description}")

    asyncio.run(main())

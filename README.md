# 项目背景
出行助手agent

# 方案选型
## mcp server
行程规划，高德mcp server
自驾路线，高德mcp server
火车余票查询，github拉取https://github.com/Joooook/12306-mcp.git
npx -y 12306-mcp --port 8166  远程http协议
npx -y 12306-mcp
飞机余票查询，github拉取git@github.com:xiaonieli7/FlightTicketMCP.git
set MCP_TRANSPORT=streamable-http
set MCP_HOST=127.0.0.1
set MCP_PORT=8016
python flight_ticket_server.py

# 方案部署

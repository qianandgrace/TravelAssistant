"""ReAct 架构旅行 agent：用 langchain.agents.create_agent 官方 API 构建。

与 utils/workflow.py 的「固定 StateGraph 流水线」对照的另一种主流架构：
LLM 在一个循环里自主决定调用哪个工具（Reason + Act），直到给出最终答案。

本模块只新增文件，**不改变默认 workflow 路径**（webapp/server 仍走 workflow）。
用途：与 workflow 做 10 问 token/时间/准确度对比基准（test/test_benchmark_workflow_vs_react.py），
也可以单独跑 `python utils/react_agent.py` 体验。
"""
import logging

from langchain.agents import create_agent
from langchain_core.tools import tool

from utils.llm import get_single_llm
from utils.research import research_summary, search_guides

logger = logging.getLogger(__name__)

# 中文旅行助手系统提示：覆盖现有四类意图（对齐 utils/intent_router.py 的类别）
SYSTEM_PROMPT = """你是旅行行程规划助手。根据用户的输入，在四种情况中采取相应行动：

1. 行程规划：当用户给出目的地和日期/天数时，规划逐日行程。
   - 先调用 geo / 文本搜索 / 周边搜索等工具，获取目的地的真实地点信息（坐标、POI、天气），
     需要时可以调用 travel_research 工具获取攻略研究摘要。
   - 用中文按天输出，格式「## 第 N 天　<当天主题>」，每天 3~5 个具体景点/活动，每个一句话并带可执行建议。
   - 尊重用户偏好（轻松/美食/亲子/爬山等），行程不要过满。

2. 目的地推荐：当用户说「不知道去哪」「推荐一下目的地」时，推荐 2~4 个具体目的地，
   每个一句话说明适合理由，并结合用户偏好（如亲子游）。

3. 旅游问答：用户询问目的地相关知识（签证、景点、美食等），直接用中文简洁准确地回答。

4. 非旅游请求：与旅行无关的请求（写程序、修 bug、无关闲聊），礼貌拒绝并说明只处理旅行相关问题，
   绝不尝试写代码或修 bug。

基于工具返回的真实数据作答，不要编造景点；工具没给出时说明缺少信息。"""

# 从高德 MCP 全量工具里精选的子集（避免 15 个全塞导致模型工具选择噪声）
_CURATED_TOOL_NAMES = {
    "maps_geo",            # 地址/地名 -> 经纬度
    "maps_text_search",    # 关键词 POI 搜索
    "maps_around_search",  # 坐标周边 POI
    "maps_weather",        # 天气
    "maps_direction_walking",  # 步行路线
}


@tool
async def travel_research(destination: str, preference: str = "", days: int = 0) -> str:
    """搜索并汇总指定目的地的攻略/游记来源，返回紧凑研究文本。

    Args:
        destination: 目的地（城市或地区名）。
        preference: 出行偏好，可空。
        days: 行程天数，可空（0 表示未指定）。
    """
    try:
        sr = await search_guides(destination, preference, days)
        sources = sr.get("sources") or []
        if not sources:
            return "未找到相关攻略来源（可能缺少搜索 key），请基于你自己的旅游知识规划。"
        summary = await research_summary(destination, preference, days, sources)
        text = str(summary.get("research_text") or "")
        src_lines = "\n".join(
            f"- {s.get('title', '')} {s.get('url', '')}" for s in sources[:6]
        )
        return f"【研究摘要】\n{text}\n\n【来源】\n{src_lines}"
    except Exception as e:  # noqa: BLE001 - 研究失败降级，不阻断 agent
        return f"攻略研究失败：{e}"


def _curate_map_tools(map_tools) -> list:
    """从全量高德工具里选出精选子集。"""
    return [t for t in map_tools if t.name in _CURATED_TOOL_NAMES]


def build_react_agent(map_tools=None, model=None, with_research: bool = True):
    """用 langchain.agents.create_agent 构建 ReAct 旅行 agent。

    Args:
        map_tools: get_map_tools() 返回的全量高德工具（内部精选子集）；None 则不带地图工具。
        model: BaseChatModel；默认 get_single_llm("qwen")（与 workflow 主模型一致）。
        with_research: 是否挂上 travel_research 攻略工具（对齐 workflow 的 do_research）。

    Returns:
        编译好的 CompiledStateGraph，用 `await agent.ainvoke({"messages": [("user", query)]})`
        调用，返回 state["messages"]，最后一条 AIMessage.content 即最终回答。
    """
    llm = model or get_single_llm("qwen")
    tools: list = []
    if map_tools:
        curated = _curate_map_tools(map_tools)
        logger.debug("ReAct agent 挂载 %d 个地图工具：%s", len(curated),
                     [t.name for t in curated])
        tools += curated
    if with_research:
        tools.append(travel_research)
    if not tools:
        logger.warning("ReAct agent 没有任何工具，仅作纯文本回答。")
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT,
                         name="react_travel_agent")
    return agent


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def main():
        from utils.tools import get_map_tools
        map_tools = await get_map_tools()
        agent = build_react_agent(map_tools=map_tools)
        state = await agent.ainvoke({
            "messages": [{"role": "user", "content": "上海今天天气怎么样？"}]})
        print("回答：", state["messages"][-1].content)

    asyncio.run(main())

"""用户输入 -> 意图识别（Intent Router）。

5 类意图：
  TRAVEL_PLANNING           规划具体行程（有目的地/日期/天数）
  DESTINATION_RECOMMENDATION 不知道去哪，需要推荐目的地
  TRAVEL_QA                 旅游相关问题，直接回答即可
  ITINERARY_MODIFICATION    修改已生成的行程
  NON_TRAVEL                与旅游无关，礼貌拒绝

识别以 LLM 为主（不靠关键词判断），LLM 失败时用极简规则兜底。
"""
import json
import logging
import re

from langchain_core.messages import HumanMessage

from utils.llm import acall_with_fallback
from utils.prompts import INTENT_ROUTER_PROMPT

logger = logging.getLogger(__name__)

INTENTS = (
    "TRAVEL_PLANNING",
    "DESTINATION_RECOMMENDATION",
    "TRAVEL_QA",
    "ITINERARY_MODIFICATION",
    "NON_TRAVEL",
)

# 各意图对应的处理路径：
#   graph=跑规划 workflow；reply=LLM 直接回答；recommend=结构化目的地推荐；refuse=直接拒绝
ROUTE_FOR = {
    "TRAVEL_PLANNING": "graph",
    "ITINERARY_MODIFICATION": "graph",
    "TRAVEL_QA": "reply",
    "DESTINATION_RECOMMENDATION": "recommend",
    "NON_TRAVEL": "refuse",
}

NON_TRAVEL_REPLY = (
    "抱歉，我是旅游行程规划助手，只处理与出行相关的问题。"
    "我可以帮你：规划具体行程、推荐出行目的地、回答旅游问题、调整已生成的行程。"
    "与旅游无关的问题我无法处理，请换个出行相关的问题试试。"
)


def _strip_json(text: str) -> str:
    """去掉 ```json ... ``` 围栏，取第一个花括号对象。"""
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else text


def _fallback_intent(query: str) -> str:
    """LLM 不可用时兜底。宁可偏宽（默认规划），也别把旅游请求拒掉。

    注意：这只是兜底，正常情况下由 LLM 分类（不靠关键词）。
    """
    q = query.strip()
    # 明显的非旅游信号：编程/技术/办公事务
    if re.search(
        r"写.{0,8}(代码|程序|脚本)|报错|bug|接口|函数|算法|数据库|sql|java|python|javascript",
        q,
        re.I,
    ):
        return "NON_TRAVEL"
    # 明显的推荐目的地意图
    if re.search(
        r"(不知道去哪|推荐.{0,6}目的地|有什么.{0,8}(值得去|适合|好玩|好逛|推荐).{0,6}(地方|城市|目的地)|去哪(里)?(玩|旅游|玩好)|哪里好玩|求推荐)",
        q,
    ):
        return "DESTINATION_RECOMMENDATION"
    # 明显的攻略/问答类问句
    if re.search(
        r"(值得|怎么|如何|攻略|好玩吗|好吃吗|需要.{0,4}吗|有什么.{0,6}(好玩|好吃|值得去|景点)|签证|通行|注意什么|[?？]\s*$)",
        q,
    ):
        return "TRAVEL_QA"
    # 明显的行程修改意图
    if re.search(
        r"(删|去掉|换掉|减|加|增加|调整|改|太累|不想去|不去了|缩短|延长).{0,12}(景点|行程|安排|天|项目|餐厅)",
        q,
    ):
        return "ITINERARY_MODIFICATION"
    return "TRAVEL_PLANNING"


async def classify_intent(query: str) -> dict:
    """识别用户输入意图。

    Returns:
        {"intent": str, "reason": str, "route": "graph"|"reply"|"refuse"}
    """
    intent, reason = None, ""
    try:
        prompt = INTENT_ROUTER_PROMPT.format(query=(query or "").strip())
        text = str((await acall_with_fallback([HumanMessage(content=prompt)])).content)
        data = json.loads(_strip_json(text))
        if isinstance(data, dict):
            candidate = str(data.get("intent") or "").strip()
            if candidate in INTENTS:
                intent, reason = candidate, str(data.get("reason") or "").strip()
    except Exception as e:  # noqa: BLE001 - 失败走兜底规则
        logger.warning("意图识别 LLM 失败，使用兜底规则：%s", e)

    if intent is None:
        intent = _fallback_intent(query or "")
        reason = "LLM 分类失败，规则兜底"

    return {"intent": intent, "reason": reason, "route": ROUTE_FOR[intent]}

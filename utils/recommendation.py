"""目的地推荐（DESTINATION_RECOMMENDATION 分支）。

用户没有目的地时，用 LLM 生成 3~6 个结构化目的地卡片：
  {destination, country, reason, best_for, recommended_days,
   estimated_budget_level, image, highlights}

流程：LLM 输出严格 JSON -> 后端校验 -> 非法则带原文 repair 一次 -> 仍失败抛异常，
由调用方（utils/tasks.py）回退到纯文本回答，保证流程不崩。
"""
import datetime
import json
import logging
import re

from langchain_core.messages import HumanMessage

from utils.llm import acall_with_fallback
from utils.prompts import (
    DESTINATION_RECOMMENDATION_PROMPT,
    DESTINATION_REPAIR_PROMPT,
)

logger = logging.getLogger(__name__)

MIN_RECS, MAX_RECS = 3, 6

CARD_KEYS = (
    "destination", "country", "reason", "best_for",
    "recommended_days", "estimated_budget_level", "image", "highlights",
)


def _strip_json(text: str) -> str:
    """去掉 ```json ... ``` 围栏，取第一个花括号对象。"""
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else text


def _normalize_item(item) -> dict | None:
    """把单个目的地卡片规整成标准结构；不合法返回 None。"""
    if not isinstance(item, dict):
        return None
    destination = str(item.get("destination") or "").strip()
    if not destination:
        return None
    try:
        days = int(item.get("recommended_days") or 0)
    except (TypeError, ValueError):
        days = 0
    if not (0 < days <= 30):
        days = 3
    highlights = item.get("highlights") or []
    if isinstance(highlights, str):
        highlights = re.split(r"[；;、\n]", highlights)
    highlights = [str(h).strip() for h in highlights if str(h).strip()][:5]
    return {
        "destination": destination,
        "country": str(item.get("country") or "").strip() or "中国",
        "reason": str(item.get("reason") or "").strip(),
        "best_for": str(item.get("best_for") or "").strip(),
        "recommended_days": days,
        "estimated_budget_level": str(item.get("estimated_budget_level") or "").strip() or "中等",
        "image": str(item.get("image") or "").strip(),
        "highlights": highlights,
    }


def _validate(data) -> list:
    """校验 LLM 输出结构，返回规整后的目的地列表；非法抛 ValueError。"""
    if not isinstance(data, dict):
        raise ValueError("推荐结果不是 JSON 对象")
    items = data.get("destinations")
    if not isinstance(items, list):
        raise ValueError("缺少 destinations 数组")
    if not (MIN_RECS <= len(items) <= MAX_RECS):
        raise ValueError(f"目的地数量应为 {MIN_RECS}~{MAX_RECS}，实际 {len(items)}")
    cards = [_normalize_item(it) for it in items]
    if any(c is None for c in cards):
        raise ValueError("存在不合法目的地卡片")
    return [c for c in cards if c]


def _parse_cards(text: str) -> list:
    return _validate(json.loads(_strip_json(text)))


async def recommend_destinations(query: str, today: str = "") -> dict:
    """推荐目的地。返回 {"destinations": [...]}；JSON 两次都失败则抛异常。"""
    today = today or datetime.date.today().isoformat()
    prompt = DESTINATION_RECOMMENDATION_PROMPT.format(query=query, today=today)
    res = await acall_with_fallback([HumanMessage(content=prompt)])
    text = str(res.content).strip()

    try:
        cards = _parse_cards(text)
        return {"destinations": cards, "raw": text}
    except Exception as e:  # noqa: BLE001 - 第一次解析失败，带原文让 LLM 修复一次
        logger.warning("推荐 JSON 解析失败，尝试修复：%s", e)
        fix_prompt = DESTINATION_REPAIR_PROMPT.format(
            query=query, today=today, invalid=text, error=str(e)
        )
        res2 = await acall_with_fallback([HumanMessage(content=fix_prompt)])
        text2 = str(res2.content).strip()
        try:
            cards = _parse_cards(text2)
            return {"destinations": cards, "raw": text2}
        except Exception as e2:  # noqa: BLE001
            logger.error("推荐 JSON 修复失败：%s", e2)
            raise ValueError(f"目的地推荐 JSON 无法解析：{e2}") from e2

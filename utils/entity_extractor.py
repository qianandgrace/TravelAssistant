"""对话式输入 -> 关键实体抽取（目的地 / 起止日期 / 天数 / 偏好）。

把用户的一句话（如「9月2号到9月5号去天津，轻松点」）用 LLM 抽成结构化的
{destination, days, preference, start_date, end_date}，供 workflow 作为输入。
"""
import json
import logging
import re
from datetime import date, datetime
from typing import Optional

from langchain_core.messages import HumanMessage

from utils.llm import acall_with_fallback
from utils.prompts import ENTITY_EXTRACTOR_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 3


def _strip_json_fence(text: str) -> str:
    """去掉 ```json ... ``` 等围栏，取第一个花括号对象。"""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return m.group(0)
    return text


def _parse_date(s) -> Optional[date]:
    """解析多种日期写法，返回 date 或 None。"""
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    m = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{1,2})月(\d{1,2})[号日]?", s)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = date.today().year
        d = date(year, month, day)
        if d < date.today():  # 该日期今年已过，推为明年
            d = date(year + 1, month, day)
        return d
    return None


def _as_int(value, default: int = DEFAULT_DAYS) -> int:
    try:
        n = int(value)
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default


def _compute_days(start: Optional[date], end: Optional[date], raw_days) -> int:
    if start and end and end >= start:
        return (end - start).days + 1
    return _as_int(raw_days)


async def extract_travel_entities(query: str) -> dict:
    """用 LLM 抽取关键实体；解析失败时兜底（目的地=原文，天数=3）。

    Returns:
        {"destination": str, "days": int, "preference": str,
         "start_date": str, "end_date": str}
    """
    prompt = ENTITY_EXTRACTOR_PROMPT.format(query=query, today=date.today().isoformat())
    text = str((await acall_with_fallback([HumanMessage(content=prompt)])).content)
    try:
        data = json.loads(_strip_json_fence(text))
        if not isinstance(data, dict):
            raise ValueError("不是 JSON 对象")
    except Exception as e:  # noqa: BLE001
        logger.warning("实体抽取解析失败，使用兜底：%s | 原文：%s", e, text[:200])
        data = {}

    destination = str(data.get("destination") or "").strip() or query
    start = _parse_date(data.get("start_date"))
    end = _parse_date(data.get("end_date"))
    days = _compute_days(start, end, data.get("days"))
    preference = str(data.get("preference") or "").strip()

    return {
        "destination": destination,
        "days": days,
        "preference": preference,
        "start_date": start.isoformat() if start else "",
        "end_date": end.isoformat() if end else "",
    }

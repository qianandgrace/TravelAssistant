"""Travel Research Pipeline：攻略/游记来源研究。

设计原则（用户要求：不编造无来源的数据）：
- 若 .env 配置了 TAVILY_API_KEY，则真实搜索「目的地 + 旅游攻略」，得到来源（标题/URL/片段）。
- 否则 search_available=False，sources 为空，绝不编造来源/链接；
  仅用 LLM 知识 + 高德 POI 上下文生成紧凑的研究摘要，前端不显示"参考了 N 个攻略来源"。
- 无论哪种模式，产出紧凑的 TRAVEL_RESEARCH_SUMMARY 结构（非整篇文章），
  只把结论喂给行程规划 prompt，避免把整篇攻略塞进上下文。
"""
import datetime
import json
import logging
import os
import re

import requests
from langchain_core.messages import HumanMessage

from utils.llm import acall_with_fallback
from utils.prompts import TRAVEL_RESEARCH_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
DEFAULT_MAX_SOURCES = 6  # 3~8 之间


def _strip_json(text: str) -> str:
    """去掉 ```json ... ``` 围栏，取第一个花括号对象。"""
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else text


async def _tavily_search(query: str, max_results: int = DEFAULT_MAX_SOURCES) -> list[dict]:
    """调用 Tavily 搜索攻略。未配置 key 或失败返回空列表（不阻断主流程）。"""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return []
    try:
        r = requests.post(
            TAVILY_URL,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "search_depth": "basic",
            },
            timeout=20,
        )
        r.raise_for_status()
        out = []
        for item in r.json().get("results", [])[:max_results]:
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if title and url:
                out.append({
                    "title": title,
                    "url": url,
                    "content": str(item.get("content") or "").strip()[:500],
                })
        logger.info("攻略搜索到 %d 条来源", len(out))
        return out
    except Exception as e:  # noqa: BLE001 - 搜索失败不影响规划
        logger.warning("攻略搜索失败（跳过）：%s", e)
        return []


def _build_search_query(destination: str, preference: str = "", days: int = 0) -> str:
    q = f"{destination} 旅游攻略 行程 游记"
    if preference:
        q += f" {preference}"
    if days:
        q += f" {days}天"
    return q


async def search_guides(
    destination: str, preference: str = "", days: int = 0,
    max_results: int = DEFAULT_MAX_SOURCES,
) -> dict:
    """搜索攻略来源。返回 {"sources": [{title,url,content}], "search_available": bool}。"""
    items = await _tavily_search(_build_search_query(destination, preference, days), max_results)
    if not items:
        return {"sources": [], "search_available": False}
    return {"sources": items, "search_available": True}


async def research_summary(
    destination: str, preference: str, days: int,
    sources: list, today: str = "",
) -> dict:
    """生成 TRAVEL_RESEARCH_SUMMARY 结构。失败返回空结构（不阻断规划）。

    sources 字段由系统注入（真实搜索来源或空），LLM 只管研究结论，避免编造来源。
    """
    today = today or datetime.date.today().isoformat()
    if sources:
        src_lines = "\n".join(f"- {s['title']}  {s['url']}" for s in sources)
        sources_note = "以下是搜索到的攻略/游记来源（请基于它们归纳）：\n" + src_lines
    else:
        sources_note = (
            "本次没有外部攻略来源。请基于你的旅游知识整理研究摘要，"
            "不得编造任何来源标题或链接。"
        )
    prompt = TRAVEL_RESEARCH_SUMMARY_PROMPT.format(
        destination=destination, days=days,
        preference=preference or "（未提供）",
        sources_note=sources_note, today=today,
    )
    res = await acall_with_fallback([HumanMessage(content=prompt)])
    text = str(res.content).strip()
    try:
        data = json.loads(_strip_json(text))
        if not isinstance(data, dict):
            raise ValueError("不是 JSON 对象")
    except Exception as e:  # noqa: BLE001 - 摘要失败不阻断规划
        logger.warning("研究摘要解析失败，返回空结构：%s", e)
        data = {}
    # sources 一律用真实来源，防止 LLM 乱写
    data["sources"] = [{"title": s["title"], "url": s["url"]} for s in sources]
    return data

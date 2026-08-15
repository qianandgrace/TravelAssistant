"""图片获取与占位回退（Phase 6）。

原则（用户要求）：
- 图片来自 POI/搜索 API（高德 text_search / around_search 返回的 photo 字段），
  绝不由 LLM 编造 URL。
- 任何环节拿不到真实图片 -> 回退本地占位图（内联 SVG data-URI，无需外部服务）。
- 前端渲染用 <img loading="lazy"> 懒加载（Phase 7 前端 / Phase 8 HTML 导出）。
"""
import asyncio
import json
import logging
import urllib.parse

logger = logging.getLogger(__name__)

# 回退占位图：内联 SVG data-URI（灰底 + "图片加载中"），无网络依赖、可离线显示
_PLACEHOLDER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">'
    '<rect width="100%" height="100%" fill="#eef1f5"/>'
    '<text x="50%" y="50%" font-family="sans-serif" font-size="30" fill="#9aa5b1" '
    'text-anchor="middle" dominant-baseline="middle">图片加载中</text></svg>'
)
PLACEHOLDER_IMAGE = "data:image/svg+xml;charset=utf-8," + urllib.parse.quote(_PLACEHOLDER_SVG)

# 参与图片解析的 item 类型（有固定物理位置的真实地点，与路线计算保持一致）
PLACE_TYPES = {"景点", "美食", "酒店"}

# text_search 兜底单次最多尝试多少个未命中项（防滥用/控时）
MAX_FALLBACK_SEARCH = 12


def _extract_text(content_blocks) -> str:
    """MCP 工具返回 [{'type':'text','text':'...'}]，取出文本。"""
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return str(content_blocks)


def _parse_json(content_blocks) -> dict:
    return json.loads(_extract_text(content_blocks))


def _first_photo(pois: list) -> str:
    for p in pois or []:
        photo = str(p.get("photo") or "").strip()
        if photo:
            return photo
    return ""


async def _text_search_photo(text_search_tool, keywords: str, city: str) -> str:
    """用高德文本搜索取真实 POI 图片；失败返回空串。"""
    if not keywords:
        return ""
    try:
        res = await text_search_tool.ainvoke({"keywords": keywords, "city": city})
        pois = _parse_json(res).get("pois", []) or []
        return _first_photo(pois)
    except Exception as e:  # noqa: BLE001 - 单点图片失败不影响整体
        logger.warning("图片搜索失败(%s)：%s", keywords, e)
        return ""


async def fill_item_images(items: list, pois_by_id: dict, text_search_tool, city: str,
                           max_search: int = MAX_FALLBACK_SEARCH) -> list:
    """给行程 items 填充 image：
        1) poi_id 命中搜索结果 -> 直接用该 POI 的 photo（零额外调用）
        2) 未命中/无图 -> text_search 并发兜底（限次）
        3) 仍未拿到 -> 回退占位图
    只处理有固定位置的地点类型（景点/美食/酒店），其余 image 保持空串。
    """
    missing = []
    for item in items:
        if item.get("type") not in PLACE_TYPES:
            item["image"] = ""
            continue
        photo = ""
        pid = item.get("poi_id")
        if pid and pid in pois_by_id:
            photo = str(pois_by_id[pid].get("photo") or "").strip()
        if photo:
            item["image"] = photo
        else:
            missing.append(item)
    if missing and text_search_tool is not None:
        bounded = missing[:max_search]
        photos = await asyncio.gather(
            *[_text_search_photo(text_search_tool, it.get("name", ""), city) for it in bounded],
            return_exceptions=True,
        )
        for it, ph in zip(bounded, photos):
            if isinstance(ph, str) and ph:
                it["image"] = ph
    for item in items:
        if item.get("type") in PLACE_TYPES and not item.get("image"):
            item["image"] = PLACEHOLDER_IMAGE
    return items


async def fill_destination_images(cards: list, text_search_tool,
                                  max_search: int = MAX_FALLBACK_SEARCH) -> list:
    """给目的地推荐卡片填充 image（真实 POI 图片）；失败回退占位图。"""
    if text_search_tool is None:
        for c in cards:
            c["image"] = PLACEHOLDER_IMAGE
        return cards
    bounded = cards[:max_search]
    photos = await asyncio.gather(
        *[_text_search_photo(text_search_tool, c.get("destination", ""), c.get("destination", ""))
          for c in bounded],
        return_exceptions=True,
    )
    for card, ph in zip(bounded, photos):
        if isinstance(ph, str) and ph:
            card["image"] = ph
    for card in cards:
        if not card.get("image"):
            card["image"] = PLACEHOLDER_IMAGE
    return cards

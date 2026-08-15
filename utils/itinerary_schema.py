"""行程 JSON Schema：解析 -> 校验/规整 -> 非法则触发 repair/retry。

设计原则（用户要求）：
- LLM 只做"选点与排期"，坐标/路线/图片由地图服务后续计算——本模块把
  latitude/longitude 强制为 None、image 强制为空串、route 归一为空。
- parse -> validate/normalize：结构非法返回 errors，由调用方带原文修复一次，
  仍失败则回退文本版行程（保证用户一定拿到可读行程）。
- render_itinerary_md() 把结构化数据确定性渲染成 Markdown，
  供现有前端/HITL 中断继续展示（Phase 7 前端将直接用结构化数据）。
"""
import json
import re
from datetime import date, timedelta

ALLOWED_ITEM_TYPES = {"景点", "美食", "酒店", "交通", "自由活动", "其他"}
MIN_ITEMS_PER_DAY = 3
MAX_ITEMS_PER_DAY = 12


def parse_itinerary_json(text: str) -> tuple[dict | None, str]:
    """解析 LLM 输出的 JSON。返回 (data, None) 或 (None, error)。"""
    m = re.search(r"\{.*\}", (text or "").strip(), re.S)
    if not m:
        return None, "输出中找不到 JSON 对象"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:  # noqa: BLE001
        return None, f"JSON 解析失败：{e}"
    if not isinstance(data, dict):
        return None, "顶层不是 JSON 对象"
    return data, None


def _coerce_float(v):
    try:
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def _coerce_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _norm_item(item) -> dict | None:
    """规整单个 item；缺 name 视为非法返回 None。"""
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    if not name:
        return None
    itype = str(item.get("type") or "").strip()
    if itype not in ALLOWED_ITEM_TYPES:
        itype = "其他"
    duration = _coerce_int(item.get("duration_minutes"), 0)
    if duration < 0:
        duration = 0
    return {
        "type": itype,
        "name": name,
        "start_time": str(item.get("start_time") or "").strip(),
        "end_time": str(item.get("end_time") or "").strip(),
        "duration_minutes": duration,
        "reason": str(item.get("reason") or "").strip(),
        "address": str(item.get("address") or "").strip(),
        "latitude": _coerce_float(item.get("latitude")),
        "longitude": _coerce_float(item.get("longitude")),
        "image": str(item.get("image") or "").strip(),
        "poi_id": str(item.get("poi_id") or "").strip() or None,
    }


def validate_and_normalize(data, expected_days: int, start_date: str = "") -> tuple[dict, list[str]]:
    """校验并规整行程 JSON。返回 (规整后的数据, errors)；errors 为空则合法。

    - days 数组长度必须等于 expected_days
    - 每天 items 至少 MIN_ITEMS_PER_DAY 个、每个必须有 name
    - date 若给定 start_date 则强制按天递增补齐
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return {"days": [], "tips": []}, ["顶层不是 JSON 对象"]

    days = data.get("days")
    if not isinstance(days, list) or not days:
        return {"days": [], "tips": []}, ["缺少 days 数组"]
    if len(days) != expected_days:
        errors.append(f"days 数组长度应为 {expected_days}，实际 {len(days)}")

    base = None
    if start_date:
        try:
            base = date.fromisoformat(start_date)
        except ValueError:
            base = None

    norm_days = []
    for idx, raw in enumerate(days):
        raw = raw if isinstance(raw, dict) else {}
        day = _coerce_int(raw.get("day"), idx + 1)
        date_str = str(raw.get("date") or "").strip()
        if base and date_str != (base + timedelta(days=idx)).isoformat():
            date_str = (base + timedelta(days=idx)).isoformat()

        items = raw.get("items")
        if not isinstance(items, list):
            errors.append(f"第 {day} 天缺少 items 数组")
            items = []
        norm_items = []
        for it in items:
            ni = _norm_item(it)
            if ni is None:
                errors.append(f"第 {day} 天存在非法 item（缺 name）")
            else:
                norm_items.append(ni)
        if len(norm_items) < MIN_ITEMS_PER_DAY:
            errors.append(f"第 {day} 天 items 少于 {MIN_ITEMS_PER_DAY} 个")
        if len(norm_items) > MAX_ITEMS_PER_DAY:
            norm_items = norm_items[:MAX_ITEMS_PER_DAY]

        route = raw.get("route") if isinstance(raw.get("route"), dict) else {}
        norm_days.append({
            "day": day,
            "date": date_str,
            "theme": str(raw.get("theme") or "").strip(),
            "weather": str(raw.get("weather") or "").strip(),
            "items": norm_items,
            "route": {
                "points": list(route.get("points") or []) if isinstance(route.get("points"), list) else [],
                "distance_km": _coerce_float(route.get("distance_km")) or 0.0,
                "estimated_minutes": _coerce_int(route.get("estimated_minutes"), 0),
            },
        })

    tips = data.get("tips") or []
    if isinstance(tips, str):
        tips = [t.strip() for t in re.split(r"[；;、\n]", tips) if t.strip()]
    tips = [str(t).strip() for t in tips if str(t).strip()][:10]

    return {"days": norm_days, "tips": tips}, errors


def render_itinerary_md(data: dict) -> str:
    """把结构化行程确定性渲染成 Markdown（现有前端/HITL 展示用，非 LLM）。"""
    days = data.get("days") or []
    lines = []
    for day in days:
        header = f"## 第 {day.get('day')} 天"
        if day.get("theme"):
            header += f"　{day['theme']}"
        lines.append(header)
        meta = []
        if day.get("date"):
            meta.append(f"日期：{day['date']}")
        if day.get("weather"):
            meta.append(f"天气：{day['weather']}")
        if meta:
            lines.append("　".join(meta))
        for item in day.get("items") or []:
            t = item.get("type", "")
            span = f"{item.get('start_time', '')}~{item.get('end_time', '')}"
            head = f"- [{t}] {item.get('name', '')}"
            if span.strip("~"):
                head += f"（{span}）"
            lines.append(head)
            if item.get("reason"):
                lines.append(f"    {item['reason']}")
            if item.get("address"):
                lines.append(f"    地址：{item['address']}")
        route = day.get("route") or {}
        if route.get("distance_km"):
            lines.append(f"    路线约 {route['distance_km']} km，{route['estimated_minutes']} 分钟")
        lines.append("")
    tips = data.get("tips") or []
    if tips:
        lines.append("**小贴士**")
        lines.extend(f"- {t}" for t in tips)
    sources = data.get("sources") or []
    if sources:
        lines.append(f"\n**本行程参考了 {len(sources)} 个攻略/游记来源**")
    return "\n".join(lines).strip()

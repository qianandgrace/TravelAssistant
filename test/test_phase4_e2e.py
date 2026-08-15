"""Phase 4 验证：Planner 重设计 + 行程 JSON Schema。

  1. 单元：parse / validate_and_normalize / render_itinerary_md
  2. 端到端：全链路产出结构化 itinerary_data（days/items/route/sources/tips）+ Markdown 渲染
"""
import json
import sys
import time
import uuid

import requests

BASE = "http://localhost:8001"

SAMPLE = {
    "days": [
        {
            "day": 1, "date": "2026-09-02", "theme": "西湖文化", "weather": "晴 22~28°C",
            "items": [
                {"type": "景点", "name": "西湖", "start_time": "09:00", "end_time": "11:30",
                 "duration_minutes": 150, "reason": "必去", "address": "西湖风景区",
                 "latitude": None, "longitude": None, "image": "", "poi_id": "B000A"},
                {"type": "美食", "name": "楼外楼", "start_time": "12:00", "end_time": "13:00",
                 "duration_minutes": 60, "reason": "本地菜", "address": "孤山路",
                 "latitude": None, "longitude": None, "image": "", "poi_id": "B000B"},
                {"type": "景点", "name": "灵隐寺", "start_time": "14:00", "end_time": "16:00",
                 "duration_minutes": 120, "reason": "古刹", "address": "灵隐路",
                 "latitude": None, "longitude": None, "image": "", "poi_id": "B000C"},
            ],
            "route": {"points": [], "distance_km": 0, "estimated_minutes": 0},
        }
    ],
    "tips": ["带伞", "提前预约"],
}


def poll_status(uid, sid, tid, timeout=300, stop_at=()):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE}/agent/status/{uid}/{sid}/{tid}", timeout=10).json()
        if r["status"] in ("completed", "error") or r["status"] in stop_at:
            return r
        time.sleep(2)
    return {"status": "timeout"}


def resume(uid, sid, tid, command):
    r = requests.post(f"{BASE}/agent/resume", json={
        "user_id": uid, "session_id": sid, "task_id": tid, "command": command,
    }, timeout=10)
    assert r.status_code == 200, f"resume failed: {r.text}"


def unit_schema():
    from utils.itinerary_schema import parse_itinerary_json, render_itinerary_md, validate_and_normalize
    # 合法
    norm, errs = validate_and_normalize(json.loads(json.dumps(SAMPLE)), 1, start_date="2026-09-02")
    assert not errs, f"合法输入不应有错误: {errs}"
    assert norm["days"][0]["date"] == "2026-09-02"
    assert len(norm["days"][0]["items"]) == 3
    print("[PASS] 单元：合法行程通过校验")

    # 日期强制递增（给 start_date，忽略 LLM 给的错日期）
    bad_date = json.loads(json.dumps(SAMPLE))
    bad_date["days"][0]["date"] = "2020-01-01"
    norm, errs = validate_and_normalize(bad_date, 1, start_date="2026-09-02")
    assert not errs and norm["days"][0]["date"] == "2026-09-02"
    print("[PASS] 单元：date 按 start_date 强制递增")

    # 天数不符 -> error
    norm, errs = validate_and_normalize(json.loads(json.dumps(SAMPLE)), 3, start_date="2026-09-02")
    assert errs and "days 数组长度" in errs[0]
    print("[PASS] 单元：天数不符被拦截")

    # 缺 name -> error
    no_name = json.loads(json.dumps(SAMPLE))
    no_name["days"][0]["items"][0]["name"] = "  "
    norm, errs = validate_and_normalize(no_name, 1, start_date="2026-09-02")
    assert errs and "缺 name" in errs[0]
    print("[PASS] 单元：缺 name item 被拦截")

    # items 少于 3 -> error
    few = json.loads(json.dumps(SAMPLE))
    few["days"][0]["items"] = few["days"][0]["items"][:2]
    norm, errs = validate_and_normalize(few, 1, start_date="2026-09-02")
    assert errs and "少于 3" in errs[0]
    print("[PASS] 单元：item 不足被拦截")

    # parse 非法文本
    data, err = parse_itinerary_json("这不是JSON")
    assert data is None and err
    print("[PASS] 单元：非法文本解析失败")

    # render（用全新合法数据，前面 norm 已被错误用例覆盖）
    norm, errs = validate_and_normalize(json.loads(json.dumps(SAMPLE)), 1, start_date="2026-09-02")
    assert not errs
    md = render_itinerary_md(norm)
    assert "第 1 天" in md and "西湖" in md and "灵隐寺" in md
    print("[PASS] 单元：render_itinerary_md 确定性渲染 Markdown")


def e2e():
    uname = f"e2e_{uuid.uuid4().hex[:8]}"
    pw = "pass123"
    requests.post(f"{BASE}/auth/register", json={"username": uname, "password": pw}, timeout=10)
    login = requests.post(f"{BASE}/auth/login", json={"username": uname, "password": pw}, timeout=10).json()
    uid = login["user_id"]

    tid, sid = str(uuid.uuid4()), str(uuid.uuid4())
    r = requests.post(f"{BASE}/agent/invoke", json={
        "user_id": uid, "session_id": sid, "task_id": tid,
        "query": "9月2号到9月5号去杭州，轻松点",
    }, timeout=10)
    assert r.status_code == 200, f"invoke failed: {r.text}"

    st = poll_status(uid, sid, tid, stop_at=("interrupted",))
    assert st["status"] in ("interrupted", "completed"), f"规划应跑到中断/完成，实际 {st['status']}"
    if st["status"] == "interrupted":
        resume(uid, sid, tid, {"action": "accept"})
        st = poll_status(uid, sid, tid, stop_at=("interrupted",))
        if st["status"] == "interrupted":
            resume(uid, sid, tid, {"action": "save"})
            st = poll_status(uid, sid, tid)

    assert st["status"] == "completed", f"最终应 completed，实际 {st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    data = result.get("itinerary_data") or {}
    days = data.get("days") or []
    assert len(days) == 4, f"应生成 4 天行程，实际 {len(days)}"
    assert all(d.get("items") and len(d["items"]) >= 3 for d in days), "每天应有 >=3 items"
    first = days[0]
    assert first["date"] == "2026-09-02", f"第 1 天日期应 2026-09-02，实际 {first.get('date')}"
    assert first.get("weather"), "每天应有天气摘要"
    item = first["items"][0]
    for k in ("type", "name", "start_time", "end_time", "duration_minutes", "reason",
              "address", "latitude", "longitude", "image", "poi_id"):
        assert k in item, f"item 缺字段 {k}"
    # Phase 5 之后：坐标/路线由 enrich_routes 用地图服务填充（LLM 仍不编造）。
    # 此处只验证字段存在，坐标真实性与路线正值由 test_phase5_e2e 深入校验。
    assert "latitude" in item and "longitude" in item, "item 应有坐标字段"
    assert "distance_km" in first["route"] and "estimated_minutes" in first["route"]
    assert "sources" in data, "应注入研究来源"
    assert "tips" in data and data["tips"], "应有 tips"
    md = result.get("itinerary") or ""
    assert "第 1 天" in md, "应提供 Markdown 渲染版行程"
    assert "本行程参考了" in md and data.get("sources"), "Markdown 应体现来源数"
    print(f"\n结构化行程：{len(days)} 天，第1天 {len(first['items'])} 项")
    print(f"研究来源：{len(data.get('sources') or [])} 个")
    print("[PASS] 端到端：结构化 JSON 行程 + Markdown 渲染 + 来源注入")


def main():
    unit_schema()
    e2e()
    print("\nALL PHASE4 CASES PASSED")


if __name__ == "__main__":
    main()

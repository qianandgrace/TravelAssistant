"""Phase 5 验证：地图 + Route API（真实坐标 + 每日路线距离/时长）。

  1. 单元：_split_location（纯函数）
  2. 端到端：全链路产出结构化 itinerary_data，其中 items 有真实经纬度、
     每天 route 有 points(markers) + distance_km>0 + estimated_minutes>0
"""
import json
import time
import uuid

import requests

BASE = "http://localhost:8001"


def poll_status(uid, sid, tid, timeout=360, stop_at=()):
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


def unit_split():
    from utils.workflow import _split_location
    assert _split_location("120.14,30.26") == (120.14, 30.26), "标准 lng,lat"
    assert _split_location(None) is None
    assert _split_location("") is None
    assert _split_location("abc,def") is None
    assert _split_location("120.14") is None
    print("[PASS] 单元：_split_location 解析/容错")


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
    assert st["status"] in ("interrupted", "completed"), f"实际 {st['status']}"
    if st["status"] == "interrupted":
        resume(uid, sid, tid, {"action": "accept"})
        st = poll_status(uid, sid, tid, stop_at=("interrupted",))
        if st["status"] == "interrupted":
            resume(uid, sid, tid, {"action": "save"})
            st = poll_status(uid, sid, tid)

    assert st["status"] == "completed", f"实际 {st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    data = result.get("itinerary_data") or {}
    days = data.get("days") or []
    assert len(days) == 4, f"应 4 天，实际 {len(days)}"

    # 坐标：至少有 1 个 item 解析出真实经纬度
    coord_count = sum(
        1 for d in days for it in (d.get("items") or [])
        if it.get("latitude") is not None and it.get("longitude") is not None
    )
    assert coord_count >= 3, f"至少 3 个 item 应有真实坐标，实际 {coord_count}"

    # 路线：至少有 1 天有非空 route（points>=2, distance>0, minutes>0）
    routed = [d for d in days if (d.get("route") or {}).get("points")]
    assert routed, "至少 1 天应有 route.points"
    best = max(days, key=lambda d: (d.get("route") or {}).get("distance_km") or 0)
    route = best["route"]
    assert len(route["points"]) >= 2, "route.points 至少 2 个 marker"
    assert route["distance_km"] > 0, "route.distance_km 应为真实正值"
    assert route["estimated_minutes"] > 0, "route.estimated_minutes 应为真实正值"
    for p in route["points"]:
        assert p.get("latitude") is not None and p.get("longitude") is not None
        assert p.get("name"), "marker 应有名称"
    md = result.get("itinerary") or ""
    assert "路线约" in md, "Markdown 应含真实路线信息"

    print(f"\n坐标命中：{coord_count} 个 item 有真实经纬度")
    print(f"有路线的天数：{len(routed)}/{len(days)}")
    print(f"最远的一天（第 {best['day']} 天）：{route['distance_km']} km，约 {route['estimated_minutes']} 分钟，{len(route['points'])} 个途经点")
    print(f"示例 marker：{route['points'][0]['name']} ({route['points'][0]['latitude']}, {route['points'][0]['longitude']})")
    print("[PASS] 端到端：真实坐标 + 每日真实路线（markers + 距离 + 时长）")


def main():
    unit_split()
    e2e()
    print("\nALL PHASE5 CASES PASSED")


if __name__ == "__main__":
    main()

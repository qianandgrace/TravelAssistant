"""Phase 7 验证：前端 Timeline/Cards/Map。

  1. 单元：render_itinerary_product_html（55/45 布局 / 每日卡片 / SVG 地图 / 懒加载 / 转义）
  2. 单元：render_svg_map 边界（无坐标 / 单点）
  3. 单元：render_rec_cards_html（推荐卡片带图）
  4. 端到端：真实行程 -> 渲染产品 HTML；webapp 启动
"""
import time
import uuid

import requests

BASE = "http://localhost:8001"

SAMPLE_DAYS = [
    {
        "day": 1, "date": "2026-09-02", "theme": "西湖经典环游", "weather": "晴 22~28°C",
        "items": [
            {"type": "景点", "name": "西湖", "start_time": "09:00", "end_time": "11:30",
             "duration_minutes": 150, "reason": "必去地标", "address": "西湖风景区",
             "latitude": 30.2593, "longitude": 120.1417, "image": "http://img.example/xi hu.jpg",
             "poi_id": "B1"},
            {"type": "美食", "name": "楼外楼", "start_time": "12:00", "end_time": "13:00",
             "duration_minutes": 60, "reason": "本地菜", "address": "孤山路",
             "latitude": 30.2521, "longitude": 120.1508, "image": "", "poi_id": "B2"},
        ],
        "route": {"points": [
            {"name": "西湖", "latitude": 30.2593, "longitude": 120.1417},
            {"name": "楼外楼", "latitude": 30.2521, "longitude": 120.1508},
        ], "distance_km": 2.1, "estimated_minutes": 28},
    },
    {
        "day": 2, "date": "2026-09-03", "theme": "灵隐古刹", "weather": "多云 24~30°C",
        "items": [
            {"type": "景点", "name": "灵隐寺", "start_time": "08:30", "end_time": "11:00",
             "duration_minutes": 150, "reason": "千年古刹", "address": "灵隐路",
             "latitude": 30.2419, "longitude": 120.0987, "image": "http://img.example/lingyin.jpg",
             "poi_id": "B3"},
            {"type": "交通", "name": "打车至宋城", "start_time": "13:00", "end_time": "13:40",
             "duration_minutes": 40, "reason": "跨区", "address": "",
             "latitude": None, "longitude": None, "image": "", "poi_id": None},
        ],
        "route": {"points": [
            {"name": "灵隐寺", "latitude": 30.2419, "longitude": 120.0987},
        ], "distance_km": 0, "estimated_minutes": 0},
    },
]

SAMPLE_DATA = {
    "days": SAMPLE_DAYS,
    "tips": ["带伞", "提前预约"],
    "sources": [{"title": "攻略A", "url": "http://a"}, {"title": "攻略B", "url": "http://b"}],
}


def unit_product():
    from utils.render_html import render_itinerary_product_html
    result = {
        "itinerary_data": SAMPLE_DATA,
        "destination": "杭州",
        "days": 2,
        "preference": "轻松",
        "start_date": "2026-09-02",
    }
    h = render_itinerary_product_html(result)
    assert "grid-template-columns:55% 45%" in h, "桌面端应为 55/45 布局"
    assert "@media(max-width:900px)" in h, "应有移动端堆叠媒体查询"
    assert "第 1 天" in h and "第 2 天" in h, "应渲染每日卡片"
    assert "西湖" in h and "灵隐寺" in h
    assert "loading='lazy'" in h, "图片应懒加载"
    assert "<svg" in h, "应包含 SVG 地图"
    assert "<path" in h, "SVG 应有路线折线"
    assert "circle" in h, "SVG 应有 marker"
    assert "小贴士" in h and "带伞" in h, "应渲染 tips"
    assert "参考了 2 个攻略" in h, "应渲染来源数"
    assert "全程步行约 2.1 km" in h, "头部应汇总全程距离"
    print("[PASS] 单元：行程产品视图（布局/卡片/地图/懒加载/tips/头部）")


def unit_svg_edge():
    from utils.render_html import render_svg_map
    svg_empty = render_svg_map([{"items": [{"name": "x", "latitude": None, "longitude": None}]}])
    assert "暂无地图数据" in svg_empty, "无坐标应回退"
    svg_one = render_svg_map([{"items": [
        {"name": "A", "latitude": 30.25, "longitude": 120.14}]}])
    assert "<svg" in svg_one and "circle" in svg_one, "单点不应崩"
    print("[PASS] 单元：SVG 地图边界（无坐标 / 单点）")


def unit_rec():
    from utils.render_html import render_rec_cards_html
    cards = [{
        "destination": "丽江", "country": "中国", "recommended_days": 5,
        "estimated_budget_level": "中等", "reason": "古城雪山",
        "best_for": "情侣", "image": "http://img.example/lijiang.jpg",
        "highlights": ["古城", "雪山"],
    }]
    h = render_rec_cards_html(cards)
    assert "丽江" in h and "http://img.example/lijiang.jpg" in h
    assert "loading='lazy'" in h
    print("[PASS] 单元：推荐卡片 HTML（带图 + 懒加载）")


def unit_escape():
    from utils.render_html import render_itinerary_product_html
    bad = {"itinerary_data": {
        "days": [{"day": 1, "date": "", "theme": "", "weather": "", "items": [
            {"type": "景点", "name": "<script>alert(1)</script>", "start_time": "", "end_time": "",
             "duration_minutes": 0, "reason": "x", "address": "", "latitude": None,
             "longitude": None, "image": "", "poi_id": None}],
            "route": {}}],
        "tips": [], "sources": []}, "destination": "杭州", "days": 1}
    h = render_itinerary_product_html(bad)
    assert "<script>" not in h, "item 名称应被转义"
    print("[PASS] 单元：HTML 转义（防注入）")


def e2e_product():
    uname = f"e2e_{uuid.uuid4().hex[:8]}"
    pw = "pass123"
    requests.post(f"{BASE}/auth/register", json={"username": uname, "password": pw}, timeout=10)
    login = requests.post(f"{BASE}/auth/login", json={"username": uname, "password": pw}, timeout=10).json()
    uid = login["user_id"]
    tid, sid = str(uuid.uuid4()), str(uuid.uuid4())
    requests.post(f"{BASE}/agent/invoke", json={
        "user_id": uid, "session_id": sid, "task_id": tid,
        "query": "9月2号到9月5号去杭州，轻松点",
    }, timeout=10)
    deadline = time.time() + 360
    st = None
    while time.time() < deadline:
        st = requests.get(f"{BASE}/agent/status/{uid}/{sid}/{tid}", timeout=10).json()
        if st["status"] in ("completed", "error", "interrupted"):
            break
        time.sleep(2)
    if st["status"] == "interrupted":
        requests.post(f"{BASE}/agent/resume", json={
            "user_id": uid, "session_id": sid, "task_id": tid, "command": {"action": "accept"},
        }, timeout=10)
        while time.time() < deadline:
            st = requests.get(f"{BASE}/agent/status/{uid}/{sid}/{tid}", timeout=10).json()
            if st["status"] in ("completed", "error", "interrupted"):
                break
            time.sleep(2)
        if st["status"] == "interrupted":
            requests.post(f"{BASE}/agent/resume", json={
                "user_id": uid, "session_id": sid, "task_id": tid, "command": {"action": "save"},
            }, timeout=10)
            while time.time() < deadline:
                st = requests.get(f"{BASE}/agent/status/{uid}/{sid}/{tid}", timeout=10).json()
                if st["status"] in ("completed", "error"):
                    break
                time.sleep(2)
    assert st and st["status"] == "completed", f"实际 {st and st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    assert result.get("itinerary_data"), "应有结构化行程"
    from utils.render_html import render_itinerary_product_html
    h = render_itinerary_product_html(result)
    days = result["itinerary_data"]["days"]
    assert f"第 {len(days)} 天" in h, "真实行程应渲染所有天"
    assert "grid-template-columns:55% 45%" in h
    assert "<svg" in h and "loading='lazy'" in h
    real_img = sum(1 for d in days for it in d["items"] if (it.get("image") or "").startswith("http"))
    print(f"\n真实行程：{len(days)} 天，{real_img} 个真实图片 item")
    print(f"产品 HTML 长度：{len(h)} 字符（含 {len(days)} 天卡片 + SVG 地图）")
    print("[PASS] 端到端：真实行程 -> 产品 HTML 渲染")


def main():
    unit_product()
    unit_svg_edge()
    unit_rec()
    unit_escape()
    e2e_product()
    print("\nALL PHASE7 CASES PASSED")


if __name__ == "__main__":
    main()

"""Phase 6 验证：图片（真实 POI 图，非 LLM 编造；失败回退占位图；懒加载由前端做）。

  1. 单元：fill_item_images（poi_id 命中取图 / text_search 兜底 / 占位图回退 / 非地点类型留空）
  2. 端到端-行程：itinerary_data 的 items 有真实图片（http 开头或占位 data-URI）
  3. 端到端-推荐：目的地卡片 image 也被填充
"""
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


class FakeTextSearch:
    """返回固定 POI 响应的假 text_search 工具。"""
    def __init__(self, photo):
        self._photo = photo
        self.calls = 0

    async def ainvoke(self, kwargs):
        self.calls += 1
        return [{"type": "text", "text": '{"pois": [{"photo": "%s"}]}' % self._photo}]


def unit_fill():
    import asyncio
    from utils.images import PLACEHOLDER_IMAGE, fill_item_images

    # poi_id 命中 -> 直接用搜索结果 photo
    pois_by_id = {"B1": {"photo": "http://img/real.jpg"}}
    items = [{"type": "景点", "name": "西湖", "poi_id": "B1", "image": ""}]
    asyncio.run(fill_item_images(items, pois_by_id, None, "杭州"))
    assert items[0]["image"] == "http://img/real.jpg", "poi_id 命中应取搜索结果图"
    print("[PASS] 单元：poi_id 命中取搜索结果真实图")

    # 未命中 + 有 text_search -> 兜底取图
    items2 = [{"type": "景点", "name": "灵隐寺", "poi_id": "B9", "image": ""}]
    fake = FakeTextSearch("http://img/fallback.jpg")
    asyncio.run(fill_item_images(items2, {}, fake, "杭州"))
    assert items2[0]["image"] == "http://img/fallback.jpg", "text_search 兜底应取图"
    assert fake.calls == 1
    print("[PASS] 单元：text_search 兜底取真实图")

    # 无工具 + 未命中 -> 占位图
    items3 = [{"type": "美食", "name": "某店", "poi_id": None, "image": ""}]
    asyncio.run(fill_item_images(items3, {}, None, "杭州"))
    assert items3[0]["image"] == PLACEHOLDER_IMAGE, "应回退占位图"
    assert items3[0]["image"].startswith("data:image/svg+xml")
    print("[PASS] 单元：未拿到图片回退本地占位图")

    # 非地点类型 -> image 留空
    items4 = [{"type": "交通", "name": "打车", "poi_id": None, "image": ""}]
    asyncio.run(fill_item_images(items4, {}, None, "杭州"))
    assert items4[0]["image"] == "", "交通类型不填图"
    print("[PASS] 单元：非地点类型留空")


def _plan_e2e():
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
    items = [it for d in days for it in (d.get("items") or [])]
    place = [it for it in items if it.get("type") in ("景点", "美食", "酒店")]
    real = [it for it in place if it.get("image", "").startswith("http")]
    assert place, "应有地点类 item"
    assert real, "应至少 1 个地点 item 拿到真实 http 图片"
    assert len(real) >= len(place) * 0.5, f"真实图片应过半，{len(real)}/{len(place)}"
    for it in place:
        img = it["image"]
        assert img.startswith("http") or img.startswith("data:image/svg"), \
            f"图片来源非法(应为真实 API 图或占位图): {img[:50]}"
    print(f"\n行程地点类 item：{len(place)} 个，真实 http 图：{len(real)} 个")
    print(f"示例真实图：{real[0]['name']} -> {real[0]['image'][:80]}...")
    print("[PASS] 端到端-行程：真实图片填充（无 LLM 编造 URL，失败走占位图）")


def _rec_e2e():
    uname = f"e2e_{uuid.uuid4().hex[:8]}"
    pw = "pass123"
    requests.post(f"{BASE}/auth/register", json={"username": uname, "password": pw}, timeout=10)
    login = requests.post(f"{BASE}/auth/login", json={"username": uname, "password": pw}, timeout=10).json()
    uid = login["user_id"]
    tid, sid = str(uuid.uuid4()), str(uuid.uuid4())
    r = requests.post(f"{BASE}/agent/invoke", json={
        "user_id": uid, "session_id": sid, "task_id": tid,
        "query": "国庆5天不知道去哪",
    }, timeout=10)
    assert r.status_code == 200, f"invoke failed: {r.text}"
    st = poll_status(uid, sid, tid)
    assert st["status"] == "completed", f"实际 {st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    cards = result.get("recommendations") or []
    assert cards, "应返回推荐卡片"
    filled = sum(1 for c in cards if c.get("image"))
    for c in cards:
        img = c["image"]
        assert img.startswith("http") or img.startswith("data:image/svg"), \
            f"推荐卡片图片来源非法: {img[:50]}"
    print(f"\n推荐卡片：{len(cards)} 个，都有图片（{filled} 个已填充）")
    print(f"示例：{cards[0]['destination']} -> {cards[0]['image'][:70]}...")
    print("[PASS] 端到端-推荐：目的地卡片图片填充")


def main():
    unit_fill()
    _plan_e2e()
    _rec_e2e()
    print("\nALL PHASE6 CASES PASSED")


if __name__ == "__main__":
    main()

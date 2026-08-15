"""Phase 2 验证：目的地推荐。

  1. 单元：_validate / _normalize_item 对合法/非法输入的校验
  2. 端到端：真实 LLM 生成 3~6 个结构化目的地卡片，字段齐全
  3. 「选择这个目的地」产生的规划查询能正确路由为 TRAVEL_PLANNING（已在意图测试覆盖）
"""
import sys
import time
import uuid

import requests

BASE = "http://localhost:8001"


def poll_status(uid, sid, tid, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE}/agent/status/{uid}/{sid}/{tid}", timeout=10).json()
        if r["status"] in ("completed", "error"):
            return r
        time.sleep(2)
    return {"status": "timeout"}


def unit_validate():
    from utils.recommendation import _validate
    # 合法
    good = {"destinations": [
        {"destination": "杭州", "country": "中国", "reason": "西湖与宋城", "best_for": "情侣",
         "recommended_days": 5, "estimated_budget_level": "中等", "image": "",
         "highlights": ["西湖", "灵隐寺"]},
        {"destination": "大理", "country": "中国", "reason": "洱海环湖", "best_for": "摄影",
         "recommended_days": 4, "estimated_budget_level": "经济", "image": "",
         "highlights": ["洱海", "古城"]},
        {"destination": "东京", "country": "日本", "reason": "都市文化", "best_for": "美食",
         "recommended_days": 6, "estimated_budget_level": "奢华", "image": "",
         "highlights": ["浅草寺", "银座", "东京塔"]},
    ]}
    cards = _validate(good)
    assert len(cards) == 3, "合法输入应通过"
    assert all(set(("destination", "country", "reason", "best_for", "recommended_days",
                    "estimated_budget_level", "image", "highlights")) <= set(c) for c in cards)
    print("[PASS] 单元校验：合法卡片通过")

    # 非法：数量不足
    try:
        _validate({"destinations": [good["destinations"][0]]})
        raise AssertionError("数量不足应抛异常")
    except ValueError:
        pass
    # 非法：缺 destination
    try:
        bad = {"destinations": [{k: v for k, v in good["destinations"][0].items() if k != "destination"},
                                good["destinations"][1], good["destinations"][2]]}
        _validate(bad)
        raise AssertionError("缺 destination 应抛异常")
    except ValueError:
        pass
    # 非法：recommended_days 越界 -> 归一化为 3（容忍）
    item = dict(good["destinations"][0], recommended_days=999)
    cards = _validate({"destinations": [item, good["destinations"][1], good["destinations"][2]]})
    assert cards[0]["recommended_days"] == 3, "越界天数应归一化为 3"
    print("[PASS] 单元校验：非法输入正确拦截/归一化")


def e2e():
    uname = f"e2e_{uuid.uuid4().hex[:8]}"
    pw = "pass123"
    requests.post(f"{BASE}/auth/register", json={"username": uname, "password": pw}, timeout=10)
    login = requests.post(f"{BASE}/auth/login", json={"username": uname, "password": pw}, timeout=10).json()
    uid = login["user_id"]
    print(f"test user: {uname}")

    tid, sid = str(uuid.uuid4()), str(uuid.uuid4())
    r = requests.post(f"{BASE}/agent/invoke", json={
        "user_id": uid, "session_id": sid, "task_id": tid,
        "query": "国庆5天不知道去哪",
    }, timeout=10)
    assert r.status_code == 200, f"invoke failed: {r.text}"

    st = poll_status(uid, sid, tid)
    assert st["status"] == "completed", f"期望 completed，实际 {st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    assert result.get("intent") == "DESTINATION_RECOMMENDATION", f"意图错误: {result.get('intent')}"
    recs = result.get("recommendations") or []
    assert 3 <= len(recs) <= 6, f"推荐数量应 3~6，实际 {len(recs)}"
    print(f"\n推荐 {len(recs)} 个目的地：")
    for rec in recs:
        assert rec.get("destination") and rec.get("reason"), f"卡片字段缺失: {rec}"
        assert rec.get("recommended_days"), "recommended_days 应非空"
        assert "image" in rec, "image 字段应存在（Phase 6 填充）"
        print(f"  - {rec['destination']}（{rec.get('country')}）约{rec['recommended_days']}天 "
              f"预算:{rec.get('estimated_budget_level')} 亮点:{('、'.join(rec.get('highlights') or [])[:30])}")
    print("[PASS] 端到端：结构化目的地推荐生成成功")


def main():
    unit_validate()
    e2e()
    print("\nALL PHASE2 CASES PASSED")


if __name__ == "__main__":
    main()

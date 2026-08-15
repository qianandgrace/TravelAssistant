"""Phase 9 验收场景 A-D（需后端 server + Celery worker 运行）。

A. 完整东京规划（5 天，含 HITL 审阅接受 + 记忆保存两次中断）
B. 「国庆5天不知道去哪」-> 目的地推荐卡片（非空，含目的地）
C. 「写一个Java程序」-> 非旅游礼貌拒绝（is_non_travel）
D. 修改行程：同一会话再发「第三天太累删两个景点」-> 复用上一轮上下文完成修改
"""
import time
import uuid

import requests

BASE = "http://localhost:8001"


def _ready():
    try:
        requests.get(f"{BASE}/system/info", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


def _invoke(uid, sid, tid, query):
    requests.post(f"{BASE}/agent/invoke", json={
        "user_id": uid, "session_id": sid, "task_id": tid, "query": query,
    }, timeout=10)


def _poll(uid, sid, tid, deadline):
    """轮询到 completed/error，遇到 review_itinerary -> accept、confirm_memory -> save。"""
    while time.time() < deadline:
        st = requests.get(f"{BASE}/agent/status/{uid}/{sid}/{tid}", timeout=10).json()
        if st["status"] in ("completed", "error"):
            return st
        if st["status"] == "interrupted":
            idata = (st.get("last_response") or {}).get("interrupt_data") or {}
            kind = idata.get("kind", "")
            action = "accept" if kind == "review_itinerary" else "save"
            requests.post(f"{BASE}/agent/resume", json={
                "user_id": uid, "session_id": sid, "task_id": tid,
                "command": {"action": action},
            }, timeout=10)
        time.sleep(2)
    return {"status": "timeout"}


def scenario_a(uid):
    sid, tid = str(uuid.uuid4()), str(uuid.uuid4())
    _invoke(uid, sid, tid, "9月2号到9月6号去东京，轻松点")
    st = _poll(uid, sid, tid, time.time() + 600)
    assert st["status"] == "completed", f"A 未完成：{st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    days = (result.get("itinerary_data") or {}).get("days") or []
    assert days, "A 应有结构化行程"
    assert len(days) == 5, f"A 应为 5 天，实际 {len(days)}"
    real_img = sum(1 for d in days for it in d.get("items", [])
                   if (it.get("image") or "").startswith("http"))
    print(f"[PASS] A 完整东京规划：目的地={result.get('destination')}，5 天，"
          f"真实图片 {real_img} 个，来源 {len((result.get('research') or {}).get('sources') or [])} 个")
    return sid, days


def scenario_d(uid, sid, days_a):
    tid = str(uuid.uuid4())
    _invoke(uid, sid, tid, "第三天太累删两个景点")
    st = _poll(uid, sid, tid, time.time() + 600)
    assert st["status"] == "completed", f"D 未完成：{st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    assert result.get("itinerary_data"), "D 应有结构化行程"
    assert result.get("destination") == "东京", \
        f"D 目的地应复用上一轮东京，实际 {result.get('destination')}"
    days_d = result["itinerary_data"]["days"]
    orig3 = len(days_a[2]["items"]) if len(days_a) >= 3 else -1
    mod3 = len(days_d[2]["items"]) if len(days_d) >= 3 else -1
    print(f"[PASS] D 修改行程：目的地={result.get('destination')}，"
          f"第3天原 {orig3} 项 -> 改后 {mod3} 项（复用上一轮上下文）")


def scenario_b(uid):
    sid, tid = str(uuid.uuid4()), str(uuid.uuid4())
    _invoke(uid, sid, tid, "国庆5天不知道去哪")
    st = _poll(uid, sid, tid, time.time() + 300)
    assert st["status"] == "completed", f"B 未完成：{st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    recs = result.get("recommendations") or []
    assert recs, "B 应有推荐卡片"
    assert all(r.get("destination") for r in recs), "推荐卡片应含目的地"
    names = "、".join(r["destination"] for r in recs[:5])
    print(f"[PASS] B 国庆推荐：{len(recs)} 个目的地卡片：{names}")


def scenario_c(uid):
    sid, tid = str(uuid.uuid4()), str(uuid.uuid4())
    _invoke(uid, sid, tid, "写一个Java程序")
    st = _poll(uid, sid, tid, time.time() + 120)
    assert st["status"] == "completed", f"C 未完成：{st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    assert result.get("is_non_travel"), "C 应标记为非旅游"
    assert result.get("reply"), "C 应有礼貌回复"
    print(f"[PASS] C 非旅游拒绝：is_non_travel=True，回复：{result['reply'][:32]}…")


def main():
    if not _ready():
        print("[SKIP] 后端未运行，跳过验收场景（需 server + worker）")
        return
    uname = f"acc_{uuid.uuid4().hex[:8]}"
    pw = "pass123"
    requests.post(f"{BASE}/auth/register", json={"username": uname, "password": pw}, timeout=10)
    uid = requests.post(f"{BASE}/auth/login", json={"username": uname, "password": pw},
                        timeout=10).json()["user_id"]

    sid, days_a = scenario_a(uid)   # A：完整东京规划
    scenario_d(uid, sid, days_a)    # D：同一会话修改行程
    scenario_b(uid)                 # B：国庆推荐
    scenario_c(uid)                 # C：非旅游拒绝
    print("\nALL PHASE9 ACCEPTANCE CASES PASSED")


if __name__ == "__main__":
    main()

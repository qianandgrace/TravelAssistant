"""Phase 1 端到端回归：通过真实 FastAPI + Celery 验证意图路由三分支。

  1. NON_TRAVEL  -> completed + 礼貌拒绝（不走 workflow）
  2. TRAVEL_QA    -> completed + LLM 回答
  3. TRAVEL_PLANNING -> 实体抽取 -> workflow 正常运行（天津 4 天）
"""
import sys
import time
import uuid

import requests

BASE = "http://localhost:8001"


def poll_status(uid, sid, tid, timeout=240, stop_at=()):
    """轮询直到 completed/error，或到达 stop_at 中的某个状态（如 interrupted）。"""
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


def run_case(uid, label, query, want_status, poll=True):
    tid, sid = str(uuid.uuid4()), str(uuid.uuid4())
    r = requests.post(f"{BASE}/agent/invoke", json={
        "user_id": uid, "session_id": sid, "task_id": tid, "query": query,
    }, timeout=10)
    assert r.status_code == 200, f"invoke failed: {r.text}"
    print(f"\n===== [{label}] {query}")
    st = poll_status(uid, sid, tid)
    print(f"status={st['status']}")
    result = (st.get("last_response") or {}).get("result") or {}
    intent = result.get("intent") or (result.get("parsed") or {}).get("intent") or ""
    print(f"intent={intent}")
    if result.get("reply"):
        print(f"reply: {result['reply'][:120]}...")
    if result.get("itinerary"):
        print(f"itinerary: {result['itinerary'][:150]}...")
    assert st["status"] == want_status, f"期望 {want_status}，实际 {st['status']}"
    return st


def main():
    # 注册一个测试用户
    uname = f"e2e_{uuid.uuid4().hex[:8]}"
    pw = "pass123"
    reg = requests.post(f"{BASE}/auth/register", json={"username": uname, "password": pw}, timeout=10).json()
    login = requests.post(f"{BASE}/auth/login", json={"username": uname, "password": pw}, timeout=10).json()
    uid = login["user_id"]
    print(f"test user: {uname} uid={uid}")

    # 1) 非旅游 -> 礼貌拒绝，completed
    st = run_case(uid, "NON_TRAVEL", "写一个Java程序", "completed")
    result = (st["last_response"] or {}).get("result") or {}
    assert result.get("is_non_travel") and result.get("reply"), "NON_TRAVEL 应有拒绝回复"
    print("[PASS] NON_TRAVEL 拒绝")

    # 2) 旅游问答 -> LLM 直接回答
    st = run_case(uid, "TRAVEL_QA", "去日本需要签证吗", "completed")
    result = (st["last_response"] or {}).get("result") or {}
    assert result.get("reply") and result.get("intent") == "TRAVEL_QA", "TRAVEL_QA 应有回答"
    print("[PASS] TRAVEL_QA 问答")

    # 3) 目的地推荐（Phase 2 前占位）-> LLM 直接回答
    st = run_case(uid, "DEST_RECOMMEND", "国庆5天不知道去哪", "completed")
    result = (st["last_response"] or {}).get("result") or {}
    assert result.get("reply"), "推荐占位应有回答"
    print("[PASS] DEST_RECOMMEND 占位回答")

    # 4) 规划 -> 实体抽取 -> workflow；自动处理两个 HITL 中断（审阅/记忆）直到完成
    tid, sid = str(uuid.uuid4()), str(uuid.uuid4())
    r = requests.post(f"{BASE}/agent/invoke", json={
        "user_id": uid, "session_id": sid, "task_id": tid,
        "query": "9月2号到9月5号去天津，轻松点",
    }, timeout=10)
    assert r.status_code == 200, f"invoke failed: {r.text}"
    print(f"\n===== [TRAVEL_PLANNING] 9月2号到9月5号去天津，轻松点")

    st = poll_status(uid, sid, tid, stop_at=("interrupted",))
    assert st["status"] in ("interrupted", "completed"), f"规划应至少跑到中断/完成，实际 {st['status']}"

    if st["status"] == "interrupted":
        # review_itinerary 中断 -> 接受
        lr = st.get("last_response") or {}
        idata = (lr.get("interrupt_data") or {})
        print(f"interrupt kind: {idata.get('kind')}")
        resume(uid, sid, tid, {"action": "accept"})
        st = poll_status(uid, sid, tid, stop_at=("interrupted",))
        if st["status"] == "interrupted":
            # confirm_memory 中断 -> 保存
            lr = st.get("last_response") or {}
            idata = (lr.get("interrupt_data") or {})
            print(f"interrupt kind: {idata.get('kind')}")
            resume(uid, sid, tid, {"action": "save"})
            st = poll_status(uid, sid, tid)

    assert st["status"] == "completed", f"规划最终应 completed，实际 {st['status']}"
    result = (st["last_response"] or {}).get("result") or {}
    parsed = result.get("parsed") or {}
    print(f"parsed: destination={parsed.get('destination')} days={parsed.get('days')} intent={parsed.get('intent')}")
    assert parsed.get("destination") == "天津" and parsed.get("intent") == "TRAVEL_PLANNING", "实体/意图应正确"
    assert result.get("itinerary"), "规划应产出行程"
    print(f"itinerary 开头: {result['itinerary'][:150]}...")
    print("[PASS] TRAVEL_PLANNING 全链路（实体抽取 -> workflow -> 两次中断 -> 完成）")

    print("\nALL PHASE1 E2E CASES PASSED")


if __name__ == "__main__":
    main()

"""Phase 3 验证：Travel Research Pipeline。

  1. 单元：research_summary 结构（真实 LLM，无 Tavily key 时 sources 必须为空、不编造）
  2. 单元：_format_research 把结构压缩成给 planner 的文本
  3. 端到端：规划全链路带上 research，最终结果含 research.sources
"""
import asyncio
import sys
import time
import uuid

import requests

BASE = "http://localhost:8001"


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


def unit_research():
    from utils.research import research_summary, search_guides
    from utils.workflow import _format_research

    async def _run():
        sr = await search_guides("杭州")
        assert isinstance(sr["sources"], list)
        assert isinstance(sr["search_available"], bool)
        print(f"[unit] search_guides: available={sr['search_available']} sources={len(sr['sources'])}")
        if sr["search_available"]:
            # 有 Tavily key：必须是 3~8 条真实来源，每条含 title+url（不编造）
            assert 3 <= len(sr["sources"]) <= 8, f"来源数应 3~8，实际 {len(sr['sources'])}"
            assert all(s.get("title") and s.get("url") for s in sr["sources"]), "来源缺 title/url"
            print("[unit] 有搜索能力：真实来源 3~8 条，title+url 齐全")
        else:
            assert sr["sources"] == [], "无搜索能力时 sources 必须为空（不编造）"
            print("[unit] 无搜索能力：sources 为空（诚实降级）")

        summary = await research_summary("杭州", "轻松", 4, sr["sources"])
        assert len(summary.get("sources") or []) == len(sr["sources"]), "summary.sources 应等于真实来源数"
        for key in ("area_clusters", "common_routes", "popular_combinations",
                    "transportation_tips", "avoid", "practical_tips"):
            assert key in summary, f"研究摘要缺字段 {key}"
        text = _format_research(summary)
        assert text and "（无）" not in text, "格式化研究摘要应为非空结论"
        print("[unit] research_summary 结构完整，sources 注入真实来源")
        print("[unit] _format_research 示例：\n" + text)

    asyncio.run(_run())


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
    research = result.get("research") or {}
    assert "sources" in research, "结果应含 research.sources"
    assert result.get("itinerary"), "行程应生成"
    print(f"\nresearch.sources={len(research.get('sources') or [])}（真实攻略来源；前端将显示「参考攻略来源：N」）")
    print(f"research keys: {[k for k in research if k != 'sources']}")
    print("[PASS] 端到端：规划链路已带上 Travel Research")
    print(f"itinerary 开头: {result['itinerary'][:120]}...")


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    unit_research()
    e2e()
    print("\nALL PHASE3 CASES PASSED")


if __name__ == "__main__":
    main()

"""Phase 8 验证：HTML 导出（独立文件 + Key 安全）。

  1. 单元：render_itinerary_export_html（完整文档结构 / 标题 / 页脚 / 转义）
  2. 单元：export_filename（非法字符清除 / 空回退 / 限长）
  3. 单元：Key 安全 —— 即使设置了服务端 Key 环境变量，导出 HTML 也不含其值
  4. 端点：TestClient + fake session manager（无需 Redis/Postgres）验证 /agent/export 各分支
  5. 端到端：真实完成一次行程 -> 导出 HTML -> 校验内容与 Key 安全（需后端运行中）
"""
import os
import sys
import time
import uuid

import requests

BASE = "http://localhost:8001"

# 服务端可能存在的全部敏感 Key 环境变量（导出文件不得出现它们的值）
SENSITIVE_ENV = [
    "AMAP_MAPS_API_KEY", "TAVILY_API_KEY", "QWEN_API_KEY",
    "DEEPSEEK_API_KEY", "LAOZHANG_API_KEY", "OPENAI_API_KEY",
]


def _sample_result():
    return {
        "destination": "东京",
        "days": 2,
        "preference": "轻松好吃",
        "start_date": "2026-09-02",
        "itinerary_data": {
            "days": [
                {
                    "day": 1, "date": "2026-09-02", "theme": "浅草・晴空塔", "weather": "晴",
                    "route": {"distance_km": 3.2, "estimated_minutes": 45, "points": []},
                    "items": [
                        {"type": "酒店", "name": "浅草酒店", "start_time": "15:00", "end_time": "18:00",
                         "reason": "放行李后轻松出发", "address": "台东区浅草",
                         "image": "https://store.is.autonavi.com/showpic/9a1b2c.jpg",
                         "latitude": 35.7140, "longitude": 139.7960},
                        {"type": "景点", "name": "浅草寺", "start_time": "18:30", "end_time": "20:00",
                         "reason": "夜游体验", "address": "台东区浅草2-3-1", "image": "",
                         "latitude": 35.7148, "longitude": 139.7967},
                        {"type": "美食", "name": "一兰拉面", "start_time": "20:00", "end_time": "21:00",
                         "reason": "晚餐", "address": "台东区", "image": "",
                         "latitude": 35.7100, "longitude": 139.7980},
                    ],
                },
                {
                    "day": 2, "date": "2026-09-03", "theme": "涩谷・新宿", "weather": "多云",
                    "route": {"distance_km": 5.0, "estimated_minutes": 70, "points": []},
                    "items": [
                        {"type": "景点", "name": "明治神宫", "start_time": "09:00", "end_time": "11:00",
                         "reason": "清晨静谧", "address": "涩谷区", "image": "",
                         "latitude": 35.6764, "longitude": 139.6993},
                        {"type": "美食", "name": "一风堂", "start_time": "12:00", "end_time": "13:00",
                         "reason": "午餐", "address": "新宿", "image": "",
                         "latitude": 35.6938, "longitude": 139.7034},
                        {"type": "酒店", "name": "新宿酒店", "start_time": "21:00", "end_time": "22:00",
                         "reason": "入住", "address": "新宿", "image": "",
                         "latitude": 35.6900, "longitude": 139.7000},
                    ],
                },
            ],
            "tips": ["提前订票", "晚上早点回酒店"],
            "sources": [{"title": "东京攻略", "url": "http://example.com/tokyo-guide"}],
        },
    }


def unit_export_html():
    from utils.render_html import render_itinerary_export_html
    h = render_itinerary_export_html(_sample_result())
    assert h.startswith("<!DOCTYPE html>"), "应为完整 HTML 文档"
    assert "<meta charset='utf-8'>" in h, "应声明 UTF-8 编码"
    assert "<title>东京 · 行程规划</title>" in h, "标题应含目的地"
    assert h.rstrip().endswith("</html>"), "应以 </html> 结尾"
    assert "由 TravelAssistant 生成" in h, "应有导出页脚"
    assert "浅草寺" in h and "浅草酒店" in h, "应包含每日卡片内容"
    assert "tp-layout" in h and "<svg" in h, "应包含产品视图布局与内联 SVG 地图"
    assert "小贴士" in h and "提前订票" in h, "应包含 tips"
    assert "route" and "3.2" in h, "应包含路线距离"
    print("[PASS] 单元：导出 HTML 文档结构")


def unit_escape():
    from utils.render_html import render_itinerary_export_html
    r = _sample_result()
    r["destination"] = '东京"><script>alert(1)</script>'
    h = render_itinerary_export_html(r)
    assert "<script>alert(1)</script>" not in h, "注入脚本应被转义"
    assert "&lt;script&gt;" in h, "转义实体应存在"
    print("[PASS] 单元：导出 HTML 转义（防注入）")


def unit_filename():
    from utils.render_html import export_filename
    assert export_filename("东京") == "行程_东京.html", "中文名应保留"
    assert export_filename("") == "行程.html", "空目的地回退"
    bad = export_filename('a<b>:c"d/e\\f|g?h*i')
    assert all(ch not in bad for ch in '<>:"/\\|?*'), f"非法字符应被清除：{bad}"
    long_name = export_filename("一" * 100)
    assert long_name.startswith("行程_") and long_name.endswith(".html")
    assert len(long_name) <= len("行程_") + 40 + len(".html"), f"过长的文件名应截断：{long_name}"
    print("[PASS] 单元：导出文件名（非法字符/空值/限长）")


def unit_no_keys():
    from utils.render_html import render_itinerary_export_html
    html = render_itinerary_export_html(_sample_result())
    leaked = [k for k in SENSITIVE_ENV if os.getenv(k) and os.getenv(k) in html]
    assert not leaked, f"导出 HTML 泄漏了服务端 Key：{leaked}"
    # 强断言：即使把每个 Key 环境变量改成哨兵值，导出也不含该值
    saved = {}
    try:
        for k in SENSITIVE_ENV:
            saved[k] = os.environ.get(k)
            os.environ[k] = f"SENTINEL_{k}_7f3a"
        html2 = render_itinerary_export_html(_sample_result())
        for k in SENSITIVE_ENV:
            assert f"SENTINEL_{k}_7f3a" not in html2, f"导出 HTML 竟包含 {k} 的值"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("[PASS] 单元：Key 安全（任何 Key 环境变量值都不会出现在导出 HTML）")


def endpoint_testclient():
    """用 fake session manager 注入，无需 Redis/Postgres 验证端点各分支。"""
    from fastapi.testclient import TestClient
    import server

    class _SM:
        def __init__(self, session=None, exists=True):
            self._session = session
            self._exists = exists

        async def session_task_id_exists(self, *a):
            return self._exists

        async def get_session_by_task(self, *a):
            return self._session

    # 1) 正常：completed + itinerary_data -> 200 附件下载
    server.app.state.session_manager = _SM(
        {"status": "completed", "last_response": {"result": _sample_result()}}
    )
    r = TestClient(server.app).get("/agent/export/u1/s1/t1")
    assert r.status_code == 200, f"应 200，实际 {r.status_code}"
    assert "text/html" in r.headers.get("content-type", ""), "应返回 text/html"
    assert "浅草寺" in r.text and "<title>东京 · 行程规划</title>" in r.text
    dispo = r.headers.get("content-disposition", "")
    assert "attachment" in dispo, "应以附件下载"
    assert "filename*=UTF-8''" in dispo, "应带 UTF-8 文件名"
    for k in SENSITIVE_ENV:
        if os.getenv(k):
            assert os.getenv(k) not in r.text, f"端点导出泄漏 {k}"
    print("[PASS] 端点(TestClient)：正常导出 200 + 附件 + Key 安全")

    # 2) completed 但无行程数据 -> 400
    server.app.state.session_manager = _SM(
        {"status": "completed", "last_response": {"result": {"destination": "东京"}}}
    )
    r = TestClient(server.app).get("/agent/export/u1/s1/t1")
    assert r.status_code == 400, "无行程数据应 400"
    print("[PASS] 端点(TestClient)：completed 无数据 -> 400")

    # 3) interrupted -> 400（未完成的草稿不允许导出）
    server.app.state.session_manager = _SM(
        {"status": "interrupted", "last_response": {"partial": {"itinerary_data": _sample_result()}}}
    )
    r = TestClient(server.app).get("/agent/export/u1/s1/t1")
    assert r.status_code == 400, "未完成应 400"
    assert "尚未完成" in r.json().get("detail", ""), "应给出清晰提示"
    print("[PASS] 端点(TestClient)：interrupted 未完成 -> 400")

    # 4) 任务不存在 -> 404
    server.app.state.session_manager = _SM(exists=False)
    r = TestClient(server.app).get("/agent/export/u1/s1/t1")
    assert r.status_code == 404, "不存在应 404"
    print("[PASS] 端点(TestClient)：任务不存在 -> 404")


def _poll_complete(uid, sid, tid, deadline):
    while time.time() < deadline:
        st = requests.get(f"{BASE}/agent/status/{uid}/{sid}/{tid}", timeout=10).json()
        if st["status"] in ("completed", "error"):
            return st
        if st["status"] == "interrupted":
            idata = (st.get("last_response") or {}).get("interrupt_data") or {}
            kind = idata.get("kind")
            action = "accept" if kind == "review_itinerary" else "save"
            requests.post(f"{BASE}/agent/resume", json={
                "user_id": uid, "session_id": sid, "task_id": tid,
                "command": {"action": action},
            }, timeout=10)
        time.sleep(2)
    return {"status": "timeout"}


def e2e_export():
    try:
        requests.get(f"{BASE}/system/info", timeout=5)
    except Exception:  # noqa: BLE001
        print("[SKIP] 端到端：后端未运行，跳过（导出逻辑已由单元/TestClient 覆盖）")
        return
    uname = f"e2e_{uuid.uuid4().hex[:8]}"
    requests.post(f"{BASE}/auth/register", json={"username": uname, "password": "pass123"}, timeout=10)
    uid = requests.post(f"{BASE}/auth/login", json={"username": uname, "password": "pass123"}, timeout=10).json()["user_id"]
    tid, sid = str(uuid.uuid4()), str(uuid.uuid4())
    requests.post(f"{BASE}/agent/invoke", json={
        "user_id": uid, "session_id": sid, "task_id": tid,
        "query": "9月2号到9月3号去杭州，轻松点",
    }, timeout=10)
    st = _poll_complete(uid, sid, tid, time.time() + 420)
    assert st["status"] == "completed", f"行程未完成，实际 {st['status']}"
    result = (st.get("last_response") or {}).get("result") or {}
    assert result.get("itinerary_data"), "应有结构化行程"
    dest = result.get("destination") or "杭州"

    r = requests.get(f"{BASE}/agent/export/{uid}/{sid}/{tid}", timeout=15)
    assert r.status_code == 200, f"导出应 200，实际 {r.status_code}"
    assert "text/html" in r.headers.get("content-type", "")
    assert "attachment" in r.headers.get("content-disposition", "")
    assert dest in r.text, f"导出 HTML 应包含目的地 {dest}"
    leaked = [k for k in SENSITIVE_ENV if os.getenv(k) and os.getenv(k) in r.text]
    assert not leaked, f"真实导出泄漏了服务端 Key：{leaked}"

    name = r.headers.get("content-disposition", "")
    import re
    m = re.search(r"filename\*=UTF-8''([^;]+)", name)
    fname = __import__("urllib.parse", fromlist=["unquote"]).unquote(m.group(1)) if m else "itinerary.html"
    days = result["itinerary_data"]["days"]
    real_img = sum(1 for d in days for it in d["items"] if (it.get("image") or "").startswith("http"))
    print(f"\n真实导出：{dest}，{len(days)} 天，{real_img} 个真实图片，文件 {fname}")
    print(f"导出 HTML 长度：{len(r.text)} 字符")
    print("[PASS] 端到端：真实行程 -> /agent/export -> 独立 HTML（Key 安全）")


def e2e_do_export():
    """真实前端处理函数 do_export：后端导出 -> 前端写盘 -> 文件可读（无需再跑一次 LLM）。"""
    try:
        requests.get(f"{BASE}/system/info", timeout=5)
    except Exception:  # noqa: BLE001
        print("[SKIP] 端到端：后端未运行，跳过 do_export")
        return
    import asyncio
    import webapp
    from utils.session_manager import get_session_manager

    uname = f"exp_{uuid.uuid4().hex[:8]}"
    requests.post(f"{BASE}/auth/register", json={"username": uname, "password": "pass123"}, timeout=10)
    uid = requests.post(f"{BASE}/auth/login", json={"username": uname, "password": "pass123"}, timeout=10).json()["user_id"]
    sid, tid = str(uuid.uuid4()), str(uuid.uuid4())

    async def seed():
        sm = get_session_manager()
        if not await sm.session_task_id_exists(uid, sid, tid):
            await sm.create_session(
                user_id=uid, session_id=sid, task_id=tid,
                status="idle", last_updated=time.time(), ttl=3600,
            )
        await sm.update_session(
            uid, sid, tid, status="completed",
            last_response={"result": _sample_result()},
        )
        await sm.close()

    asyncio.run(seed())

    msg, path = webapp.do_export({"user_id": uid, "session_id": sid, "task_id": tid})
    assert path and os.path.exists(path), f"应写出导出文件，msg={msg}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "东京" in content and "浅草寺" in content, "导出文件应含行程内容"
    assert not any(os.getenv(k) and os.getenv(k) in content for k in SENSITIVE_ENV), "导出文件泄漏服务端 Key"
    print(f"[PASS] 端到端：do_export 写出文件 {os.path.basename(path)}（{len(content)} 字符）")


def main():
    unit_export_html()
    unit_escape()
    unit_filename()
    unit_no_keys()
    endpoint_testclient()
    e2e_export()
    e2e_do_export()
    print("\nALL PHASE8 CASES PASSED")


if __name__ == "__main__":
    main()

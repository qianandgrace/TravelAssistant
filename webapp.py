"""Gradio 前端：旅游行程规划 Agent 的 Web 交互界面。

依赖后端 FastAPI（server.py，默认 http://localhost:8001）：
- 登录后自动打开最近一次会话，无则新建
- 对话式输入（自然语言）-> 异步提交 -> task_id 立即返回 -> 随时刷新查询状态
- HITL：行程审阅（接受/编辑/拒绝）与保存记忆确认（保存/跳过）
- 历史会话管理（切换 / 删除）、动态调整会话过期时间（TTL）、写入长期记忆
"""
import json
import os
import time
import uuid

import gradio as gr
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

# ---------- 后端 API 封装 ----------
def _req(method, path, **kw):
    r = requests.request(method, f"{API_BASE_URL}{path}", timeout=30, **kw)
    r.raise_for_status()
    return r.json()


def api_active_session(uid):
    return _req("GET", f"/agent/active/sessionid/{uid}")


def api_sessionids(uid):
    return _req("GET", f"/agent/sessionids/{uid}")


def api_tasks(uid, sid):
    return _req("GET", f"/agent/tasks/{uid}/{sid}")


def api_status(uid, sid, tid):
    return _req("GET", f"/agent/status/{uid}/{sid}/{tid}")


def api_invoke(uid, sid, query):
    payload = {"user_id": uid, "query": query}
    if sid:
        payload["session_id"] = sid
    return _req("POST", "/agent/invoke", json=payload)


def api_resume(uid, sid, tid, command):
    payload = {"user_id": uid, "session_id": sid, "task_id": tid, "command": command}
    return _req("POST", "/agent/resume", json=payload)


def api_set_ttl(uid, sid, ttl):
    payload = {"user_id": uid, "session_id": sid, "ttl": ttl}
    return _req("POST", "/session/ttl", json=payload)


def api_write_note(uid, text):
    return _req("POST", "/agent/write/longterm", json={"user_id": uid, "memory_info": text})


def api_register(username, password):
    return _req("POST", "/auth/register", json={"username": username, "password": password})


def api_login(username, password):
    return _req("POST", "/auth/login", json={"username": username, "password": password})


def _detail(exc):
    """从 requests.HTTPError 里取出后端返回的 detail 提示。"""
    try:
        return exc.response.json().get("detail") or str(exc)
    except Exception:  # noqa: BLE001
        return str(exc)


def api_delete_session(uid, sid):
    return _req("DELETE", f"/agent/session/{uid}/{sid}")


def _session_choices(uid):
    try:
        ids = api_sessionids(uid).get("session_ids", [])
    except Exception:
        return []
    return [(f"{sid[:8]}…", sid) for sid in ids]


def _find_interrupted_task(uid, sid):
    try:
        for t in api_tasks(uid, sid).get("task_ids", []):
            if t.get("status") == "interrupted":
                return t["task_id"]
    except Exception:
        pass
    return ""


def _interrupt_signature(idata: dict) -> tuple:
    """中断内容签名：同一中断不重渲染裁决面板，避免定时器冲掉用户的选择。"""
    kind = idata.get("kind")
    if kind == "review_itinerary":
        return ("review_itinerary", idata.get("itinerary", ""))
    if kind == "confirm_memory":
        return ("confirm_memory", tuple(idata.get("knowledge") or []))
    return ("other", str(idata))


# ---------- 注册 / 登录 ----------
def do_register(username, password, confirm):
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return "请输入用户名和密码。"
    if len(password) < 6:
        return "密码至少 6 位。"
    if password != (confirm or ""):
        return "两次输入的密码不一致。"
    try:
        api_register(username, password)
        return "注册成功！请切换到「登录」标签登录。"
    except requests.HTTPError as e:
        return _detail(e)
    except Exception as e:  # noqa: BLE001
        return f"注册失败：{e}"


def do_login(username, password, ui, chatbot):
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return "请输入用户名和密码。", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ui, chatbot, gr.update(choices=[], value=None)
    try:
        r = api_login(username, password)
    except requests.HTTPError as e:
        return _detail(e), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ui, chatbot, gr.update(choices=[], value=None)
    except Exception as e:  # noqa: BLE001
        return f"登录失败：{e}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ui, chatbot, gr.update(choices=[], value=None)

    uid = r["user_id"]
    auth_state = {"username": r["username"], "user_id": uid}
    ui = {"user_id": uid, "session_id": "", "task_id": "",
          "shown_interrupt": "", "last_shown_task": ""}
    active = r.get("active_session_id") or ""
    if active:
        ui["session_id"] = active
        ui["task_id"] = _find_interrupted_task(uid, active)
        msg = f"已自动打开最近一次会话：{active[:8]}…"
    else:
        msg = "没有最近会话，首次发送时将新建会话。"
    header = f"**当前用户：{r['username']}**"
    sid_show = ui["session_id"][:8] + "…" if ui["session_id"] else "（新建）"
    return (f"{msg}\n会话：{sid_show}", auth_state,
            gr.update(visible=False), gr.update(visible=True), header, msg, ui, [],
            gr.update(choices=_session_choices(uid), value=None))


def do_logout(ui, chatbot):
    auth_state = {}
    ui = {}
    return (auth_state, gr.update(visible=True), gr.update(visible=False),
            "", "### 请先登录", ui, [], gr.update(choices=[], value=None), "", "")


def send(query, ui, chatbot):
    uid = ui.get("user_id")
    if not uid:
        return "请先登录。", ui, chatbot, ""
    query = (query or "").strip()
    if not query:
        return "请输入内容。", ui, chatbot, ""
    sid, tid = ui.get("session_id"), ui.get("task_id")
    if sid and tid:
        try:
            if api_status(uid, sid, tid).get("status") == "interrupted":
                return "当前会话有未处理的裁决，请先在下方处理，再发起新请求。", ui, chatbot, ""
        except Exception:
            pass
    try:
        r = api_invoke(uid, sid, query)
    except Exception as e:
        return f"提交失败：{e}", ui, chatbot, ""
    ui["session_id"] = r["session_id"]
    ui["task_id"] = r["task_id"]
    ui["shown_interrupt"] = ""
    chatbot = list(chatbot or []) + [{"role": "user", "content": query}]
    return (f"已提交，task_id={r['task_id'][:8]}…\n随时点击「刷新状态」查看进度。",
            ui, chatbot, "")


def refresh(ui, chatbot, interrupt_md, action_radio):
    uid, sid, tid = ui.get("user_id"), ui.get("session_id"), ui.get("task_id")
    if not (uid and sid and tid):
        return "请先登录并发起一次任务。", ui, chatbot, interrupt_md, gr.update(choices=[], value=None)
    try:
        st = api_status(uid, sid, tid)
    except Exception as e:
        return f"查询失败：{e}", ui, chatbot, interrupt_md, action_radio
    status, resp = st.get("status"), st.get("last_response") or {}

    if status in ("pending", "running"):
        parsed = resp.get("parsed") or {}
        if parsed:
            msg = (f"任务运行中… 已识别：目的地={parsed.get('destination')}　"
                   f"{parsed.get('days')}天　偏好={parsed.get('preference') or '无'}\n"
                   f"（{parsed.get('start_date')} ~ {parsed.get('end_date')}）")
        else:
            msg = "任务运行中，请稍候刷新…"
        return msg, ui, chatbot, interrupt_md, gr.update(choices=[], value=None)

    if status == "interrupted":
        idata = resp.get("interrupt_data") or {}
        kind = idata.get("kind")
        if kind == "review_itinerary":
            md = f"**请审阅以下生成的行程**\n\n{idata.get('itinerary', '')}"
            choices = ["接受", "编辑", "拒绝"]
        elif kind == "confirm_memory":
            ks = idata.get("knowledge") or []
            lines = "\n".join(f"{i + 1}. {k}" for i, k in enumerate(ks)) or "（无新知识）"
            md = f"**保存长期记忆确认**\n\n{lines}"
            choices = ["保存", "跳过"]
        else:
            md = f"中断数据：{idata}"
            choices = []
        sig = _interrupt_signature(idata)
        if ui.get("shown_interrupt") == sig:
            # 同一中断已渲染，直接透传，不清掉用户正在做的选择
            return "会话中断，请在下方做出选择。", ui, chatbot, interrupt_md, action_radio
        ui["shown_interrupt"] = sig
        return "会话中断，请在下方做出选择。", ui, chatbot, md, gr.update(choices=choices, value=None)

    if status == "completed":
        result = resp.get("result") or {}
        itinerary = result.get("itinerary")
        if ui.get("last_shown_task") != tid and itinerary:
            chatbot = list(chatbot or []) + [
                {"role": "assistant", "content": f"**行程规划完成**\n\n{itinerary}"}
            ]
            ui["last_shown_task"] = tid
        parsed = result.get("parsed") or {}
        ent = (f"已识别：目的地={parsed.get('destination')}　{parsed.get('days')}天　"
               f"偏好={parsed.get('preference') or '无'}") if parsed else ""
        mem = f"　{result.get('memory_saved')}" if result.get("memory_saved") else ""
        return f"任务完成。{ent}{mem}", ui, chatbot, "", gr.update(choices=[], value=None)

    if status == "error":
        return f"任务出错：{resp.get('message', '未知错误')}", ui, chatbot, "", gr.update(choices=[], value=None)

    return f"当前状态：{status}", ui, chatbot, interrupt_md, action_radio


def resolve(ui, action, text):
    uid, sid, tid = ui.get("user_id"), ui.get("session_id"), ui.get("task_id")
    if not (uid and sid and tid):
        return "请先登录并发起任务。", ui, gr.update(choices=[], value=None), ""
    mapping = {"接受": "accept", "编辑": "edit", "拒绝": "reject", "保存": "save", "跳过": "skip"}
    act = mapping.get(action, "accept")
    command = {"action": act}
    if act in ("edit", "reject") and (text or "").strip():
        command["text"] = text.strip()
    try:
        api_resume(uid, sid, tid, command)
        return "已提交裁决，任务继续运行，可点击「刷新状态」查看进度。", ui, gr.update(choices=[], value=None), ""
    except Exception as e:
        return f"提交裁决失败：{e}", ui, gr.update(choices=[], value=None), ""


def switch_session(ui, selected):
    if not selected:
        return "请先选择一个历史会话。", ui, gr.update()
    ui["session_id"] = selected
    ui["task_id"] = _find_interrupted_task(ui.get("user_id", ""), selected)
    ui["last_shown_task"] = ""
    ui["shown_interrupt"] = ""
    return f"已切换到会话 {selected[:8]}…", ui, gr.update()


def delete_session(ui, selected):
    if not selected:
        return "请先选择一个历史会话。", ui, gr.update(choices=[], value=None)
    try:
        api_delete_session(ui.get("user_id", ""), selected)
        if ui.get("session_id") == selected:
            ui["session_id"] = ""
            ui["task_id"] = ""
        return f"会话 {selected[:8]}… 已删除。", ui, gr.update(choices=_session_choices(ui.get("user_id", "")), value=None)
    except Exception as e:
        return f"删除失败：{e}", ui, gr.update(choices=_session_choices(ui.get("user_id", "")), value=None)


def set_ttl(ui, ttl):
    uid, sid = ui.get("user_id"), ui.get("session_id")
    if not uid:
        return "请先登录。"
    ttl = int(ttl or 3600)
    try:
        r = api_set_ttl(uid, sid, ttl)
        return f"已调整过期时间为 {r.get('ttl')} 秒，涉及 {r.get('affected_tasks')} 个任务。"
    except Exception as e:
        return f"设置 TTL 失败：{e}"


def write_note(ui, note):
    uid = ui.get("user_id")
    if not uid:
        return "请先登录。", ""
    note = (note or "").strip()
    if not note:
        return "请输入内容。", ""
    try:
        api_write_note(uid, note)
        return "已写入长期记忆。", ""
    except Exception as e:
        return f"写入失败：{e}", ""


# ---------- UI ----------
with gr.Blocks(title="旅游行程规划 Agent") as demo:
    auth_state = gr.State({})
    ui = gr.State({})

    # ===== 认证视图（登录 / 注册）=====
    with gr.Group() as auth_view:
        gr.Markdown("# 旅游行程规划 Agent\n请先注册，再登录进入对话。")
        with gr.Tabs():
            with gr.Tab("登录"):
                login_user = gr.Textbox(label="用户名", placeholder="已注册的用户名")
                login_pwd = gr.Textbox(label="密码", type="password")
                login_btn = gr.Button("登录", variant="primary")
                login_msg = gr.Markdown("")
            with gr.Tab("注册"):
                reg_user = gr.Textbox(label="用户名（2-32 字符，字母/数字/_/-/中文）")
                reg_pwd = gr.Textbox(label="密码（至少 6 位）", type="password")
                reg_pwd2 = gr.Textbox(label="确认密码", type="password")
                reg_btn = gr.Button("注册", variant="primary")
                reg_msg = gr.Markdown("")

    # ===== 对话视图（登录后显示）=====
    with gr.Group(visible=False) as chat_view:
        with gr.Row():
            header = gr.Markdown("")
            with gr.Row():
                logout_btn = gr.Button("退出登录")
                refresh_btn = gr.Button("刷新状态")

        status_md = gr.Markdown("### 请先登录")

        chatbot = gr.Chatbot(label="对话", height=380)

        query_box = gr.Textbox(
            label="输入出行需求（自然语言）",
            placeholder="如：9月2号到9月5号去天津，轻松点",
            lines=2,
        )
        send_btn = gr.Button("发送", variant="primary")

        interrupt_md = gr.Markdown("")
        with gr.Row():
            action_radio = gr.Radio(choices=[], label="请选择", interactive=True, scale=1)
            action_text = gr.Textbox(label="意见 / 编辑内容（编辑或拒绝时填写）", lines=4, scale=2)
        resolve_btn = gr.Button("提交裁决")

        with gr.Accordion("历史会话", open=False):
            session_dropdown = gr.Dropdown(label="历史会话", choices=[], interactive=True)
            with gr.Row():
                switch_btn = gr.Button("切换会话")
                del_session_btn = gr.Button("删除会话")

        with gr.Accordion("偏好 / 长期记忆", open=False):
            note_box = gr.Textbox(label="写入内容（如：我喜欢历史，不吃辣）", lines=2)
            note_btn = gr.Button("写入长期记忆")

        with gr.Accordion("会话过期时间", open=False):
            ttl_box = gr.Number(label="TTL（秒）", value=3600, precision=0)
            ttl_btn = gr.Button("设置 TTL")

    # ---- 认证：注册 / 登录 / 退出 ----
    reg_btn.click(do_register, [reg_user, reg_pwd, reg_pwd2], [reg_msg])
    login_btn.click(do_login, [login_user, login_pwd, ui, chatbot],
                    [login_msg, auth_state, auth_view, chat_view, header, status_md, ui, chatbot, session_dropdown])
    logout_btn.click(do_logout, [ui, chatbot],
                     [auth_state, auth_view, chat_view, header, status_md, ui, chatbot,
                      session_dropdown, login_pwd, reg_pwd])

    # ---- 对话 ----
    send_btn.click(send, [query_box, ui, chatbot], [status_md, ui, chatbot, query_box])
    refresh_btn.click(refresh, [ui, chatbot, interrupt_md, action_radio],
                      [status_md, ui, chatbot, interrupt_md, action_radio])
    resolve_btn.click(resolve, [ui, action_radio, action_text],
                      [status_md, ui, action_radio, action_text])
    switch_btn.click(switch_session, [ui, session_dropdown], [status_md, ui, session_dropdown])
    del_session_btn.click(delete_session, [ui, session_dropdown], [status_md, ui, session_dropdown])
    ttl_btn.click(set_ttl, [ui, ttl_box], [status_md])
    note_btn.click(write_note, [ui, note_box], [status_md, note_box])

    gr.Timer(3).tick(refresh, [ui, chatbot, interrupt_md, action_radio],
                     [status_md, ui, chatbot, interrupt_md, action_radio])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

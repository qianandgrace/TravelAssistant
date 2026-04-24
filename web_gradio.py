import gradio as gr
import requests

state = {"thread_id": None, "interrupt": None}


def send_message(user_id, message, history):
    resp = requests.post(
        "http://127.0.0.1:8000/chat",
        json={
            "user_id": user_id,
            "message": message,
            "thread_id": state["thread_id"]
        }
    ).json()

    state["thread_id"] = resp["thread_id"]
    state["interrupt"] = resp["interrupt"]

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": resp["response"]})

    return history


with gr.Blocks() as demo:
    user_id = gr.Textbox(value="gq", label="用户ID")

    chatbot = gr.Chatbot(type="messages")  # ✅ 关键

    msg = gr.Textbox()

    btn = gr.Button("发送")

    btn.click(
        send_message,
        inputs=[user_id, msg, chatbot],
        outputs=chatbot
    )

demo.launch()
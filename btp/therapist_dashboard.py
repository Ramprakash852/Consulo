import json
import os
import time

import streamlit as st

CHAT_FILE = "chat_data.json"


st.set_page_config(
    page_title="Therapist Dashboard",
    page_icon="👨‍⚕️",
    layout="wide",
)


def load_chat():
    if not os.path.exists(CHAT_FILE):
        return []

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as file:
            messages = json.load(file)
            return messages if isinstance(messages, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_chat(messages):
    with open(CHAT_FILE, "w", encoding="utf-8") as file:
        json.dump(messages, file, indent=2, ensure_ascii=False)


def render_chat(messages):
    for message in messages[-10:]:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        display_role = "assistant" if role == "system" else role

        with st.chat_message(display_role):
            st.write(content)


st.title("👨‍⚕️ Therapist Dashboard")
st.caption("Read and reply to the shared chat file.")
st.caption("🔄 Auto-refreshing every 2 seconds...")

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Refresh Chat", use_container_width=True):
        st.rerun()

chat_history = load_chat()
render_chat(chat_history)

st.divider()
st.subheader("Therapist Reply")
therapist_reply = st.chat_input("Type therapist reply...")

# Mark typing state
if therapist_reply:
    st.session_state["typing"] = True
else:
    st.session_state["typing"] = False

if therapist_reply:
    chat_history = load_chat()
    chat_history.append({"role": "assistant", "content": f"👨‍⚕️ Therapist: {therapist_reply.strip()}"})
    save_chat(chat_history)
    st.rerun()

if not st.session_state.get("typing", False):
    time.sleep(2)
    st.rerun()

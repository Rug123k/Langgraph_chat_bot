import os
import hashlib
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from backend import chatbot, retrieve_all_threads, generate_user_thread_id, create_user, authenticate_user, normalize_email

load_dotenv()

# -------------------- Session State --------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "email" not in st.session_state:
    st.session_state["email"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

# -------------------- Helpers --------------------
def load_conversation(thread_id):
    config = {"configurable": {"thread_id": thread_id, "user_id": st.session_state["user_id"]}}
    state = chatbot.get_state(config=config)
    messages = state.values.get("messages", [])
    return messages

def add_thread(thread_id):
    if thread_id and thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    thread_id = generate_user_thread_id(st.session_state["user_id"])
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

# -------------------- Auth UI --------------------
if not st.session_state["logged_in"]:
    st.title("🔐 Login to LangGraph Chatbot")

    tabs = st.tabs(["Login", "Sign up"])

    # ----- Login tab -----
    with tabs[0]:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            ok, result = authenticate_user(login_email, login_password)
            if ok:
                st.session_state["email"] = normalize_email(login_email)
                st.session_state["user_id"] = result
                st.session_state["logged_in"] = True

                # Load threads
                st.session_state["chat_threads"] = retrieve_all_threads(st.session_state["user_id"])
                if st.session_state["chat_threads"]:
                    st.session_state["thread_id"] = st.session_state["chat_threads"][0]
                else:
                    st.session_state["thread_id"] = generate_user_thread_id(st.session_state["user_id"])

                # Load conversation
                msgs = load_conversation(st.session_state["thread_id"])
                st.session_state["message_history"] = [
                    {
                        "role": "user" if isinstance(m, HumanMessage) else "assistant",
                        "content": getattr(m, "content", ""),
                    }
                    for m in msgs
                ]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(result)

    # ----- Sign up tab -----
    with tabs[1]:
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password (min 8 chars)", type="password", key="signup_password")
        signup_password2 = st.text_input("Confirm Password", type="password", key="signup_password2")
        if st.button("Create Account"):
            if signup_password != signup_password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(signup_email, signup_password)
                if ok:
                    st.success("Account created. Please login.")
                else:
                    st.error(msg)

    # Optional: warn if LLM key missing
    if not os.getenv("OPENAI_API_KEY"):
        st.info("Set OPENAI_API_KEY in .env or Streamlit Secrets.")
else:
    # -------------------- Sidebar --------------------
    st.sidebar.title("LangGraph Chatbot")
    st.sidebar.write(f"Logged in as: {st.session_state['email']}")

    if st.sidebar.button("New Chat", key="new_chat_btn"):
        reset_chat()

    st.sidebar.header("My Conversations")
    for thread_id in st.session_state["chat_threads"]:
        if not thread_id:
            continue
        short = thread_id.split("_", 1)[-1][:8] + "..."
        if st.sidebar.button(short, key=f"thread_btn_{thread_id}"):
            st.session_state["thread_id"] = thread_id
            msgs = load_conversation(thread_id)
            st.session_state["message_history"] = [
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": getattr(m, "content", ""),
                }
                for m in msgs
            ]

    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state.clear()
        st.rerun()

    # -------------------- Main Chat UI --------------------
    # Initialize threads/thread_id if empty
    if not st.session_state["chat_threads"]:
        st.session_state["chat_threads"] = retrieve_all_threads(st.session_state["user_id"])
    if not st.session_state["thread_id"]:
        if st.session_state["chat_threads"]:
            st.session_state["thread_id"] = st.session_state["chat_threads"][0]
        else:
            st.session_state["thread_id"] = generate_user_thread_id(st.session_state["user_id"])
    add_thread(st.session_state["thread_id"])

    # Load initial history if empty
    if not st.session_state["message_history"]:
        msgs = load_conversation(st.session_state["thread_id"])
        st.session_state["message_history"] = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": getattr(m, "content", ""),
            }
            for m in msgs
        ]

    st.title("💬 Chat with AI")
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Type here")
    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        CONFIG = {
            "configurable": {
                "thread_id": st.session_state["thread_id"],
                "user_id": st.session_state["user_id"],
            }
        }

        with st.chat_message("assistant"):
            def generate_response():
                for message_chunk, _ in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)], "user_id": st.session_state["user_id"]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    yield message_chunk.content

            ai_msg = st.write_stream(generate_response())
            if ai_msg:
                st.session_state["message_history"].append({"role": "assistant", "content": ai_msg})
                add_thread(st.session_state["thread_id"])

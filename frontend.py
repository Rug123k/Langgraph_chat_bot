import streamlit as st
from backend import chatbot, retrieve_all_threads, generate_user_thread_id
from langchain_core.messages import HumanMessage, AIMessage  # <-- Includes AIMessage
import uuid
import hashlib
from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------- Twilio Setup --------------------
# Use st.secrets for deployed apps; fallback to os.getenv() for local dev
try:
    twilio_account_sid = st.secrets['TWILIO_ACCOUNT_SID']
    twilio_auth_token = st.secrets['TWILIO_AUTH_TOKEN']
    twilio_phone = st.secrets['TWILIO_PHONE_NUMBER']
except KeyError:
    # Fallback for local development (if .env is loaded)
    twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')

if not all([twilio_account_sid, twilio_auth_token, twilio_phone]):
    st.error("Twilio credentials not found. Please check your secrets or .env file.")
else:
    twilio_client = Client(twilio_account_sid, twilio_auth_token)
    TWILIO_PHONE = twilio_phone

# -------------------- Utility Functions --------------------
def hash_phone(phone: str) -> str:
    return hashlib.sha256(phone.encode()).hexdigest()

def send_otp(phone: str) -> str:
    otp = str(uuid.uuid4().int)[:6]
    try:
        twilio_client.messages.create(
            body=f"Your OTP for LangGraph Chatbot is: {otp}",
            from_=TWILIO_PHONE,
            to=phone
        )
        return otp
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")
        return None

# -------------------- Session State --------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'otp' not in st.session_state:
    st.session_state['otp'] = None
if 'phone' not in st.session_state:
    st.session_state['phone'] = None

# -------------------- Login Page --------------------
if not st.session_state['logged_in']:
    st.title("🔐 Login to LangGraph Chatbot")
    phone = st.text_input("Mobile Number (+91XXXXXXXXXX)", key="phone_input")
    if st.button("Send OTP"):
        if phone and phone.startswith('+91') and len(phone) == 13:
            st.session_state['otp'] = send_otp(phone)
            st.session_state['phone'] = phone
            if st.session_state['otp']:
                st.success("OTP sent! Check your SMS.")
        else:
            st.error("Enter a valid phone number in +91XXXXXXXXXX format.")

    otp_input = st.text_input("Enter 6-digit OTP", key="otp_input")
    if st.button("Verify OTP"):
        if st.session_state.get('otp') and otp_input == st.session_state['otp']:
            st.session_state['user_id'] = hash_phone(st.session_state['phone'])
            st.session_state['logged_in'] = True
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid OTP. Please try again.")

# -------------------- Chat UI --------------------
else:
    def reset_chat():
        thread_id = generate_user_thread_id(st.session_state['user_id'])
        st.session_state['thread_id'] = thread_id
        add_thread(thread_id)
        st.session_state['message_history'] = []

    def add_thread(thread_id):
        if thread_id and thread_id not in st.session_state['chat_threads']:
            st.session_state['chat_threads'].append(thread_id)

    def load_conversation(thread_id):
        config = {'configurable': {'thread_id': thread_id, 'user_id': st.session_state['user_id']}}
        state = chatbot.get_state(config=config)
        messages = state.values.get('messages', [])
        print(f"Loaded {len(messages)} messages for thread {thread_id}")  # Debug
        return messages

    # --- Initialize session ---
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = retrieve_all_threads(st.session_state['user_id'])
    
    if 'thread_id' not in st.session_state:
        if st.session_state['chat_threads']:
            st.session_state['thread_id'] = st.session_state['chat_threads'][-1]
        else:
            st.session_state['thread_id'] = generate_user_thread_id(st.session_state['user_id'])
    
    add_thread(st.session_state['thread_id'])

    if not st.session_state['message_history']:
        messages = load_conversation(st.session_state['thread_id'])
        st.session_state['message_history'] = [
            {'role': 'user' if isinstance(msg, HumanMessage) else 'assistant', 'content': msg.content}
            for msg in messages
        ]

    # --- Sidebar UI ---
    st.sidebar.title('LangGraph Chatbot')
    st.sidebar.write(f"Logged in as: {st.session_state['phone']}")
    if st.sidebar.button('New Chat'):
        reset_chat()
    st.sidebar.header('My Conversations')
    for thread_id in st.session_state['chat_threads'][::-1]:
        if thread_id:
            display_id = thread_id.split('_', 1)[1][:8] + '...'
            if st.sidebar.button(display_id):
                st.session_state['thread_id'] = thread_id
                messages = load_conversation(thread_id)
                st.session_state['message_history'] = [
                    {'role': 'user' if isinstance(msg, HumanMessage) else 'assistant', 'content': msg.content}
                    for msg in messages
                ]
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    # --- Main chat UI ---
    st.title("💬 Chat with AI")
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.write(message['content'])

    user_input = st.chat_input('Type here')
    if user_input:
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.write(user_input)

        CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"], "user_id": st.session_state["user_id"]}}

        with st.chat_message('assistant'):
            def generate_response():
                for message_chunk, _ in chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)], 'user_id': st.session_state["user_id"]},
                    config=CONFIG,
                    stream_mode='messages'
                ):
                    yield message_chunk.content

            ai_message = st.write_stream(generate_response())
            if ai_message:
                st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
                # Force save
                chatbot.update_state(  # <-- FIXED: Config first, then values
                    CONFIG,
                    {"messages": [HumanMessage(content=m['content']) if m['role'] == 'user' else AIMessage(content=m['content']) for m in st.session_state['message_history']], "user_id": st.session_state["user_id"]}
                )

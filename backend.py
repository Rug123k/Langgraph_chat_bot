from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Tuple
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import hashlib
import uuid
import time
import re
from collections import defaultdict
import bcrypt

load_dotenv()

# ----- LLM -----
# Pick a model you have access to
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ----- SQLite (shared for checkpoints + users) -----
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")

# ----- Users table (email/password) -----
def init_users_table():
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        user_id TEXT UNIQUE NOT NULL,
        created_at INTEGER NOT NULL
    )
    """)
    conn.commit()

def normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))

def email_to_user_id(email: str) -> str:
    # Deterministic user_id for LangGraph based on email
    return hashlib.sha256(email.encode("utf-8")).hexdigest()

def create_user(email: str, password: str) -> Tuple[bool, str]:
    """
    Returns (ok, message). On success, message is 'created'.
    """
    email_n = normalize_email(email)
    if not is_valid_email(email_n):
        return False, "Invalid email format."
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters."

    user_id = email_to_user_id(email_n)
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, user_id, created_at) VALUES (?, ?, ?, ?)",
            (email_n, pw_hash, user_id, int(time.time()))
        )
        conn.commit()
        return True, "created"
    except sqlite3.IntegrityError:
        return False, "Email already exists."

def authenticate_user(email: str, password: str) -> Tuple[bool, str]:
    """
    Returns (ok, user_id_or_error_message).
    """
    email_n = normalize_email(email)
    cur = conn.execute("SELECT password_hash, user_id FROM users WHERE email = ?", (email_n,))
    row = cur.fetchone()
    if not row:
        return False, "User not found."
    stored_hash, user_id = row
    if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
        return True, user_id
    return False, "Invalid password."

init_users_table()

# ----- LangGraph state -----
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response], "user_id": state["user_id"]}

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads(user_id: str):
    """
    Return thread_ids for this user_id sorted by most recent activity.
    """
    thread_timestamps = defaultdict(list)
    try:
        for checkpoint in checkpointer.list(None):
            config = checkpoint.config or {}
            cfg = config.get("configurable") or {}
            thread_id = cfg.get("thread_id")
            uid = cfg.get("user_id")
            if not thread_id or uid != user_id:
                continue

            ts = 0
            if hasattr(checkpoint, "ts") and checkpoint.ts is not None:
                ts = checkpoint.ts
            elif getattr(checkpoint, "metadata", None) and "ts" in checkpoint.metadata:
                ts = checkpoint.metadata["ts"]
            thread_timestamps[thread_id].append(ts)

        sorted_threads = sorted(
            thread_timestamps.keys(),
            key=lambda t: max(thread_timestamps[t]) if thread_timestamps[t] else 0,
            reverse=True,
        )
    except Exception as e:
        print(f"Error retrieving threads: {e}. Returning unsorted threads.")
        sorted_threads = list(thread_timestamps.keys())

    return sorted_threads

def generate_user_thread_id(user_id: str):
    return f"{user_id}_{uuid.uuid4().hex}"
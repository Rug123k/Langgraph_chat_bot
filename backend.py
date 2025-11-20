from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import hashlib
import uuid
from collections import defaultdict

load_dotenv()
llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    print(f"Saving state for user {state['user_id']}: {len(messages)} messages")  # Debug
    return {"messages": [response], "user_id": state["user_id"]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads(user_id: str):
    thread_timestamps = defaultdict(list)
    
    try:
        for checkpoint in checkpointer.list(None):
            config = checkpoint.config
            thread_id = config['configurable'].get('thread_id')
            if thread_id and config['configurable'].get('user_id') == user_id:
                timestamp = 0
                if hasattr(checkpoint, 'ts'):
                    timestamp = checkpoint.ts
                elif hasattr(checkpoint, 'metadata') and 'ts' in checkpoint.metadata:
                    timestamp = checkpoint.metadata['ts']
                thread_timestamps[thread_id].append(timestamp)
        
        sorted_threads = sorted(thread_timestamps.keys(), key=lambda t: max(thread_timestamps[t]), reverse=True)
    except Exception as e:
        print(f"Error retrieving threads: {e}. Returning unsorted threads.")
        sorted_threads = list(thread_timestamps.keys())
    
    return sorted_threads

def generate_user_thread_id(user_id: str):
    return f"{user_id}_{uuid.uuid4().hex}"
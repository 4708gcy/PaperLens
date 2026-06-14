"""langgraph 1.2.5 checkpointer 多轮记忆最小验证"""
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from app.config import settings


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # reducer 追加


llm = ChatOpenAI(
    api_key=settings["llm"]["api_key"],
    base_url=settings["llm"]["base_url"],
    model=settings["llm"]["fast_model"],  # qwen-turbo 省钱
    temperature=0.7,
)


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=InMemorySaver())

# 测试多轮记忆
config = {"configurable": {"thread_id": "user-001"}}

print("=== 第一轮 ===")
r1 = graph.invoke({"messages": [HumanMessage(content="我叫张三，请记住我的名字")]}, config)
print("AI:", r1["messages"][-1].content[:100])

print("\n=== 第二轮（验证是否记住）===")
r2 = graph.invoke({"messages": [HumanMessage(content="我叫什么名字？")]}, config)
print("AI:", r2["messages"][-1].content[:100])

if "张三" in r2["messages"][-1].content:
    print("\n[OK] checkpointer 多轮记忆验证通过")
else:
    print("\n[FAIL] AI 没记住名字，checkpointer 可能没生效")

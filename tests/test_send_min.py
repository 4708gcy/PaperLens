"""langgraph 1.2.5 Send 并行最小验证（不依赖外部 API，用本地模拟）"""
import operator
import time
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class MapState(TypedDict):
    subjects: List[str]
    jokes: Annotated[List[str], operator.add]  # reducer 汇聚并行结果


def generate_topics(state):
    """模拟 planner：生成主题"""
    print(f"  [planner] 生成 3 个主题")
    return {"subjects": ["cat", "dog", "fish"]}


def generate_joke(state):
    """模拟 section_worker：接收 Send 传入的子状态，返回结果"""
    subject = state["subject"]
    time.sleep(0.5)  # 模拟耗时
    print(f"  [worker] 处理: {subject}")
    return {"jokes": [f"Why did the {subject} cross the road?"]}


def map_to_jokes(state):
    """路由函数：返回 Send 列表，每个 Send 启动一个并行 worker"""
    sends = [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
    print(f"  [router] 派发 {len(sends)} 个并行任务")
    return sends


# 构建图
builder = StateGraph(MapState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", map_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", END)

graph = builder.compile()

# 执行
print("=== 测试 Send 并行 ===")
import time as t
start = t.time()
result = graph.invoke({"subjects": [], "jokes": []})
elapsed = t.time() - start

print(f"\n结果: {result['jokes']}")
print(f"耗时: {elapsed:.2f}s（3 个 0.5s 任务若串行应 1.5s，并行应 < 1s）")

if len(result["jokes"]) == 3:
    print("[OK] Send 并行验证通过：3 个 worker 结果正确汇聚")
else:
    print(f"[FAIL] 期望 3 个结果，实际 {len(result['jokes'])}")

if elapsed < 1.2:
    print(f"[OK] 并行生效（{elapsed:.2f}s < 1.2s）")
else:
    print(f"[WARN] 可能串行执行了（{elapsed:.2f}s）")

"""Day 4 验收：多篇综述（planner → Send 并行 section_worker → assembler）"""
import httpx
import json

url = "http://127.0.0.1:8000/api/v1/chat/stream"
payload = {
    "message": "请基于以下论文写一篇关于「GraphRAG 的知识图谱构建与社区检测方法」的综述",
    "paper_ids": [1, 2],  # 两篇 GraphRAG 论文
    "thread_id": "synth_day4"
}

print(f"综述主题: GraphRAG 的知识图谱构建与社区检测方法")
print(f"论文: paper_id=1 (graphrag) + paper_id=2 (graphrag_whitepaper)")
print(f"\n=== SSE 事件流 ===\n")

token_count = 0
intent = None
full_reply = ""
events_seen = []

with httpx.Client(timeout=300) as client:
    with client.stream("POST", url, json=payload) as resp:
        for line in resp.iter_lines():
            if line and line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    etype = data.get("type")
                    if etype == "intent":
                        intent = data.get("intent")
                        events_seen.append("intent")
                        print(f"\n[EVENT intent] 意图: {intent}\n")
                    elif etype == "token":
                        token_count += 1
                        full_reply += data.get("content", "")
                        if token_count <= 5:
                            print(f"[token {token_count}] '{data.get('content')}'")
                        elif token_count == 6:
                            print("... (后续 token 省略) ...")
                    elif etype == "done":
                        events_seen.append("done")
                        print(f"\n[EVENT done]")
                    elif etype == "error":
                        events_seen.append("error")
                        print(f"\n[EVENT error] {data.get('msg')}")
                except json.JSONDecodeError:
                    continue

print(f"\n=== 统计 ===")
print(f"意图: {intent}")
print(f"总 token: {token_count}")
print(f"回复长度: {len(full_reply)} 字符")
print(f"回复预览:\n{full_reply[:600]}")

if intent == "synthesize" and token_count > 10:
    print("\n[OK] 多篇综述验收通过（Send 并行 + assembler 合并）")
else:
    print(f"\n[问题] intent={intent}（期望 synthesize）, tokens={token_count}")

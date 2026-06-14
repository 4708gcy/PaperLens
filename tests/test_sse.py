"""Day 2 验收 3/3：SSE 流式逐字输出测试"""
import httpx
import json

url = "http://127.0.0.1:8000/api/v1/chat/stream"
payload = {
    "message": "GraphRAG 用了什么社区检测算法？",
    "paper_ids": [1],
    "thread_id": "sse_test"
}

print(f"请求: {payload['message']}\n")
print("=== SSE 事件流 ===\n")

token_count = 0
intent = None
full_reply = ""

with httpx.Client(timeout=120) as client:
    with client.stream("POST", url, json=payload) as resp:
        for line in resp.iter_lines():
            if line and line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    etype = data.get("type")
                    if etype == "intent":
                        intent = data.get("intent")
                        print(f"\n[EVENT intent] 意图: {intent}\n")
                    elif etype == "token":
                        token_count += 1
                        full_reply += data.get("content", "")
                        # 只打印前 10 个 token 和最后几个，避免刷屏
                        if token_count <= 10:
                            print(f"[token {token_count}] '{data.get('content')}'")
                        elif token_count == 11:
                            print("... (后续 token 省略打印) ...")
                    elif etype == "done":
                        print(f"\n[EVENT done] 意图: {data.get('intent')}")
                    elif etype == "error":
                        print(f"\n[EVENT error] {data.get('msg')}")
                except json.JSONDecodeError:
                    continue

print(f"\n=== 统计 ===")
print(f"总 token 数: {token_count}")
print(f"检测意图: {intent}")
print(f"完整回复长度: {len(full_reply)} 字符")
print(f"回复预览: {full_reply[:200]}")

if token_count > 5 and intent:
    print("\n[OK] SSE 流式验收通过")
else:
    print("\n[FAIL] SSE 流式异常")

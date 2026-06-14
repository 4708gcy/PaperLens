"""后端 API 客户端"""
import json
import httpx
from typing import Generator, Dict, Any, List

BASE_URL = "http://127.0.0.1:8000/api/v1"


def upload_document(file) -> dict:
    """上传论文"""
    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{BASE_URL}/documents/upload",
            files={"file": (file.name, file, "application/octet-stream")}
        )
    return response.json()


def list_documents() -> dict:
    """列出所有论文"""
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{BASE_URL}/documents/list")
    return response.json()


def delete_document(paper_id: int) -> dict:
    """删除论文"""
    with httpx.Client(timeout=30) as client:
        response = client.delete(f"{BASE_URL}/documents/{paper_id}")
    return response.json()


def chat(message: str, paper_ids: List[int], thread_id: str = "default") -> dict:
    """普通对话"""
    with httpx.Client(timeout=180) as client:
        response = client.post(
            f"{BASE_URL}/chat/",
            json={"message": message, "paper_ids": paper_ids, "thread_id": thread_id}
        )
    return response.json()


def chat_stream(message: str, paper_ids: List[int], thread_id: str = "default") -> Generator[dict, None, None]:
    """
    流式对话 —— 生成器，逐段 yield 事件 dict

    事件类型：
    - {"type": "intent", "intent": "qa"}
    - {"type": "token", "content": "..."}
    - {"type": "done", "intent": "..."}
    - {"type": "error", "msg": "..."}
    """
    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            json={"message": message, "paper_ids": paper_ids, "thread_id": thread_id}
        ) as response:
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        yield data
                    except json.JSONDecodeError:
                        continue


def analyze(paper_id: int, thread_id: str = "analyze") -> Generator[dict, None, None]:
    """结构化解读（流式）"""
    message = "请帮我分析这篇论文"
    yield from chat_stream(message, [paper_id], thread_id)


def synthesize(paper_ids: List[int], topic: str, thread_id: str = "synthesize") -> Generator[dict, None, None]:
    """多篇综述（流式）"""
    message = f"请基于以下论文写一篇关于「{topic}」的综述"
    yield from chat_stream(message, paper_ids, thread_id)

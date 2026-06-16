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


# ───────────────────────────────────────────────
# 学习助手（复用 chat/stream，body 带 learn_mode）
# ───────────────────────────────────────────────
def learn_stream(
    message: str,
    paper_ids: List[int],
    mode: str,
    thread_id: str = "learn",
) -> Generator[dict, None, None]:
    """
    学习助手流式对话。

    mode 取值：
    - qa       辅导式问答（chat 流式，逐字显示）
    - summary  一键总结（流式输出 Markdown 摘要）
    - flashcard 知识卡片（流式输出 JSON，前端在 done 后解析）
    - quiz     自测练习（流式输出 JSON，前端在 done 后解析）
    """
    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            json={
                "message": message,
                "paper_ids": paper_ids,
                "thread_id": thread_id,
                "learn_mode": mode,
            }
        ) as response:
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        yield data
                    except json.JSONDecodeError:
                        continue


def _as_id_list(paper_ids):
    """统一把单个 int 或 list 转成 list[int]，方便上层两种都能传"""
    if isinstance(paper_ids, int):
        return [paper_ids]
    return list(paper_ids)


def learn_qa(message: str, paper_ids, thread_id: str = "learn") -> Generator[dict, None, None]:
    """学习助手 · 辅导式问答（支持多篇资料）"""
    yield from learn_stream(message, _as_id_list(paper_ids), "qa", thread_id)


def learn_summary(paper_ids, thread_id: str = "learn") -> Generator[dict, None, None]:
    """学习助手 · 一键总结（支持多篇资料）"""
    yield from learn_stream("请生成这份资料的结构化学习摘要", _as_id_list(paper_ids), "summary", thread_id)


def learn_flashcard(paper_ids, thread_id: str = "learn") -> Generator[dict, None, None]:
    """学习助手 · 知识卡片（支持多篇资料，返回 JSON 文本，前端解析）"""
    yield from learn_stream("请基于这份资料生成知识闪卡", _as_id_list(paper_ids), "flashcard", thread_id)


def learn_quiz(paper_ids, thread_id: str = "learn") -> Generator[dict, None, None]:
    """学习助手 · 自测练习（支持多篇资料，返回 JSON 文本，前端解析）"""
    yield from learn_stream("请基于这份资料出一份自测选择题", _as_id_list(paper_ids), "quiz", thread_id)


def learn_notes(paper_ids, thread_id: str = "learn") -> Generator[dict, None, None]:
    """学习助手 · 复习笔记（支持多篇资料，Markdown 流式输出）"""
    yield from learn_stream("请基于这份资料生成一份复习笔记", _as_id_list(paper_ids), "notes", thread_id)


def learn_slides(paper_ids, thread_id: str = "learn") -> Generator[dict, None, None]:
    """学习助手 · PPT 生成（支持多篇资料，Marp 格式 Markdown）"""
    yield from learn_stream("请基于这份资料生成一份 Marp 格式的 PPT", _as_id_list(paper_ids), "slides", thread_id)

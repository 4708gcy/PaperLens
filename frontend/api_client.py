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


def rag_chat_stream(message: str, paper_ids: List[int], thread_id: str = "rag") -> Generator[dict, None, None]:
    """综合问答 · 跨文档检索问答（流式）

    与 chat_stream 的区别：带 retrieval_mode="rag"，后端会走 ES 检索
    （从 paper_ids 对应的文档里捞出相关片段）而不是全文直喂。
    适合「在大量文档里找讲过 X 的地方」这类跨文档定位问题。
    """
    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            json={
                "message": message,
                "paper_ids": paper_ids,
                "thread_id": thread_id,
                "retrieval_mode": "rag",
            }
        ) as response:
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        yield data
                    except json.JSONDecodeError:
                        continue


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
    learn_outline: str = "",
    learn_config: Dict[str, Any] = None,
) -> Generator[dict, None, None]:
    """
    学习助手流式对话。

    mode 取值：
    - qa       辅导式问答（chat 流式，逐字显示）
    - summary  一键总结（流式输出 Markdown 摘要）
    - flashcard 知识卡片（流式输出 JSON，前端在 done 后解析）
    - quiz     自测练习（流式输出 JSON，前端在 done 后解析）
    - notes    复习笔记（全文 + 用户大纲）
    - slides   PPT 生成（全文 + 用户大纲 + 主题/页数）

    learn_outline: 用户确认后的大纲（分步向导产物），透传给后端
    learn_config: 其他配置（detail_level / theme / page_count / focus）
    """
    body = {
        "message": message,
        "paper_ids": paper_ids,
        "thread_id": thread_id,
        "learn_mode": mode,
    }
    if learn_outline:
        body["learn_outline"] = learn_outline
    if learn_config:
        body["learn_config"] = learn_config
    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            json=body,
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


# ───────────────────────────────────────────────
# 全文获取（用于复习笔记 / PPT 等「需要通读全文」的任务）
# ───────────────────────────────────────────────
def get_markdown(paper_id: int) -> dict:
    """获取单篇论文的完整 Markdown。返回后端 APIResponse dict。"""
    with httpx.Client(timeout=60) as client:
        response = client.get(f"{BASE_URL}/documents/{paper_id}/markdown")
    return response.json()


def get_full_markdown(paper_ids) -> str:
    """合并多篇论文的完整 Markdown，供笔记/PPT 的 LLM 当全文上下文。

    多篇时用分隔标注区分，单篇直接返回。
    返回拼接后的纯文本。任意一篇读取失败会抛 RuntimeError。
    """
    ids = _as_id_list(paper_ids)
    if not ids:
        raise RuntimeError("未选择任何文档")
    parts = []
    for i, pid in enumerate(ids, 1):
        resp = get_markdown(pid)
        if resp.get("code") and resp["code"] != 200:
            raise RuntimeError(f"读取论文 {pid} 全文失败：{resp.get('msg')}")
        data = resp.get("data") or {}
        title = data.get("title", f"论文{pid}")
        md = data.get("markdown", "")
        if not md:
            raise RuntimeError(f"论文 {pid}（{title}）的全文为空，可能尚未解析完成")
        if len(ids) == 1:
            return md
        parts.append(f"\n\n===== 论文{i}：{title}（paper_id={pid}）=====\n\n{md}")
    return "".join(parts)


# ───────────────────────────────────────────────
# 大纲生成（非流式，用 fast 模型，分步向导第一步）
# ───────────────────────────────────────────────
def generate_outline(paper_ids, mode: str, focus: str = "", **extra) -> str:
    """生成笔记/PPT 的大纲（纯文本）。

    mode: 'notes' 或 'slides'
    focus: 用户填的侧重点
    extra: slides 的 page_count/theme，notes 的 detail_level 等
    返回大纲字符串；失败抛 RuntimeError。
    """
    body = {
        "paper_ids": _as_id_list(paper_ids),
        "mode": mode,
        "focus": focus,
    }
    body.update(extra)
    with httpx.Client(timeout=180) as client:
        response = client.post(f"{BASE_URL}/learn/outline", json=body)
    resp = response.json()
    if resp.get("code") and resp["code"] != 200:
        raise RuntimeError(f"生成大纲失败：{resp.get('msg')}")
    return (resp.get("data") or {}).get("outline", "")


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


def learn_notes(
    paper_ids,
    thread_id: str = "learn",
    outline: str = "",
    detail_level: str = "",
    focus: str = "",
) -> Generator[dict, None, None]:
    """学习助手 · 复习笔记（支持多篇资料，Markdown 流式输出）

    outline: 用户确认后的笔记大纲（分步向导第 2 步产物）。为空则 AI 自行组织。
    detail_level: 精简/标准/详尽
    focus: 本次侧重点
    """
    yield from learn_stream(
        "请基于这份资料生成一份复习笔记",
        _as_id_list(paper_ids),
        "notes",
        thread_id,
        learn_outline=outline,
        learn_config={"detail_level": detail_level, "focus": focus},
    )


def learn_slides(
    paper_ids,
    thread_id: str = "learn",
    outline: str = "",
    theme: str = "default",
    page_count: int = 10,
    focus: str = "",
) -> Generator[dict, None, None]:
    """学习助手 · PPT 生成（支持多篇资料，Marp 格式 Markdown）

    outline: 用户确认后的 PPT 大纲（分步向导第 2 步产物）。为空则 AI 自行组织。
    theme: Marp 主题（default/gaia/uncover）
    page_count: 期望页数
    focus: 本次侧重点
    """
    yield from learn_stream(
        "请基于这份资料生成一份 Marp 格式的 PPT",
        _as_id_list(paper_ids),
        "slides",
        thread_id,
        learn_outline=outline,
        learn_config={"theme": theme, "page_count": page_count, "focus": focus},
    )

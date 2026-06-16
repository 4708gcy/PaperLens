"""Pydantic 请求/响应模型"""
from pydantic import BaseModel
from typing import Optional, Any, List


class APIResponse(BaseModel):
    """统一响应格式"""
    code: int = 200
    msg: str = "success"
    data: Optional[Any] = None


class ChatRequest(BaseModel):
    """对话请求"""
    message: str
    paper_ids: List[int] = []
    thread_id: str = "default"
    # 学习助手模式：留空走论文精读；qa/summary/flashcard/quiz 走学习辅导
    learn_mode: str = ""
    # 学习助手 · 用户确认后的大纲（笔记/PPT 分步向导第 2 步产物）
    learn_outline: str = ""
    # 学习助手 · 其他配置（detail_level / theme / page_count / focus）
    learn_config: dict = {}
    # 综合问答模块用：retrieval_mode="rag" 时走 ES 检索（跨文档），
    # 而非默认的全文直喂。综合问答页传 "rag"，其他页不传（走全文）。
    retrieval_mode: str = ""
    # 用户本次提问附带的多模态图片（data URL 列表，qwen3.7-plus 视觉理解）
    images: List[str] = []


class OutlineRequest(BaseModel):
    """分步向导 · 生成大纲请求（用 fast 模型，非流式）"""
    paper_ids: List[int]
    # notes / slides
    mode: str
    focus: str = ""
    # slides 专用
    page_count: int = 10
    theme: str = "default"
    # notes 专用
    detail_level: str = "标准"


class AnalyzeRequest(BaseModel):
    """结构化解读请求"""
    paper_id: int
    thread_id: str = "analyze"


class SynthesizeRequest(BaseModel):
    """综述请求"""
    paper_ids: List[int]
    topic: str
    thread_id: str = "synthesize"


class ChatResponse(BaseModel):
    """对话响应（带意图）"""
    code: int = 200
    msg: str = "success"
    data: Optional[Any] = None

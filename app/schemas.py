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

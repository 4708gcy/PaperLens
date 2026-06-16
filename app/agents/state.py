"""
LangGraph 状态定义

StateGraph 核心概念：
— 状态是 TypedDict，所有节点共享
— 每个节点返回状态增量（不是替换）
— Annotated + reducer 定义字段的合并策略
"""
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator


class PaperLensState(TypedDict):
    """
    多分支 Agent 系统共享状态

    字段设计原则：刚好够用，不多不少
    """
    # 对话历史（reducer 追加，不覆盖）
    messages: Annotated[list[BaseMessage], operator.add]
    # 意图分类结果：qa / analyze / synthesize / general / learn
    intent: str
    # RAG 检索到的上下文文本
    context: str
    # 目标论文 ID 列表（单论文传 1 个，综述传多个）
    paper_ids: List[int]
    # 学习助手模式："" / "qa" / "summary" / "flashcard" / "quiz"
    # 非空时 triage 直接走 learn 意图，retrieve 后由 learn_agent 处理
    learn_mode: str
    # 学习助手 · 用户确认后的大纲（笔记/PPT 分步向导第 2 步产物）
    learn_outline: str
    # 学习助手 · 其他配置（detail_level / theme / page_count / focus）
    learn_config: dict
    # 综合问答模块用：retrieval_mode="rag" 时 qa 走 ES 检索；空则走全文
    retrieval_mode: str
    # 用户本次提问附带的多模态图片（data URL 列表，供 qwen3.7-plus 视觉理解）
    # 留空则纯文本问答；非空时节点构造多模态 HumanMessage
    images: List[str]
    # 综述主题（仅 synthesize 用）
    topic: str
    # 综述章节（Send 并行 reduce，Phase 3 用）
    sections: Annotated[list, operator.add]
    # 综述大纲（planner 生成）
    outline: list
    # 最终报告
    final_report: str

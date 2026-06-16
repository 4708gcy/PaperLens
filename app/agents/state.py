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
    # 综述主题（仅 synthesize 用）
    topic: str
    # 综述章节（Send 并行 reduce，Phase 3 用）
    sections: Annotated[list, operator.add]
    # 综述大纲（planner 生成）
    outline: list
    # 最终报告
    final_report: str

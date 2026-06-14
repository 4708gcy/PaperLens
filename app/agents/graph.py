"""
LangGraph StateGraph —— 多分支 Agent 编排（Day 2 版本）

图结构（Day 2 阶段）：
  START → triage（意图分类）
              │
         [条件路由]
              │
    ┌─────────┴─────────┐
    ▼                   ▼
  retrieve           general_agent
    │
  [条件路由 2]
    │
    ▼
  qa_agent / analyze_agent
    │
    ▼
   END

关键学习点：
— Annotated[list, operator.add] reducer（messages 追加）
— checkpointer（多轮记忆）
— add_conditional_edges（条件路由）
— Day 4 加 Send 并行综述分支
"""
from typing import Literal
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Send

from app.config import settings
from app.agents.state import PaperLensState
from app.agents.prompts import (
    TRIAGE_SYSTEM, QA_SYSTEM, ANALYZE_SYSTEM, GENERAL_SYSTEM,
    SYNTHESIZE_PLANNER_SYSTEM, SECTION_WRITER_SYSTEM, ASSEMBLER_SYSTEM
)
from app.logger import logger


# ── LLM 实例 ──
def _make_llm(temperature: float = 0.7, model: str = None) -> ChatOpenAI:
    """创建 LLM 实例（OpenAI 兼容协议调 Qwen，含 retry 应对 QPS 限流）"""
    return ChatOpenAI(
        api_key=settings["llm"]["api_key"],
        base_url=settings["llm"]["base_url"],
        model=model or settings["llm"]["model"],
        temperature=temperature,
        max_tokens=settings["llm"]["max_tokens"],
        max_retries=5,  # 自动重试（应对 Send 并发触发 QPS 限流）
        timeout=60,
    )


# 主力模型（用于回答）
llm = _make_llm(temperature=0.7)
# 快速模型（用于意图分类，省钱）
fast_llm = _make_llm(temperature=0, model=settings["llm"]["fast_model"])


# ──────────────────────────────────────────────
# 节点函数
# ──────────────────────────────────────────────

def triage_node(state: PaperLensState) -> dict:
    """
    意图分类节点：只分类，不回答。

    为什么用独立的轻量模型做分类？
    — 分类是简单任务，用 qwen-turbo 省钱
    — 分离后每个 Agent 的 Prompt 更聚焦，表现更好
    """
    last_message = state["messages"][-1]
    paper_ids = state.get("paper_ids", [])
    has_papers = len(paper_ids) > 0

    # 没选论文，强制走 general
    if not has_papers:
        return {"intent": "general"}

    response = fast_llm.invoke([
        SystemMessage(content=TRIAGE_SYSTEM),
        HumanMessage(content=f"用户消息：{last_message.content}")
    ])

    intent = response.content.strip().strip('"').lower()
    valid_intents = ["qa", "analyze", "synthesize", "general"]
    if intent not in valid_intents:
        intent = "general"

    # synthesize 需要 ≥2 篇论文
    if intent == "synthesize" and len(paper_ids) < 2:
        intent = "qa"

    logger.info(f"意图分类: '{str(last_message.content)[:50]}...' → {intent}")
    return {"intent": intent}


def retrieve_node(state: PaperLensState) -> dict:
    """
    RAG 检索节点：获取论文内容上下文。

    为什么检索是独立节点？
    — qa / analyze 都需要检索
    — 抽成独立节点，将来换检索策略只改一处

    analyze 意图特殊处理：用户原话（如"请帮我分析"）太宽泛，
    BM25/向量会偏向表格数字。所以 analyze 固定检索 5 个主题关键词，
    分别召回论文不同部分（背景/方法/贡献/实验/局限）。
    """
    from app.core.rag_engine import rag_engine  # 延迟导入避免循环依赖

    intent = state.get("intent", "qa")
    paper_ids = state.get("paper_ids", [])

    if intent == "analyze":
        # 结构化解读：固定检索 5 个主题
        themes = [
            ("研究背景与动机", "background motivation problem introduction"),
            ("核心方法", "method approach model architecture algorithm"),
            ("主要贡献", "contribution novel we propose"),
            ("实验结果", "experiment results evaluation dataset performance"),
            ("局限与未来工作", "limitation future work conclusion"),
        ]
        all_results = []
        seen = set()
        for label, query in themes:
            results = rag_engine.retrieve(query, paper_ids, top_k=3)
            for r in results:
                key = r.content[:80]
                if key not in seen:
                    seen.add(key)
                    all_results.append((label, r))
        # 拼装
        parts = []
        for i, (label, r) in enumerate(all_results[:15], 1):
            parts.append(f"[资料 {i}]（{label}，论文{r.paper_id} 第{r.source_page}页）\n{r.content}")
        context = "\n\n---\n\n".join(parts) if parts else "（未检索到相关论文内容）"
        logger.info(f"analyze 检索完成: {len(all_results)} 条（5 主题）")
    else:
        # qa：用用户原话检索
        last_message = state["messages"][-1]
        results = rag_engine.retrieve(
            query=last_message.content,
            paper_ids=paper_ids,
            top_k=settings["rag"]["rerank_top_k"]
        )
        if not results:
            context = "（未检索到相关论文内容）"
        else:
            parts = []
            for i, r in enumerate(results, 1):
                parts.append(
                    f"[资料 {i}]（论文{r.paper_id} 第{r.source_page}页，相关性 {r.score:.3f}）\n{r.content}"
                )
            context = "\n\n---\n\n".join(parts)
        logger.info(f"qa 检索完成: {len(results)} 条结果")

    return {"context": context}


def qa_node(state: PaperLensState) -> dict:
    """知识问答节点"""
    last_msg = state["messages"][-1]
    prompt = QA_SYSTEM.format(context=state.get("context", ""))
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=last_msg.content)
    ])
    return {"messages": [response]}


def analyze_node(state: PaperLensState) -> dict:
    """结构化解读节点"""
    last_msg = state["messages"][-1]
    prompt = ANALYZE_SYSTEM.format(context=state.get("context", ""))
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=last_msg.content)
    ])
    return {"messages": [response]}


def general_node(state: PaperLensState) -> dict:
    """一般对话节点"""
    last_msg = state["messages"][-1]
    response = llm.invoke([
        SystemMessage(content=GENERAL_SYSTEM),
        HumanMessage(content=last_msg.content)
    ])
    return {"messages": [response]}


# ──────────────────────────────────────────────
# 路由函数
# ──────────────────────────────────────────────

def route_by_intent(state: PaperLensState) -> Literal["retrieve", "general_agent"]:
    """
    根据 triage 结果路由。
    qa / analyze → retrieve（先检索）
    general → general_agent（无需检索）
    """
    intent = state.get("intent", "general")
    if intent in ("qa", "analyze"):
        return "retrieve"
    return "general_agent"


def route_after_retrieve(state: PaperLensState) -> Literal["qa_agent", "analyze_agent"]:
    """检索完成后，按原始意图路由"""
    intent = state.get("intent", "qa")
    return "analyze_agent" if intent == "analyze" else "qa_agent"


# ──────────────────────────────────────────────
# 综述 Agent 节点（Day 4：Send 并行 + assembler）
# ──────────────────────────────────────────────

def _extract_topic_from_message(state: PaperLensState) -> str:
    """从用户消息中提取综述主题"""
    last_msg = state["messages"][-1]
    content = str(last_msg.content)
    # 简单提取：找「」或""内的内容，或去掉前缀话术
    for sep in ["「", "」", """, """, "关于", "的综述", "写一篇"]:
        content = content.replace(sep, "")
    # 去掉常见的请求前缀
    for prefix in ["请基于以下论文", "请帮我", "生成", "写"]:
        if content.startswith(prefix):
            content = content[len(prefix):]
    return content.strip()[:50] if content.strip() else "论文综述"


def synthesize_planner_node(state: PaperLensState) -> dict:
    """
    综述大纲规划：LLM 生成 4-6 章节大纲（JSON）

    输出格式：[{"section_title":"...","retrieval_queries":[...]}]
    """
    paper_ids = state.get("paper_ids", [])
    topic = state.get("topic") or _extract_topic_from_message(state)

    paper_list = ", ".join([f"论文{pid}" for pid in paper_ids])
    prompt = SYNTHESIZE_PLANNER_SYSTEM.format(topic=topic, paper_list=paper_list)

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="请生成综述大纲")
    ])

    # 解析 JSON（容错 markdown 代码块包裹）
    try:
        text = response.content.strip()
        # 去除 ```json ... ``` 包裹
        if "```" in text:
            # 提取 ``` 之间的内容
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    text = part
                    break
        outline = json.loads(text)
        if not isinstance(outline, list):
            outline = []
    except json.JSONDecodeError as e:
        logger.error(f"大纲 JSON 解析失败: {e}\n原始: {response.content[:200]}")
        outline = [
            {"section_title": "研究背景", "retrieval_queries": [topic, "background motivation"]},
            {"section_title": "主要方法", "retrieval_queries": [topic, "method approach"]},
        ]

    logger.info(f"综述大纲生成: {len(outline)} 章 - {[s.get('section_title','?') for s in outline]}")
    return {"outline": outline, "topic": topic}


def section_worker_node(state: dict) -> dict:
    """
    单章节写作节点（被 Send 并行调用）

    注意：接收的 state 是 Send 传入的子状态，不是 PaperLensState
    返回 {"sections": [chapter_text]}，由 reducer 汇聚

    含异常捕获：单章节失败不影响整体综述（降级为占位说明）
    """
    section_title = state["section_title"]
    retrieval_queries = state.get("retrieval_queries", [section_title])
    paper_ids = state["paper_ids"]

    try:
        # 对每个 query 检索，合并结果
        from app.core.rag_engine import rag_engine
        all_context = []
        seen = set()
        for query in retrieval_queries:
            results = rag_engine.retrieve(query, paper_ids, top_k=3)
            for r in results:
                key = r.content[:80]
                if key not in seen:
                    seen.add(key)
                    all_context.append(
                        f"[论文{r.paper_id}] {r.content}"
                    )

        context = "\n\n---\n\n".join(all_context[:5]) or "（未检索到相关内容）"

        prompt = SECTION_WRITER_SYSTEM.format(
            section_title=section_title, context=context
        )

        # 带 retry 的 LLM 调用（应对并发限流）
        import time
        last_err = None
        for attempt in range(4):
            try:
                response = llm.invoke([
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"请撰写章节：{section_title}")
                ])
                logger.info(f"章节完成: {section_title}")
                return {"sections": [f"## {section_title}\n\n{response.content}"]}
            except Exception as e:
                last_err = e
                logger.warning(f"章节 {section_title} 第 {attempt+1} 次失败: {e}")
                time.sleep(2 * (attempt + 1))  # 指数退避

        # 全部 retry 失败，降级
        logger.error(f"章节 {section_title} 全部重试失败: {last_err}")
        return {"sections": [f"## {section_title}\n\n（本章节因服务限流暂未生成，请稍后重试）"]}

    except Exception as e:
        logger.error(f"章节 {section_title} 异常: {e}", exc_info=True)
        return {"sections": [f"## {section_title}\n\n（本章节生成异常：{str(e)[:100]}）"]}


def assembler_node(state: PaperLensState) -> dict:
    """合并所有章节为完整综述报告"""
    sections = state.get("sections", [])
    topic = state.get("topic", "")
    paper_ids = state.get("paper_ids", [])

    body = "\n\n".join(sections)
    references = "\n".join([f"- 论文{pid}" for pid in paper_ids])

    prompt = ASSEMBLER_SYSTEM.format(topic=topic)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"各章节内容：\n\n{body}\n\n参考文献论文：{references}")
    ])

    return {"messages": [response], "final_report": response.content}


def route_after_triage(state: PaperLensState):
    """三路路由：triage 后根据意图分发"""
    intent = state.get("intent", "general")
    if intent == "synthesize":
        return "planner"
    elif intent in ("qa", "analyze"):
        return "retrieve"
    return "general_agent"


def route_synthesize(state: PaperLensState):
    """
    路由函数：从 planner 出来后，Send 并行启动所有 section_worker

    Send API 核心：返回 [Send(node_name, sub_state), ...]
    LangGraph 会为每个 Send 启动一个并行节点执行
    """
    outline = state.get("outline", [])
    paper_ids = state.get("paper_ids", [])

    if not outline:
        return "assembler"

    sends = [
        Send("section_worker", {
            "section_title": section["section_title"],
            "retrieval_queries": section.get("retrieval_queries", []),
            "paper_ids": paper_ids
        })
        for section in outline
    ]
    logger.info(f"派发 {len(sends)} 个并行综述章节")
    return sends


# ──────────────────────────────────────────────
# 构建图
# ──────────────────────────────────────────────

def build_graph():
    """构建 PaperLens StateGraph（完整版：qa/analyze/general/synthesize）"""
    builder = StateGraph(PaperLensState)

    # 添加节点
    builder.add_node("triage", triage_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("qa_agent", qa_node)
    builder.add_node("analyze_agent", analyze_node)
    builder.add_node("general_agent", general_node)
    # 综述节点
    builder.add_node("planner", synthesize_planner_node)
    builder.add_node("section_worker", section_worker_node)
    builder.add_node("assembler", assembler_node)

    # 入口
    builder.add_edge(START, "triage")

    # 条件边 1：triage → retrieve / planner / general_agent（三路）
    builder.add_conditional_edges(
        "triage", route_after_triage,
        {"retrieve": "retrieve", "planner": "planner", "general_agent": "general_agent"}
    )

    # 条件边 2：retrieve → qa_agent / analyze_agent
    builder.add_conditional_edges(
        "retrieve", route_after_retrieve,
        {"qa_agent": "qa_agent", "analyze_agent": "analyze_agent"}
    )

    # ★ 综述：planner → Send 并行 section_worker（或兜底 assembler）
    builder.add_conditional_edges(
        "planner", route_synthesize, ["section_worker", "assembler"]
    )

    # ★ 所有 section_worker 完成后汇聚到 assembler
    builder.add_edge("section_worker", "assembler")

    # 终止
    builder.add_edge("qa_agent", END)
    builder.add_edge("analyze_agent", END)
    builder.add_edge("general_agent", END)
    builder.add_edge("assembler", END)

    # 编译（含 checkpointer）
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# 全局图实例
study_graph = build_graph()

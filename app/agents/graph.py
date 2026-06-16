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
    TRIAGE_SYSTEM, QA_SYSTEM, RAG_QA_SYSTEM, ANALYZE_SYSTEM, GENERAL_SYSTEM,
    SYNTHESIZE_PLANNER_SYSTEM, SECTION_WRITER_SYSTEM, ASSEMBLER_SYSTEM,
    LEARN_QA_SYSTEM, LEARN_SUMMARY_SYSTEM, LEARN_FLASHCARD_SYSTEM, LEARN_QUIZ_SYSTEM,
    LEARN_NOTES_SYSTEM, LEARN_SLIDES_SYSTEM,
    LEARN_NOTES_OUTLINE_SYSTEM, LEARN_SLIDES_OUTLINE_SYSTEM,
)
from app.logger import logger


def _build_human_message(text: str, images: list) -> HumanMessage:
    """构造 HumanMessage：有图片时用多模态（qwen3.7-plus 视觉），无图片则纯文本。

    images 是 data URL 字符串列表（如 "data:image/jpeg;base64,..."）。
    """
    images = [img for img in (images or []) if img]
    if not images:
        return HumanMessage(content=text)
    content = [{"type": "text", "text": text}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": img}})
    return HumanMessage(content=content)


# ── LLM 实例 ──
def _make_llm(temperature: float = 0.7, model: str = None) -> ChatOpenAI:
    """创建 LLM 实例（OpenAI 兼容协议调 Qwen，含 retry 应对 QPS 限流）

    qwen3.7-plus 是「混合思考模型」：
    - enable_thinking=true：先输出 reasoning_content（content 为空），再输出正式回答。
      质量更高，但流式首 token 要等 ~45s（思考静默期）。
    - enable_thinking=false：立即以 content 流式输出，首 token ~1s，但回答不经过深思。
    开关由 config.yaml 的 llm.enable_thinking 控制，默认 true（重质量）。
    见 https://www.alibabacloud.com/help/en/model-studio/deep-thinking
    """
    enable_thinking = bool(settings["llm"].get("enable_thinking", True))
    return ChatOpenAI(
        api_key=settings["llm"]["api_key"],
        base_url=settings["llm"]["base_url"],
        model=model or settings["llm"]["model"],
        temperature=temperature,
        max_tokens=settings["llm"]["max_tokens"],
        max_retries=5,  # 自动重试（应对 Send 并发触发 QPS 限流）
        timeout=60,
        # 思考模式开关：DashScope OpenAI 兼容接口用 extra_body 传 enable_thinking
        # （官方建议用 extra_body 而非 model_kwargs 传非标准参数）
        extra_body={"enable_thinking": enable_thinking},
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

    分类策略（规则优先，LLM 兜底）：
    0. 学习模式（learn_mode 非空）→ learn（学习助手页直接指定模式）
    1. 没选论文 → general（闲聊）
    2. 选了论文 + 明确闲聊词（你好/谢谢/你是谁） → general
    3. 选了论文 + 含"分析/解读/综述"关键词 → analyze/synthesize
    4. 选了论文 + 含"根据已上传/这篇文章/列出内容"等指向已选文档的词 → qa（强制走 RAG）
    5. 选了论文 + 其他 → qa（默认走 RAG，不要让 LLM 把通用概念判成闲聊）

    为什么不全用 LLM 分类？
    — 实测"什么是 git"这种通用概念问题会被 LLM 误判为"闲聊"，
      导致选了论文却不用 RAG，直接用 LLM 通用知识回答。
    — 规则保证：只要选了论文，除了纯打招呼，一律走 RAG 检索。
    """
    # 0. 学习模式优先（学习助手页直接指定 mode，无需分类）
    if state.get("learn_mode"):
        return {"intent": "learn"}

    last_message = state["messages"][-1]
    paper_ids = state.get("paper_ids", [])
    has_papers = len(paper_ids) > 0
    msg_text = str(last_message.content).strip().lower()

    # 1. 没选论文，强制走 general
    if not has_papers:
        return {"intent": "general"}

    # 2. 纯闲聊词（选了论文也要允许打招呼）
    greetings = {"你好", "hello", "hi", "谢谢", "感谢", "thanks", "thank you",
                 "你是谁", "你能做什么", "帮助", "再见", "bye", "在吗"}
    # 去掉标点后判断
    msg_clean = msg_text.replace("？", "").replace("?", "").replace("！", "").replace("!", "").replace("。", "").replace(".", "").replace(",", "").replace("，", "").strip()
    if msg_clean in greetings or len(msg_clean) <= 2:
        return {"intent": "general"}

    # 3. 关键词识别 analyze / synthesize
    if any(k in msg_text for k in ["综述", "对比", "比较这几篇", "跨论文"]):
        if len(paper_ids) >= 2:
            return {"intent": "synthesize"}
        return {"intent": "qa"}

    if any(k in msg_text for k in ["分析", "解读", "结构化", "帮我看", "总结这篇", "梳理"]):
        return {"intent": "analyze"}

    # 4. 明确指向"已选文档"的提问 → 强制走 qa（避免被判成 general 说"无法访问文件"）
    doc_hint_words = ["已上传", "已传", "上传的", "这篇文章", "这篇文档", "这篇论文", "这篇",
                      "根据文章", "根据文档", "根据论文", "刚传", "列出", "列出来", "全部内容", "所有内容"]
    if any(k in msg_text for k in doc_hint_words):
        logger.info(f"意图分类（指向已选文档）→ qa: '{str(last_message.content)[:50]}...'")
        return {"intent": "qa"}

    # 5. 其他全部走 qa（默认用 RAG 检索论文，不闲聊）
    logger.info(f"意图分类（规则）: '{str(last_message.content)[:50]}...' → qa")
    return {"intent": "qa"}


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
        # 查文档标题，context 里用「论文{id}: 标题」而不是纯数字 ID
        from app.models.orm import Database, Paper
        title_map = {}
        try:
            db = Database.get_session()
            try:
                for p in db.query(Paper).filter(Paper.paper_id.in_(paper_ids)).all():
                    title_map[p.paper_id] = p.title
            finally:
                db.close()
        except Exception:
            pass

        # 在 context 顶部注入「语料范围内所有文档清单」，这样「列出文档」类问题能看到全部
        doc_list_lines = [f"- 论文{pid}：{title_map.get(pid, '(未知标题)')}" for pid in paper_ids]
        doc_list_block = "【语料范围内所有文档】\n" + "\n".join(doc_list_lines) + "\n"

        if not results:
            context = doc_list_block + "（未检索到与问题相关的片段）"
        else:
            parts = [doc_list_block]
            for i, r in enumerate(results, 1):
                _t = title_map.get(r.paper_id, "")
                _label = f"论文{r.paper_id}（{_t}）" if _t else f"论文{r.paper_id}"
                parts.append(
                    f"[资料 {i}]（{_label} 第{r.source_page}页，相关性 {r.score:.3f}）\n{r.content}"
                )
            context = "\n\n---\n\n".join(parts)
        logger.info(f"qa 检索完成: {len(results)} 条结果")

    return {"context": context}


def qa_node(state: PaperLensState) -> dict:
    """
    知识问答节点

    素材来源：
    - retrieval_mode="rag"（综合问答模块）：用 retrieve 检索到的 ES 片段（跨文档定位）
    - 默认（论文精读）：读全文 markdown（1M 上下文塞得下）

    检索为空时的兜底：在 system 里明确标注"未检索到相关内容"，
    让模型诚实回应（区分"文档没讲"和"通用理解"）。
    """
    last_msg = state["messages"][-1]
    retrieval_mode = state.get("retrieval_mode", "")
    images = state.get("images", [])

    if retrieval_mode == "rag":
        # 综合问答：用 ES 检索片段
        context = state.get("context", "")
        source = "检索片段（ES 跨文档）"
        if not context or not context.strip():
            context = '（注意：未检索到相关内容。请如实说明所选语料里没有直接讲这个问题，并可补充明确标注为「通用理解（仅供参考）」的内容。）'
            source = "检索为空"
        prompt = RAG_QA_SYSTEM.format(context=context)
    else:
        # 论文精读：全文
        paper_ids = state.get("paper_ids", []) or []
        try:
            context = _load_full_markdown(paper_ids)
            source = "全文"
        except Exception as e:
            logger.warning(f"qa_node 全文读取失败，降级检索片段: {e}")
            context = state.get("context", "")
            source = "检索片段（全文读取失败降级）"
        prompt = QA_SYSTEM.format(context=context)

    response = llm.invoke([
        SystemMessage(content=prompt),
        _build_human_message(last_msg.content, images)
    ])
    logger.info(f"qa 完成（素材={source}）")
    return {"messages": [response]}


def analyze_node(state: PaperLensState) -> dict:
    """结构化解读节点（优先全文，1篇全文塞得下）"""
    last_msg = state["messages"][-1]
    paper_ids = state.get("paper_ids", []) or []
    try:
        context = _load_full_markdown(paper_ids)
        source = "全文"
    except Exception as e:
        logger.warning(f"analyze_node 全文读取失败，降级检索片段: {e}")
        context = state.get("context", "")
        source = "检索片段"
    prompt = ANALYZE_SYSTEM.format(context=context)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=last_msg.content)
    ])
    logger.info(f"analyze 完成（素材={source}）")
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
# 学习助手节点（复用 retrieve 检索到的课件片段，按 mode 选 prompt）
# ──────────────────────────────────────────────

def learn_node(state: PaperLensState) -> dict:
    """
    学习助手节点：根据 learn_mode 选对应 prompt。

    模式：
    - qa: 苏格拉底式辅导问答
    - summary: 整篇课件结构化大纲摘要
    - flashcard: 知识卡片（JSON）
    - quiz: 自测选择题（JSON）
    - notes: 复习笔记（读全文 Markdown + 用户大纲，Markdown 输出）
    - slides: PPT 生成（读全文 Markdown + 用户大纲 + 主题/页数，Marp 格式）

    素材来源：
    - 片段型（qa/summary/flashcard/quiz）：用 retrieve_node 检索到的 state.context（top-5 ES 片段）
    - 全文型（notes/slides）：改读磁盘上的完整 Markdown —— 这类任务要"通读全文"，
      只看 top-5 片段会只见树木不见森林。

    flashcard/quiz 不在此节点解析 JSON —— 保持原始文本流式推送，
    由前端在 'done' 后解析（解析容错：去掉 ```json 包裹）。
    """
    mode = state.get("learn_mode", "qa") or "qa"
    last_msg = state["messages"][-1]
    retrieved_context = state.get("context", "")
    images = state.get("images", [])
    config = state.get("learn_config", {}) or {}
    user_outline = state.get("learn_outline", "") or ""
    paper_ids = state.get("paper_ids", []) or []

    # ── 学习助手一律走全文（用户要求：所有模式优先通读全文再回答）──
    # 全文优先：context 用磁盘上的完整 Markdown（含图表描述），LLM 能看全。
    # 检索片段(retrieved_context)仅作文案/日志参考，不再喂给 LLM。
    # 若全文读取失败（如文档未解析完），降级用检索片段，保证不报错。
    try:
        context = _load_full_markdown(paper_ids)
        source_note = "完整资料（MinerU 解析的全文 Markdown，含图表说明）"
    except Exception as e:
        logger.warning(f"全文读取失败，降级用检索片段（mode={mode}）: {e}")
        context = retrieved_context
        source_note = "检索到的课件片段（全文读取失败，降级）"

    # ── 选 prompt 模板 ──
    prompt_map = {
        "qa": LEARN_QA_SYSTEM,
        "summary": LEARN_SUMMARY_SYSTEM,
        "flashcard": LEARN_FLASHCARD_SYSTEM,
        "quiz": LEARN_QUIZ_SYSTEM,
        "notes": LEARN_NOTES_SYSTEM,
        "slides": LEARN_SLIDES_SYSTEM,
    }
    template = prompt_map.get(mode, LEARN_QA_SYSTEM)

    # ── 填充模板占位符 ──
    # 片段型只有 {context}；全文型还有 {outline}/{focus}/{detail_level}/{theme}/{page_count}
    placeholders = {"context": context}
    if mode == "notes":
        placeholders["outline"] = user_outline if user_outline.strip() else "（用户未提供大纲，请自行组织结构）"
        placeholders["focus"] = config.get("focus", "") or ""
        placeholders["detail_level"] = config.get("detail_level", "") or "标准"
    elif mode == "slides":
        placeholders["outline"] = user_outline if user_outline.strip() else "（用户未提供大纲，请自行规划页序）"
        placeholders["theme"] = config.get("theme", "") or "default"
        placeholders["page_count"] = str(config.get("page_count", "") or 10)
        placeholders["focus"] = config.get("focus", "") or ""

    try:
        prompt = template.format(**placeholders)
    except KeyError as e:
        # 模板里有未提供的占位符，兜底用 context
        logger.warning(f"learn_node prompt 缺占位符 {e}，回退只填 context")
        prompt = template.format(context=context)

    # summary/flashcard/quiz/notes/slides 是"一键生成"类任务：用户消息作为任务指令补充
    # qa 模式则是正常问答，且支持用户附带图片（多模态）
    if mode == "qa":
        user_content = last_msg.content
        human_msg = _build_human_message(user_content, images)
    else:
        user_content = f"请基于上面的【{source_note}】完成此任务。学生原始输入：{last_msg.content}"
        human_msg = HumanMessage(content=user_content)

    response = llm.invoke([
        SystemMessage(content=prompt),
        human_msg
    ])
    logger.info(f"学习助手完成（mode={mode}, 素材={source_note}）")
    return {"messages": [response]}


def _load_full_markdown(paper_ids: list, max_chars: int = 600000) -> str:
    """把多篇论文的完整 Markdown 拼成一段全文，供「通读全文」任务使用。

    qwen3.7-plus 上下文 1M tokens（≈中文 180 万字）。这里留出 prompt + 输出空间，
    默认 max_chars=60万字（≈30万tokens），单篇/少数几篇完全够用。

    超限处理（一次性选几十篇课件的极端情况）：
    - 不直接报错，而是【按篇均摊截断】：每篇只保留开头 + 结尾各一半，保证每篇都能被看到。
    - 截断时打明确标记，让 LLM 知道这是节选。

    单篇直接返回；多篇用分隔标注区分。任何一篇读取失败会跳过并记日志。
    """
    from app.services.document_service import document_service
    docs = []  # [(pid, md)]
    for pid in paper_ids:
        try:
            docs.append((pid, document_service.get_full_markdown(pid)))
        except Exception as e:
            logger.warning(f"读取论文 {pid} 全文失败，跳过该篇: {e}")
    if not docs:
        return "（未能读取到任何全文内容，请检查文档是否已解析完成）"

    total = sum(len(md) for _, md in docs)
    if total <= max_chars:
        # 没超限，全量拼接
        if len(docs) == 1:
            return docs[0][1]
        parts = []
        for i, (pid, md) in enumerate(docs, 1):
            parts.append(f"\n\n===== 资料{i}（paper_id={pid}）=====\n\n{md}")
        return "".join(parts)

    # 超限：按篇均摊预算，每篇保留开头+结尾
    logger.warning(
        f"全文合计 {total} 字超限（max={max_chars}），{len(docs)} 篇将做均摊截断"
    )
    per_doc_budget = max(max_chars // len(docs), 2000)  # 每篇至少留 2000 字
    parts = []
    for i, (pid, md) in enumerate(docs, 1):
        if len(md) <= per_doc_budget:
            chunk = md
        else:
            half = per_doc_budget // 2
            chunk = (
                md[:half]
                + f"\n\n……（本篇已截断，原文 {len(md)} 字，此处仅保留开头与结尾）……\n\n"
                + md[-half:]
            )
        parts.append(f"\n\n===== 资料{i}（paper_id={pid}，{len(md)}字）=====\n\n{chunk}")
    header = f"（注意：你选择的 {len(docs)} 篇资料合计超长，以下为每篇节选拼接，详见各篇标记。）\n"
    return header + "".join(parts)


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


def route_after_retrieve(state: PaperLensState) -> Literal["qa_agent", "analyze_agent", "learn_agent"]:
    """检索完成后，按原始意图路由（learn 走学习助手节点）"""
    intent = state.get("intent", "qa")
    if intent == "analyze":
        return "analyze_agent"
    if intent == "learn":
        return "learn_agent"
    return "qa_agent"


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
    # 参考文献带上标题，避免 LLM 只输出无意义的"论文1/论文2"
    from app.models.orm import Database, Paper
    ref_lines = []
    try:
        db = Database.get_session()
        try:
            for pid in paper_ids:
                p = db.query(Paper).filter(Paper.paper_id == pid).first()
                title = p.title if p else "(未知标题)"
                ref_lines.append(f"- 论文{pid}：{title}")
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"读取论文标题失败，参考文献回退纯编号: {e}")
        ref_lines = [f"- 论文{pid}" for pid in paper_ids]
    references = "\n".join(ref_lines)

    prompt = ASSEMBLER_SYSTEM.format(topic=topic)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"各章节内容：\n\n{body}\n\n参考文献（含论文编号与标题，请在报告末尾原样列出，不要省略标题）：\n{references}")
    ])

    return {"messages": [response], "final_report": response.content}


def route_after_triage(state: PaperLensState):
    """路由：triage 后根据意图分发

    - synthesize → planner（综述规划，仍走检索）
    - qa → qa_agent、analyze → analyze_agent（直达，跳过检索；走全文 markdown）
    - learn → learn_agent（直达，跳过检索；学习助手走全文 markdown）
    - 其他 → general_agent（闲聊）

    为什么 qa/analyze/learn 默认跳过 retrieve？
    — qwen3.7-plus 上下文 1M，单篇/少数几篇全文塞得下，全文比 top-5 片段质量更高。
    — retrieve_node 现在主要被 synthesize 的 section_worker 间接复用。

    例外：retrieval_mode="rag"（综合问答模块）时，qa 走 retrieve（ES 跨文档检索），
    因为综合问答场景是「跨大量文档定位相关片段」，全文塞不下也不该塞。
    """
    intent = state.get("intent", "general")
    retrieval_mode = state.get("retrieval_mode", "")
    if intent == "synthesize":
        return "planner"
    elif intent == "learn":
        return "learn_agent"
    elif intent == "qa":
        # 综合问答模块显式要求走 ES 检索（跨文档）；否则走全文直喂
        return "retrieve" if retrieval_mode == "rag" else "qa_agent"
    elif intent == "analyze":
        return "analyze_agent"
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
    builder.add_node("learn_agent", learn_node)
    builder.add_node("general_agent", general_node)
    # 综述节点
    builder.add_node("planner", synthesize_planner_node)
    builder.add_node("section_worker", section_worker_node)
    builder.add_node("assembler", assembler_node)

    # 入口
    builder.add_edge(START, "triage")

    # 条件边 1：triage → planner / qa_agent / analyze_agent / learn_agent / general_agent
    # qa/analyze/learn 全部直达对应 agent（跳过 retrieve，走全文）。
    # synthesize 仍走 planner。retrieve 节点仅被综述的 section_worker 间接复用。
    builder.add_conditional_edges(
        "triage", route_after_triage,
        {
            "retrieve": "retrieve",
            "planner": "planner",
            "qa_agent": "qa_agent",
            "analyze_agent": "analyze_agent",
            "learn_agent": "learn_agent",
            "general_agent": "general_agent",
        }
    )

    # 条件边 2：retrieve → qa_agent / analyze_agent / learn_agent
    builder.add_conditional_edges(
        "retrieve", route_after_retrieve,
        {"qa_agent": "qa_agent", "analyze_agent": "analyze_agent", "learn_agent": "learn_agent"}
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
    builder.add_edge("learn_agent", END)
    builder.add_edge("general_agent", END)
    builder.add_edge("assembler", END)

    # 编译（含 checkpointer）
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# 全局图实例
study_graph = build_graph()

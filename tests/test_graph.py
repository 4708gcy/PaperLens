"""Agent 路由逻辑测试（不依赖 LLM）"""
from app.agents.graph import (
    route_after_triage, route_after_retrieve, _build_chat_messages,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ── 多轮记忆修复的回归测试（不调 LLM，只验消息序列构造）──
def test_build_chat_messages_includes_history():
    """核心 bug 回归：节点必须把历史 Human/AIMessage 拼进消息序列，
    否则模型只看到当前一句，回答"第一次交流"。"""
    history = [
        HumanMessage(content="我叫张三"),
        AIMessage(content="你好张三"),
        HumanMessage(content="我叫什么？"),
    ]
    state = {"messages": history, "images": []}
    msgs = _build_chat_messages("你是助手", state)
    # 期望：[System, 历史 Human, 历史 AI, 当前 Human] = 4 条
    assert len(msgs) == 4
    assert isinstance(msgs[0], SystemMessage)
    assert "张三" in msgs[1].content              # 历史 user 在
    assert "张三" in msgs[2].content              # 历史 ai 在
    assert "我叫什么" in msgs[3].content          # 当前 user 在


def test_build_chat_messages_respects_window():
    """历史超过 window 时只取最近 N 轮，避免上下文爆炸"""
    history = [HumanMessage(content=f"u{i}") for i in range(50)] + \
              [HumanMessage(content="当前")]
    state = {"messages": history, "images": []}
    msgs = _build_chat_messages("sys", state, history_window=3)
    # system + 最近 6 条(3 轮×2) + 当前 1 = 8
    assert len(msgs) == 1 + 3 * 2 + 1
    # 最早的消息被裁掉
    contents = [m.content for m in msgs]
    assert "u0" not in contents


def test_build_chat_messages_empty_state():
    """空状态不崩"""
    msgs = _build_chat_messages("sys", {})
    assert len(msgs) >= 2   # 至少 system + 一条空 human


# ── 章节感知分批的回归测试（不调 LLM，只验切分/定位逻辑）──
from app.agents.graph import _split_by_chapters, _normalize_title, _looks_like_single_chapter


def test_split_by_chapters():
    """按 # / ## 标题切分，### 不切顶层"""
    md = """前言内容

# 第1章 绪论
绪论正文

## 1.1 背景
背景内容

# 第2章 方法
方法正文
"""
    chapters = _split_by_chapters(md)
    titles = [t for t, _ in chapters]
    # 前言 + 第1章 + 第2章（###/## 1.1 不算顶层边界）
    assert len(chapters) >= 3
    assert any("绪论" in t for t in titles)
    assert any("方法" in t for t in titles)


def test_normalize_title():
    """去章节序号/页码"""
    assert _normalize_title("第3章 最小二乘学习法 22") == "最小二乘学习法"
    assert _normalize_title("1. 什么是机器学习") == "什么是机器学习"
    assert _normalize_title("第Ⅱ部分 有监督回归") == "有监督回归"


def test_looks_like_single_chapter():
    """单章标题 vs 完整大纲"""
    assert _looks_like_single_chapter("最小二乘学习法") is True
    assert _looks_like_single_chapter("第3章 最小二乘学习法") is True
    # 多行大纲不是单章
    multi = "1. 绪论\n2. 学习模型\n3. 最小二乘\n4. 鲁棒学习"
    assert _looks_like_single_chapter(multi) is False




def test_route_synthesize():
    assert route_after_triage({"intent": "synthesize"}) == "planner"


def test_route_qa():
    """论文精读问答：直达 qa_agent（跳过 ES 检索，走全文 markdown）"""
    assert route_after_triage({"intent": "qa"}) == "qa_agent"


def test_route_analyze():
    """结构化解读：直达 analyze_agent（跳过 ES 检索，走全文 markdown）"""
    assert route_after_triage({"intent": "analyze"}) == "analyze_agent"


def test_route_general():
    assert route_after_triage({"intent": "general"}) == "general_agent"


def test_route_learn():
    """学习助手意图：triage 后直达 learn_agent（跳过 ES 检索，走全文 markdown）"""
    assert route_after_triage({"intent": "learn"}) == "learn_agent"


def test_route_unknown():
    assert route_after_triage({"intent": "unknown"}) == "general_agent"


def test_route_none():
    assert route_after_triage({}) == "general_agent"


def test_route_qa_rag():
    """综合问答模块（retrieval_mode=rag）：qa 走 ES 检索，而非全文直喂"""
    assert route_after_triage({"intent": "qa", "retrieval_mode": "rag"}) == "retrieve"


def test_route_qa_default_fulltext():
    """默认（无 retrieval_mode）：qa 走全文，直达 qa_agent"""
    assert route_after_triage({"intent": "qa", "retrieval_mode": ""}) == "qa_agent"


def test_route_after_retrieve_qa():
    assert route_after_retrieve({"intent": "qa"}) == "qa_agent"


def test_route_after_retrieve_analyze():
    assert route_after_retrieve({"intent": "analyze"}) == "analyze_agent"


def test_route_after_retrieve_learn():
    """学习助手意图：检索后路由到 learn_agent"""
    assert route_after_retrieve({"intent": "learn"}) == "learn_agent"


def test_route_after_retrieve_default():
    """未识别的意图默认走 qa_agent"""
    assert route_after_retrieve({"intent": "synthesize"}) == "qa_agent"  # 默认

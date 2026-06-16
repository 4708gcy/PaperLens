"""Agent 路由逻辑测试（不依赖 LLM）"""
from app.agents.graph import route_after_triage, route_after_retrieve


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

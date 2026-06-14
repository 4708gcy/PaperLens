"""Agent 路由逻辑测试（不依赖 LLM）"""
from app.agents.graph import route_after_triage, route_after_retrieve


def test_route_synthesize():
    assert route_after_triage({"intent": "synthesize"}) == "planner"


def test_route_qa():
    assert route_after_triage({"intent": "qa"}) == "retrieve"


def test_route_analyze():
    assert route_after_triage({"intent": "analyze"}) == "retrieve"


def test_route_general():
    assert route_after_triage({"intent": "general"}) == "general_agent"


def test_route_unknown():
    assert route_after_triage({"intent": "unknown"}) == "general_agent"


def test_route_none():
    assert route_after_triage({}) == "general_agent"


def test_route_after_retrieve_qa():
    assert route_after_retrieve({"intent": "qa"}) == "qa_agent"


def test_route_after_retrieve_analyze():
    assert route_after_retrieve({"intent": "analyze"}) == "analyze_agent"


def test_route_after_retrieve_default():
    assert route_after_retrieve({"intent": "synthesize"}) == "qa_agent"  # 默认

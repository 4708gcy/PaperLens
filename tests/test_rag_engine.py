"""RAG 引擎测试（需 ES 运行）"""
import pytest
from app.core.rag_engine import rag_engine


@pytest.fixture
def test_paper_id():
    """测试用 paper_id"""
    return 99999


def test_es_connection():
    """ES 连接正常"""
    info = rag_engine.es.info()
    assert "version" in info
    assert "number" in info["version"]


def test_index_name():
    """索引名生成正确"""
    assert rag_engine._index_name(1) == "paperlens_paper_1"
    assert rag_engine._index_name(999) == "paperlens_paper_999"


def test_rrf_fusion_basic():
    """RRF 融合：两路都有的文档应排前"""
    bm25 = [
        {"_id": "1", "_index": "test", "_source": {"chunk_content": "doc1", "chunk_index": 1}},
        {"_id": "2", "_index": "test", "_source": {"chunk_content": "doc2", "chunk_index": 2}},
    ]
    vector = [
        {"_id": "2", "_index": "test", "_source": {"chunk_content": "doc2", "chunk_index": 2}},
        {"_id": "3", "_index": "test", "_source": {"chunk_content": "doc3", "chunk_index": 3}},
    ]
    fused = rag_engine._rrf_fusion(bm25, vector)
    # doc2 在两路都出现，得分最高
    assert fused[0]["_id"] == "2"
    assert len(fused) == 3


def test_rrf_fusion_empty():
    """RRF 空输入"""
    assert rag_engine._rrf_fusion([], []) == []
    assert rag_engine._rrf_fusion([{"_id": "1", "_index": "t", "_source": {}}], []) != []


def test_delete_nonexistent_index():
    """删除不存在的索引不报错"""
    rag_engine.delete_index(99999)  # 应静默成功

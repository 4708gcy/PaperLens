"""
RAG 检索引擎：BM25 + 向量检索 → RRF 融合 → Cross-Encoder 重排序

检索流程：
  用户查询
    ├─→ BM25 关键词检索（ES match query）──→ top_k
    ├─→ BGE-m3 向量检索（ES KNN）────→ top_k
    │
    └─→ RRF 融合 ──→ 统一排序 ──→ top-20 候选
                                      │
                                      ▼
                              BGE-reranker 精排
                                      │
                                      ▼
                                最终 top-5 结果

RRF（Reciprocal Rank Fusion）公式：
  RRF_score(doc) = Σ 1 / (rank_in_list + k)
  其中 k=60 是平滑常数（Cormack et al., 2009 推荐值，ES 官方默认）

为什么用 RRF 而不是直接分数加权？
— BM25 分数和余弦相似度的尺度不同，直接加权需要调参且不稳定
— RRF 只使用排名，天然具备尺度不变性
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from app.config import settings
from app.core.embedding import embedding_manager
from app.logger import logger


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    score: float
    source_page: int
    chunk_index: int
    chunk_type: str = "text"
    paper_id: int = -1


class RAGEngine:
    """多路召回 RAG 引擎"""

    def __init__(self):
        self.es = Elasticsearch(
            hosts=settings["elasticsearch"]["hosts"],
            request_timeout=30  # elasticsearch 9.x 用 request_timeout（8.x 是 timeout）
        )
        self.index_prefix = settings["elasticsearch"]["index_prefix"]
        self.rrf_k = settings["rag"]["rrf_k"]
        self.retrieval_top_k = settings["rag"]["retrieval_top_k"]
        self.rerank_top_k = settings["rag"]["rerank_top_k"]
        self.rerank_candidates = settings["rag"]["rerank_candidates"]

    def _index_name(self, paper_id: int) -> str:
        """根据 paper_id 生成索引名"""
        return f"{self.index_prefix}_{paper_id}"

    # ──────────────────────────────────────────────
    # 索引管理
    # ──────────────────────────────────────────────

    def create_index(self, paper_id: int) -> None:
        """创建一篇论文的 ES 索引"""
        index_name = self._index_name(paper_id)

        if self.es.indices.exists(index=index_name):
            return

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "chunk_content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                        "fields": {
                            "english": {"type": "text", "analyzer": "english"}
                        }
                    },
                    "embedding_vector": {
                        "type": "dense_vector",
                        "dims": embedding_manager.dimension,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 16,
                            "ef_construction": 256
                        }
                    },
                    "paper_id": {"type": "integer"},
                    "source_page": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "chunk_type": {"type": "keyword"}
                }
            }
        }

        self.es.indices.create(index=index_name, body=mapping)
        logger.info(f"创建 ES 索引: {index_name}")

    def delete_index(self, paper_id: int) -> None:
        """删除一篇论文的索引（同步删除，避免孤儿数据）"""
        index_name = self._index_name(paper_id)
        try:
            if self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
                logger.info(f"已删除 ES 索引: {index_name}")
        except Exception as e:
            logger.error(f"删除索引失败 {index_name}: {e}")

    # ──────────────────────────────────────────────
    # 索引数据写入
    # ──────────────────────────────────────────────

    def index_chunks(self, paper_id: int, chunks: List[dict]) -> int:
        """
        批量向量化 + 写入 ES（用 helpers.bulk，而非单条循环）

        参数 chunks: [{"content", "source_page", "chunk_index", "chunk_type"}]
        返回成功写入条数
        """
        if not chunks:
            return 0

        index_name = self._index_name(paper_id)
        self.create_index(paper_id)

        # 批量向量化
        texts = [c["content"] for c in chunks]
        vectors = embedding_manager.embed(texts)

        # 构造 bulk actions
        actions = []
        for i, chunk in enumerate(chunks):
            actions.append({
                "_index": index_name,
                "_source": {
                    "chunk_content": chunk["content"],
                    "embedding_vector": vectors[i].tolist(),
                    "paper_id": paper_id,
                    "source_page": chunk.get("source_page", -1),
                    "chunk_index": chunk.get("chunk_index", i),
                    "chunk_type": chunk.get("chunk_type", "text")
                }
            })

        # bulk 批量写入
        success, errors = bulk(self.es, actions, raise_on_error=False, chunk_size=500)
        if errors:
            logger.warning(f"ES 批量写入部分失败: {len(errors)} 条错误")
        self.es.indices.refresh(index=index_name)
        logger.info(f"索引 {index_name} 写入 {success} 条")
        return success

    # ──────────────────────────────────────────────
    # 检索
    # ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        paper_ids: List[int],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        完整检索流程：BM25 + 向量 → RRF → 重排序

        支持 paper_ids 传多个（综述场景），在多个索引中并行检索
        """
        if not paper_ids:
            return []

        if top_k is None:
            top_k = self.rerank_top_k

        # 单论文 vs 多论文检索
        if len(paper_ids) == 1:
            bm25_results = self._bm25_search(query, self._index_name(paper_ids[0]))
            vector_results = self._vector_search(query, self._index_name(paper_ids[0]))
        else:
            bm25_results = self._bm25_search_multi(query, paper_ids)
            vector_results = self._vector_search_multi(query, paper_ids)

        # RRF 融合
        fused = self._rrf_fusion(bm25_results, vector_results)

        if not fused:
            return []

        # 取前 rerank_candidates 个做精排
        candidates = fused[:min(self.rerank_candidates, len(fused))]
        candidate_texts = [hit["_source"]["chunk_content"] for hit in candidates]

        # 懒加载 reranker（避免启动时强制下载模型）
        from app.core.reranker import reranker
        ranked = reranker.rerank(query, candidate_texts, top_k=top_k)

        results = []
        for orig_idx, score in ranked:
            source = candidates[orig_idx]["_source"]
            results.append(RetrievalResult(
                content=source["chunk_content"],
                score=score,
                source_page=source.get("source_page", -1),
                chunk_index=source.get("chunk_index", -1),
                chunk_type=source.get("chunk_type", "text"),
                paper_id=source.get("paper_id", -1)
            ))

        return results

    def _bm25_search(self, query: str, index_name: str) -> List[dict]:
        """单索引 BM25 检索"""
        try:
            response = self.es.search(
                index=index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunk_content", "chunk_content.english^2"]
                        }
                    },
                    "size": self.retrieval_top_k
                }
            )
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []

    def _vector_search(self, query: str, index_name: str) -> List[dict]:
        """单索引 KNN 向量检索"""
        try:
            query_vector = embedding_manager.embed_query(query)
            response = self.es.search(
                index=index_name,
                body={
                    "knn": {
                        "field": "embedding_vector",
                        "query_vector": query_vector,
                        "k": self.retrieval_top_k,
                        "num_candidates": self.retrieval_top_k * 2
                    },
                    "size": self.retrieval_top_k
                }
            )
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def _bm25_search_multi(self, query: str, paper_ids: List[int]) -> List[dict]:
        """多索引并行 BM25 检索（综述用）"""
        indices = [self._index_name(pid) for pid in paper_ids]
        existing = [idx for idx in indices if self.es.indices.exists(index=idx)]
        if not existing:
            return []
        try:
            response = self.es.search(
                index=existing,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunk_content", "chunk_content.english^2"]
                        }
                    },
                    "size": self.retrieval_top_k
                }
            )
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f"多索引 BM25 失败: {e}")
            return []

    def _vector_search_multi(self, query: str, paper_ids: List[int]) -> List[dict]:
        """多索引并行 KNN 检索（综述用）"""
        indices = [self._index_name(pid) for pid in paper_ids]
        existing = [idx for idx in indices if self.es.indices.exists(index=idx)]
        if not existing:
            return []
        try:
            query_vector = embedding_manager.embed_query(query)
            response = self.es.search(
                index=existing,
                body={
                    "knn": {
                        "field": "embedding_vector",
                        "query_vector": query_vector,
                        "k": self.retrieval_top_k,
                        "num_candidates": self.retrieval_top_k * 2
                    },
                    "size": self.retrieval_top_k
                }
            )
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f"多索引 KNN 失败: {e}")
            return []

    def _rrf_fusion(
        self,
        bm25_results: List[dict],
        vector_results: List[dict]
    ) -> List[dict]:
        """
        Reciprocal Rank Fusion

        核心公式：RRF_score = Σ 1 / (rank + k)，rank 从 1 起算（ES 官方标准）
        """
        k = self.rrf_k
        fusion_scores: Dict[str, float] = {}
        doc_map: Dict[str, dict] = {}

        # rank 从 1 开始（enumerate 默认从 0，所以 +1）
        for rank, hit in enumerate(bm25_results, 1):
            doc_id = hit["_id"] + "@" + hit["_index"]  # 跨索引需加索引名区分
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + 1.0 / (rank + k)
            doc_map[doc_id] = hit

        for rank, hit in enumerate(vector_results, 1):
            doc_id = hit["_id"] + "@" + hit["_index"]
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + 1.0 / (rank + k)
            doc_map[doc_id] = hit

        sorted_ids = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_ids]


# 全局实例
rag_engine = RAGEngine()

"""
RAG 检索评测 —— 对比 4 种策略

策略：
  A. 纯 BM25（关键词）
  B. 纯向量（语义）
  C. BM25 + 向量 + RRF（混合）
  D. C + BGE-reranker（精排）

指标：
  - MRR（Mean Reciprocal Rank）：第一个相关结果的排名倒数平均
  - Recall@3 / Recall@5：前 K 个结果中包含相关的比例
"""
import os
import json
from pathlib import Path
from typing import List, Dict
from elasticsearch import Elasticsearch
from app.config import settings
from app.core.embedding import embedding_manager
from app.core.rag_engine import rag_engine
from app.logger import logger

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def load_dataset(path: str = "eval/dataset.json") -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# 四种检索策略
# ──────────────────────────────────────────────

def strategy_bm25(query: str, paper_id: int, top_k: int = 5) -> List[dict]:
    """策略 A：纯 BM25"""
    index_name = rag_engine._index_name(paper_id)
    if not rag_engine.es.indices.exists(index=index_name):
        return []
    response = rag_engine.es.search(
        index=index_name,
        body={
            "query": {"multi_match": {"query": query, "fields": ["chunk_content", "chunk_content.english^2"]}},
            "size": top_k
        }
    )
    return response["hits"]["hits"]


def strategy_vector(query: str, paper_id: int, top_k: int = 5) -> List[dict]:
    """策略 B：纯向量"""
    index_name = rag_engine._index_name(paper_id)
    if not rag_engine.es.indices.exists(index=index_name):
        return []
    query_vector = embedding_manager.embed_query(query)
    response = rag_engine.es.search(
        index=index_name,
        body={
            "knn": {"field": "embedding_vector", "query_vector": query_vector,
                    "k": top_k, "num_candidates": top_k * 2},
            "size": top_k
        }
    )
    return response["hits"]["hits"]


def strategy_hybrid(query: str, paper_id: int, top_k: int = 5) -> List[dict]:
    """策略 C：BM25 + 向量 + RRF"""
    bm25 = strategy_bm25(query, paper_id, top_k=50)
    vector = strategy_vector(query, paper_id, top_k=50)
    fused = rag_engine._rrf_fusion(bm25, vector)
    return fused[:top_k]


def strategy_full(query: str, paper_id: int, top_k: int = 5) -> List[dict]:
    """策略 D：C + reranker"""
    from app.core.reranker import reranker
    fused = strategy_hybrid(query, paper_id, top_k=10)  # 降到 10 个候选加速
    if not fused:
        return []
    texts = [hit["_source"]["chunk_content"] for hit in fused]
    ranked = reranker.rerank(query, texts, top_k=top_k)
    return [fused[idx] for idx, _ in ranked]


# ──────────────────────────────────────────────
# 评测指标
# ──────────────────────────────────────────────

def compute_metrics(results: List[dict], relevant_indices: set, top_k: int = 5) -> Dict[str, float]:
    """计算单条 query 的 MRR + Recall@K"""
    retrieved_indices = [hit["_source"].get("chunk_index", -1) for hit in results]

    # MRR：第一个相关的排名倒数
    mrr = 0.0
    for rank, idx in enumerate(retrieved_indices, 1):
        if idx in relevant_indices:
            mrr = 1.0 / rank
            break

    # Recall@K
    top3 = set(retrieved_indices[:3])
    top5 = set(retrieved_indices[:5])
    recall_at_3 = 1.0 if (top3 & relevant_indices) else 0.0
    recall_at_5 = 1.0 if (top5 & relevant_indices) else 0.0

    return {"mrr": mrr, "recall@3": recall_at_3, "recall@5": recall_at_5}


def evaluate_strategy(strategy_fn, dataset: List[dict], strategy_name: str) -> Dict:
    """评测单个策略在全集上的表现"""
    all_mrr, all_r3, all_r5 = [], [], []
    skipped = 0

    for item in dataset:
        paper_id = item["paper_id"]
        index_name = rag_engine._index_name(paper_id)
        if not rag_engine.es.indices.exists(index=index_name):
            skipped += 1
            continue
        try:
            results = strategy_fn(item["query"], paper_id, top_k=5)
            metrics = compute_metrics(results, set(item["relevant_chunk_indices"]))
            all_mrr.append(metrics["mrr"])
            all_r3.append(metrics["recall@3"])
            all_r5.append(metrics["recall@5"])
        except Exception as e:
            logger.error(f"评测失败 {strategy_name} {item['query'][:30]}: {e}")
            skipped += 1

    n = len(all_mrr)
    if n == 0:
        return {"strategy": strategy_name, "error": "no valid queries"}

    return {
        "strategy": strategy_name,
        "num_queries": n,
        "skipped": skipped,
        "mrr": sum(all_mrr) / n,
        "recall@3": sum(all_r3) / n,
        "recall@5": sum(all_r5) / n
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run_full_evaluation():
    logger.info("=" * 60)
    logger.info("开始 RAG 检索评测")
    logger.info("=" * 60)

    dataset = load_dataset()
    # 抽样 20 条（每篇 10 条）避免 reranker 策略过慢
    import random
    random.seed(42)
    if len(dataset) > 20:
        dataset = random.sample(dataset, 20)
    logger.info(f"加载评测集: {len(dataset)} 条（抽样）")

    strategies = [
        ("A. 纯 BM25", strategy_bm25),
        ("B. 纯向量", strategy_vector),
        ("C. BM25+向量+RRF", strategy_hybrid),
        ("D. C+Reranker", strategy_full),
    ]

    # 预加载 reranker（避免策略 D 每条重新加载）
    logger.info("预加载 Reranker 模型...")
    from app.core.reranker import reranker
    _ = reranker  # 触发单例加载

    results = []
    for name, fn in strategies:
        logger.info(f"\n评测策略: {name}")
        r = evaluate_strategy(fn, dataset, name)
        results.append(r)
        if "error" not in r:
            logger.info(f"  {name}: MRR={r['mrr']:.3f}, R@3={r['recall@3']:.1%}, R@5={r['recall@5']:.1%}")

    # 输出 Markdown 表
    print("\n\n" + "=" * 60)
    print("📊 评测结果对比表")
    print("=" * 60)
    print(f"| 策略 | 样本数 | MRR | Recall@3 | Recall@5 |")
    print(f"|------|--------|-----|----------|----------|")
    for r in results:
        if "error" not in r:
            print(f"| {r['strategy']} | {r['num_queries']} | {r['mrr']:.3f} | {r['recall@3']:.1%} | {r['recall@5']:.1%} |")

    output_path = Path("eval/results_retrieval.json")
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"结果已保存: {output_path}")

    return results


# ──────────────────────────────────────────────
# 可视化（matplotlib 对比图，放 README）
# ──────────────────────────────────────────────

def plot_results(results, output_path: str = "eval/results_chart.png"):
    import matplotlib
    matplotlib.use('Agg')  # 无界面环境
    import matplotlib.pyplot as plt
    import numpy as np

    # 中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    strategies = [r["strategy"] for r in results if "error" not in r]
    mrr = [r["mrr"] for r in results if "error" not in r]
    r3 = [r["recall@3"] for r in results if "error" not in r]
    r5 = [r["recall@5"] for r in results if "error" not in r]

    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, mrr, width, label='MRR', color='#2196F3')
    bars2 = ax.bar(x, r3, width, label='Recall@3', color='#4CAF50')
    bars3 = ax.bar(x + width, r5, width, label='Recall@5', color='#FF9800')

    ax.set_ylabel('Score')
    ax.set_title('PaperLens RAG 检索策略对比（4 策略 × 60 条问答对）')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # 在柱子上标数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"对比图已保存: {output_path}")


if __name__ == "__main__":
    results = run_full_evaluation()
    plot_results(results)

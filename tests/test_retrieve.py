"""Day 1 验收：检索测试"""
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from app.core.rag_engine import rag_engine

query = "What is GraphRAG and how does it use community detection?"
results = rag_engine.retrieve(query, [1], top_k=3)
print(f"查询: {query}")
print(f"检索到 {len(results)} 条结果:\n")
for i, r in enumerate(results, 1):
    print(f"[{i}] score={r.score:.3f} page={r.source_page} type={r.chunk_type}")
    content = r.content[:150].replace('\n', ' ')
    print(f"    {content}...\n")

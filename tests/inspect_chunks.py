"""检查 ES 索引里的 chunk 内容质量"""
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from app.core.rag_engine import rag_engine

# 抽样前 5 个 chunk
resp = rag_engine.es.search(
    index='paperlens_paper_1',
    body={'query': {'match_all': {}}, 'size': 5, 'sort': [{'chunk_index': 'asc'}]}
)
print("=== 前 5 个 chunk（按顺序）===")
for h in resp['hits']['hits']:
    c = h['_source']['chunk_content'][:250].replace('\n', ' ')
    print(f"[idx={h['_source']['chunk_index']}] {c}")
    print('---')

# 统计含 HTML 标签的 chunk 比例
resp2 = rag_engine.es.search(
    index='paperlens_paper_1',
    body={'query': {'match_phrase': {'chunk_content': '<td'}}, 'size': 0}
)
td_count = resp2['hits']['total']['value'] if isinstance(resp2['hits']['total'], dict) else resp2['hits']['total']

resp3 = rag_engine.es.search(
    index='paperlens_paper_1',
    body={'query': {'match_all': {}}, 'size': 0}
)
total = resp3['hits']['total']['value'] if isinstance(resp3['hits']['total'], dict) else resp3['hits']['total']

print(f"\n=== 统计 ===")
print(f"总 chunk 数: {total}")
print(f"含 <td> 标签的 chunk 数: {td_count} ({td_count/total*100:.1f}%)")

# 看含 'Abstract' 或 'Introduction' 的 chunk（正文标志）
resp4 = rag_engine.es.search(
    index='paperlens_paper_1',
    body={'query': {'match_phrase': {'chunk_content': 'Abstract'}}, 'size': 1}
)
print(f"含 'Abstract' 的 chunk: {resp4['hits']['total']['value'] if isinstance(resp4['hits']['total'], dict) else resp4['hits']['total']}")

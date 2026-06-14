"""调试：看 LLM 对单个 chunk 的真实返回"""
import os, sys
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'E:\其他\大模型项目\PaperLens')

import json
from app.config import settings
from openai import OpenAI
from app.core.rag_engine import rag_engine

c = OpenAI(api_key=settings["llm"]["api_key"], base_url=settings["llm"]["base_url"])

# 取一个 chunk
resp = rag_engine.es.search(index='paperlens_paper_1', body={
    'query': {'match_all': {}}, 'size': 1, 'sort': [{'chunk_index': 'asc'}]
})
content = resp['hits']['hits'][0]['_source']['chunk_content'][:600]

prompt = """你是一个学术问答数据集生成器。基于下面从论文中摘录的文本片段，生成一个具体的问题，这个问题的答案能在这段文本中找到。

要求：
1. 问题必须具体（能从该片段直接回答），不要泛泛而谈
2. 问题用中文，专业术语保留英文
3. 同时给出一个简短的参考答案（50字以内）
4. 输出 JSON：{"query": "...", "reference_answer": "..."}

论文片段：
""" + content

r = c.chat.completions.create(
    model=settings["llm"]["fast_model"],
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200, temperature=0.3
)
text = r.choices[0].message.content

# 写到文件避免编码问题
with open(r'E:\其他\大模型项目\PaperLens\tests\debug_output.txt', 'w', encoding='utf-8') as f:
    f.write("=== RAW OUTPUT ===\n")
    f.write(text)
    f.write("\n\n=== repr ===\n")
    f.write(repr(text))

print("已写入 tests/debug_output.txt")
print("长度:", len(text))

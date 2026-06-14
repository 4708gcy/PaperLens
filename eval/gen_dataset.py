"""
自动生成评测数据集：从已索引的论文 chunks 自动生成问答对

策略：
1. 从 ES 抽样若干 chunks（覆盖论文不同部分）
2. 用 Qwen 为每个 chunk 生成 1 个问题（ground truth 来源就是这个 chunk）
3. 这样 relevant_chunk_indices 自动已知 = 该 chunk 的 chunk_index
4. 同时生成 reference_answer

这比手动标注快 100 倍，且可复现、可扩展到任意论文。
"""
import json
import os
from pathlib import Path
from openai import OpenAI
from app.config import settings
from app.core.rag_engine import rag_engine
from app.logger import logger

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

client = OpenAI(
    api_key=settings["llm"]["api_key"],
    base_url=settings["llm"]["base_url"]
)

GENERATE_PROMPT_TEMPLATE = """你是一个学术问答数据集生成器。基于下面从论文中摘录的文本片段，生成一个具体的问题，这个问题的答案能在这段文本中找到。

要求：
1. 问题必须具体（能从该片段直接回答），不要泛泛而谈
2. 问题用中文，专业术语保留英文
3. 同时给出一个简短的参考答案（50字以内）
4. 只输出 JSON，不要任何其他内容，不要 markdown 代码块包裹
5. JSON 格式：两个 key 是 query 和 reference_answer

论文片段：
{chunk}"""


def build_prompt(chunk: str) -> str:
    """用字符串替换（避免 .format 把 JSON 花括号当占位符）"""
    return GENERATE_PROMPT_TEMPLATE.replace("{chunk}", chunk)


def generate_dataset(paper_id: int, num_questions: int = 30) -> list:
    """为一篇论文自动生成评测问答对"""
    index_name = rag_engine._index_name(paper_id)

    if not rag_engine.es.indices.exists(index=index_name):
        print(f"索引 {index_name} 不存在")
        return []

    # 抽样 chunks：均匀分布，覆盖全文
    resp = rag_engine.es.search(
        index=index_name,
        body={
            "query": {"match_all": {}},
            "size": num_questions * 3,  # 多取一些，过滤后够用
            "sort": [{"chunk_index": "asc"}]
        }
    )
    all_chunks = resp["hits"]["hits"]
    # 均匀抽样
    step = max(1, len(all_chunks) // num_questions)
    sampled = [all_chunks[i] for i in range(0, len(all_chunks), step)][:num_questions]

    # 过滤掉太短的 chunk（< 100 字符的通常是表格碎片）
    sampled = [h for h in sampled if len(h["_source"]["chunk_content"]) > 100]

    dataset = []
    for i, hit in enumerate(sampled):
        src = hit["_source"]
        content = src["chunk_content"][:800]  # 截断避免 prompt 过长
        chunk_idx = src["chunk_index"]

        print(f"  [{i+1}/{len(sampled)}] 生成问题 (chunk_index={chunk_idx})...", end=" ")

        try:
            resp = client.chat.completions.create(
                model=settings["llm"]["fast_model"],  # qwen-turbo 省钱
                messages=[{
                    "role": "user",
                    "content": build_prompt(content)
                }],
                max_tokens=200,
                temperature=0.3
            )
            text = resp.choices[0].message.content.strip()

            # 健壮的 JSON 提取（兼容 markdown 包裹、前后多余文本）
            import re
            # 先尝试直接解析
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                # 用正则找第一个 {...} 块
                match = re.search(r'\{[^{}]*"query"[^{}]*\}', text, re.DOTALL)
                if match:
                    item = json.loads(match.group())
                else:
                    raise ValueError(f"无法提取 JSON，原文: {text[:200]}")

            item["paper_id"] = paper_id
            item["relevant_chunk_indices"] = [chunk_idx]  # ground truth
            dataset.append(item)
            print(f"OK {item['query'][:40]}")
        except Exception as e:
            err_msg = str(e).replace('\n', ' ')[:100]
            print(f"FAIL: {err_msg}")

    return dataset


def main():
    import time
    all_data = []
    for paper_id in [1, 2]:  # 两篇 GraphRAG 论文
        print(f"\n=== 为 paper_id={paper_id} 生成评测集 ===")
        data = generate_dataset(paper_id, num_questions=30)
        all_data.extend(data)
        print(f"  生成 {len(data)} 条")

    output_path = Path("eval/dataset.json")
    output_path.write_text(
        json.dumps(all_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n[OK] 共 {len(all_data)} 条，已保存到 {output_path}")
    print(f"    paper 1: {sum(1 for d in all_data if d['paper_id']==1)} 条")
    print(f"    paper 2: {sum(1 for d in all_data if d['paper_id']==2)} 条")


if __name__ == "__main__":
    main()

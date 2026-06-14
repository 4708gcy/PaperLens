"""
答案质量评测 —— LLM-as-Judge

对比"有 RAG 增强"vs"无 RAG 直接答"的答案质量
用 Qwen 作为评判，1-5 分打分
"""
import os
import json
import random
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

# 注意：JUDGE_PROMPT 里不能有 { } 字面量（会和 .format 冲突）
# 用占位符标记，运行时用 replace 注入
JUDGE_PROMPT_TEMPLATE = """你是论文问答质量评测员。对比两个答案的质量。

问题：__QUESTION__
参考答案：__REFERENCE__

答案 A（无 RAG）：__ANSWER_A__
答案 B（有 RAG）：__ANSWER_B__

请分别给两个答案打分（1-5 分）：
- 5 分：完全正确，包含关键信息
- 3 分：部分正确
- 1 分：错误或无关

只返回 JSON（不要 markdown 代码块）：
{
  "score_a": 1到5的整数,
  "score_b": 1到5的整数,
  "reason": "简短说明哪个更好"
}"""


def answer_without_rag(question: str) -> str:
    """无 RAG：直接问 LLM"""
    response = client.chat.completions.create(
        model=settings["llm"]["model"],
        messages=[
            {"role": "system", "content": "你是论文研究助手，凭你的知识回答。"},
            {"role": "user", "content": question}
        ],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content


def answer_with_rag(question: str, paper_id: int) -> str:
    """有 RAG：先检索再答"""
    results = rag_engine.retrieve(question, [paper_id], top_k=5)
    context = "\n\n".join([f"[{r.content}]" for r in results]) or "无相关内容"

    response = client.chat.completions.create(
        model=settings["llm"]["model"],
        messages=[
            {"role": "system", "content": f"你是论文研究助手。基于检索片段回答：\n{context}"},
            {"role": "user", "content": question}
        ],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content


def judge(question: str, reference: str, answer_a: str, answer_b: str) -> dict:
    """LLM 打分"""
    prompt = (JUDGE_PROMPT_TEMPLATE
              .replace("__QUESTION__", question)
              .replace("__REFERENCE__", reference[:200])
              .replace("__ANSWER_A__", answer_a[:400])
              .replace("__ANSWER_B__", answer_b[:400]))

    response = client.chat.completions.create(
        model=settings["llm"]["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )
    text = response.choices[0].message.content.strip()

    # 健壮解析
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"score_a": 3, "score_b": 3, "reason": "parse error"}


def run_answer_evaluation(dataset_path: str = "eval/dataset.json", num_samples: int = 15):
    """跑答案质量评测"""
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    random.seed(42)
    if len(dataset) > num_samples:
        dataset = random.sample(dataset, num_samples)

    logger.info(f"答案质量评测：{len(dataset)} 条")
    results = []

    for i, item in enumerate(dataset):
        logger.info(f"[{i+1}/{len(dataset)}] {item['query'][:50]}")
        try:
            ans_a = answer_without_rag(item["query"])
            ans_b = answer_with_rag(item["query"], item["paper_id"])
            judge_result = judge(item["query"], item.get("reference_answer", ""), ans_a, ans_b)
            results.append({
                "query": item["query"],
                "score_no_rag": judge_result.get("score_a"),
                "score_with_rag": judge_result.get("score_b"),
                "reason": judge_result.get("reason", "")
            })
        except Exception as e:
            logger.error(f"评测失败: {e}")

    # 汇总
    valid = [r for r in results if r["score_no_rag"] and r["score_with_rag"]]
    if valid:
        avg_a = sum(r["score_no_rag"] for r in valid) / len(valid)
        avg_b = sum(r["score_with_rag"] for r in valid) / len(valid)
        print(f"\n{'='*50}")
        print(f"📊 答案质量对比（LLM-as-Judge, {len(valid)} 条）")
        print(f"{'='*50}")
        print(f"  无 RAG 平均分: {avg_a:.2f} / 5")
        print(f"  有 RAG 平均分: {avg_b:.2f} / 5")
        if avg_a > 0:
            print(f"  RAG 提升: +{avg_b - avg_a:.2f} 分 ({(avg_b-avg_a)/avg_a*100:+.1f}%)")
        print(f"\n示例:")
        for r in valid[:3]:
            print(f"  Q: {r['query'][:40]}")
            print(f"    无RAG={r['score_no_rag']}/5  有RAG={r['score_with_rag']}/5  ({r['reason'][:60]})")

    Path("eval/results_answer.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("答案质量结果已保存: eval/results_answer.json")
    return results


if __name__ == "__main__":
    run_answer_evaluation()

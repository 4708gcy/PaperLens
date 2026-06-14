"""Cross-Encoder 重排序器：对粗排候选精排"""
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.config import settings
from app.logger import logger


class Reranker:
    """
    BGE-reranker-v2-m3 Cross-Encoder

    为什么需要两阶段检索（粗排+精排）？
    — 第一阶段（Bi-Encoder）：query 和 doc 分别编码→向量匹配。
      O(1) 比较，适合全量检索，但精度有限。
    — 第二阶段（Cross-Encoder）：query 和 doc 拼接后一起编码。
      精度大幅提升，但每对需要一次完整前向传播，速度慢。
    — 所以：粗排选 top-50 → 重排序选 top-5。
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            model_name = settings["reranker"]["model"]
            logger.info(f"加载 Reranker 模型: {model_name}")
            cls._instance._tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._instance._model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            cls._instance._model.eval()
        return cls._instance

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        对文档列表重排序。

        返回: [(原始索引, 相关性分数), ...]  按分数降序
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]

        inputs = self._tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        )

        with torch.no_grad():
            scores = self._model(**inputs, return_dict=True).logits.view(-1).float()

        scores_np = scores.cpu().numpy()
        ranked_indices = np.argsort(scores_np)[::-1][:top_k]

        return [(int(idx), float(scores_np[idx])) for idx in ranked_indices]


# 全局单例
reranker = Reranker()

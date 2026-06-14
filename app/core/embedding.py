"""Embedding 模型管理：单例加载 BGE-m3，提供统一向量化接口"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.logger import logger


class EmbeddingManager:
    """
    BGE-m3 Embedding 管理器（单例）

    为什么单例？
    — SentenceTransformer 加载约需 30-60 秒（m3 模型 ~2GB），占用 ~2GB 内存
    — 全局只需一个实例，避免重复加载

    为什么选 BGE-m3 而非 BGE-small-zh？
    — 论文以英文为主，m3 是多语言模型，英文表现远超中文专用版
    — 1024 维（比 small 的 512 维语义更丰富）
    — MTEB 榜单中英表现都强
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            model_name = settings["embedding"]["model"]
            logger.info(f"加载 Embedding 模型: {model_name}（首次较慢，约 30-60s）")
            self._model = SentenceTransformer(model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """向量维度（BGE-m3 = 1024）"""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        批量文本 → 向量矩阵

        返回 numpy 数组，shape = (len(texts), dimension)
        """
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # L2 归一化 → 点积 = 余弦相似度
            show_progress_bar=len(texts) > 50
        )
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """单个查询文本 → 向量"""
        embedding = self.embed([query])
        return embedding[0].tolist()


# 全局单例
embedding_manager = EmbeddingManager()

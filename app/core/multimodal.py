"""
qwen3.7-plus 多模态图表理解

复用主力 LLM（DashScope 的 qwen3.7-plus，支持图片输入）理解论文图表，
无需单独的多模态服务（原 MiMo 已替换为 qwen3.7-plus）。

调用范式（OpenAI 兼容）：
  messages=[{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,{b64}"}},
      {"type":"text","text":"描述这张图"}
  ]}]
"""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
from app.config import settings
from app.logger import logger


class MultimodalManager:
    """多模态图表理解管理（复用 qwen3.7-plus 的视觉能力）"""

    def __init__(self):
        # 复用主力 LLM 配置（DashScope OpenAI 兼容接口 + qwen3.7-plus，支持图片输入）
        self.client = OpenAI(
            api_key=settings["llm"]["api_key"],
            base_url=settings["llm"]["base_url"]
        )
        self.model = settings["llm"]["model"]

    def _encode_image(self, image_path: str) -> str:
        """图片转 base64 data URL"""
        path = Path(image_path)
        ext = path.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "webp": "webp", "gif": "gif", "bmp": "bmp"}.get(ext, "jpeg")

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"

    def understand_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        让 qwen3.7-plus 理解一张图片，返回自然语言描述

        参数：
            image_path: 图片文件路径
            prompt: 自定义提问（默认用于论文图表理解的通用提问）
        """
        if prompt is None:
            # 默认 prompt：论文图表理解的通用模板
            prompt = (
                "This is a figure from an academic paper. "
                "Please describe: (1) what type of figure it is "
                "(architecture diagram, experiment chart, table, etc), "
                "(2) what it shows or compares, (3) key takeaways. "
                "Be concise (under 150 words)."
            )

        data_url = self._encode_image(image_path)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                # 300 tokens 足够一段图表描述（~150 词）；公式转 LaTeX 更短。
                # 原来 500 偏大，砍掉既省时间又省成本。
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"图表理解调用失败 {image_path}: {e}")
            return ""

    def describe_paper_images(
        self,
        images_dir: str,
        max_images: int = 10,
        max_workers: int = 5
    ) -> List[dict]:
        """
        批量理解论文图片目录下的所有图片（并发）。

        返回: [{"image_path", "description"}]

        性能：之前是串行（60 张约 32 分钟），改成 ThreadPoolExecutor 并发后
        5 worker 约可降到 7 分钟。DashScope OpenAI 兼容接口支持并发，
        max_workers=5 是 QPS 限额内的安全起步值（配合 LLM 的 max_retries=5）。
        """
        images_dir = Path(images_dir)
        if not images_dir.exists():
            return []

        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        images = sorted([
            p for p in images_dir.rglob("*") if p.suffix.lower() in image_exts
        ])[:max_images]

        if not images:
            return []

        # 单张就直接调，省得起线程池
        if len(images) == 1:
            logger.info(f"图表理解: {images[0].name}")
            desc = self.understand_image(str(images[0]))
            return [{"image_path": str(images[0]), "description": desc}] if desc else []

        results: List[dict] = []
        # 并发：每个图片一个任务，as_completed 按完成顺序回收
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_img = {
                ex.submit(self.understand_image, str(img)): img for img in images
            }
            done = 0
            total = len(future_to_img)
            for future in as_completed(future_to_img):
                img = future_to_img[future]
                done += 1
                try:
                    desc = future.result()
                    if desc:
                        results.append({"image_path": str(img), "description": desc})
                    if done % 5 == 0 or done == total:
                        logger.info(f"图表理解进度: {done}/{total}")
                except Exception as e:
                    logger.warning(f"图表理解失败 {img.name}: {e}")
        # 按原始图片顺序排序，保证输出稳定
        order = {str(p): i for i, p in enumerate(images)}
        results.sort(key=lambda r: order.get(r["image_path"], 0))
        return results


# 全局单例
multimodal_manager = MultimodalManager()

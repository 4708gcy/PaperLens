"""
文档处理服务：格式转换 → 拆分 → MinerU 转换 → 合并 Markdown → 分块 → 向量化 → ES 索引

完整流程：
1. 用户上传 PDF / DOCX / PPTX / ...
2. 非 PDF 先用 LibreOffice 转 PDF
3. 如果 PDF > 200 页 → pypdf 拆分（MinerU 上限 200 页）
4. 每个子 PDF 调用 mineru-open-api extract 转 Markdown
5. 按顺序合并 Markdown
6. 滑动窗口分块
7. 批量 Embedding + ES bulk 索引

⚠️ 关键：整个后端进程必须在 `conda activate ocr` 环境启动，
否则 mineru-open-api 命令不可用！
"""
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from app.core.chunker import split_text_with_overlap, TextChunk
from app.core.pdf_splitter import (
    convert_to_pdf, split_pdf, is_pdf, get_pdf_page_count, SUPPORTED_NON_PDF,
    _resolve_cmd
)
from app.core.rag_engine import rag_engine
from app.config import settings
from app.exceptions import DocumentProcessError
from app.logger import logger


# 启动时探测一次 mineru-open-api 完整路径（避免后台线程 PATH 缺失问题）
MINERU_CMD_PATH = _resolve_cmd("mineru-open-api")
if MINERU_CMD_PATH:
    logger.info(f"✓ mineru-open-api 探测到: {MINERU_CMD_PATH}")
else:
    logger.warning(
        "⚠️ mineru-open-api 未探测到！文档解析将失败。"
        "请确认后端在 conda activate ocr 环境启动。"
    )


class DocumentService:
    """文档处理服务"""

    # 支持的所有输入格式（PDF + LibreOffice 可转的 + 图片）
    SUPPORTED_EXTENSIONS = {".pdf"} | SUPPORTED_NON_PDF | {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def process_document(self, file_path: str, paper_id: int,
                         title: str = "") -> Dict:
        """
        处理一个文档：转换 → 拆分 → MinerU → 合并 → 分块 → 索引

        返回: {"markdown_path", "total_chunks", "indexed", "page_count"}
        """
        input_path = Path(file_path)
        papers_dir = Path(settings["document"]["papers_dir"])
        paper_dir = papers_dir / str(paper_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        converted_pdf_to_clean = None  # 记录 LibreOffice 生成的临时 PDF（结束清理）
        split_parts_to_clean = []     # 记录 pypdf 拆分出的临时子 PDF（结束清理）

        try:
            # ── Step 1: 非 PDF → PDF（LibreOffice）──
            if is_pdf(file_path):
                pdf_path = file_path
                logger.info(f"输入是 PDF，跳过格式转换: {input_path.name}")
            else:
                logger.info(f"非 PDF 格式（{input_path.suffix}），用 LibreOffice 转 PDF...")
                pdf_path = convert_to_pdf(file_path, output_dir=str(paper_dir))
                converted_pdf_to_clean = pdf_path

            # ── Step 2: PDF 拆分（>200 页）──
            page_count = get_pdf_page_count(pdf_path)
            sub_files = split_pdf(pdf_path, max_pages=settings["document"]["max_pdf_pages"])
            # 拆分产物（sub_files 里除原始 pdf_path 外的都是临时文件，结束后清理）
            split_parts_to_clean = [p for p in sub_files if p != pdf_path]
            logger.info(f"PDF {input_path.name}（{page_count} 页）→ {len(sub_files)} 份")

            # ── Step 3: mineru-open-api extract 转 Markdown ──
            md_files: List[Path] = []
            for sub_file in sub_files:
                md_path = self._run_mineru(sub_file, paper_dir)
                if md_path:
                    md_files.append(md_path)

            if not md_files:
                raise DocumentProcessError(f"MinerU 转换失败，无 Markdown 输出: {file_path}")

            # ── Step 4: 合并 Markdown ──
            if len(md_files) == 1:
                markdown_content = md_files[0].read_text(encoding="utf-8")
            else:
                markdown_content = self._merge_markdowns(md_files)

            if not markdown_content.strip():
                raise DocumentProcessError(f"Markdown 内容为空: {file_path}")

            # ── Step 5: 保存合并后的 Markdown ──
            merged_md_path = paper_dir / f"{input_path.stem}.md"
            merged_md_path.write_text(markdown_content, encoding="utf-8")

            # ── Step 5.5: 图表多模态理解（MiMo）──
            image_captions = []
            try:
                # MinerU 输出的图片在 mineru_xxx/images/ 目录
                from app.core.multimodal import multimodal_manager
                # 找所有 mineru 输出目录下的 images
                mineru_dirs = list(paper_dir.glob("mineru_*"))
                for md_dir in mineru_dirs:
                    images_subdir = md_dir / "images"
                    if images_subdir.exists():
                        image_captions.extend(
                            multimodal_manager.describe_paper_images(
                                str(images_subdir), max_images=10
                            )
                        )
                logger.info(f"图片理解完成: {len(image_captions)} 张")
            except Exception as e:
                logger.warning(f"图片理解失败（不影响主流程）: {e}")

            # ── Step 6: 分块（文本 + 图片描述合并）──
            chunk_size = settings["elasticsearch"]["chunk_size"]
            chunk_overlap = settings["elasticsearch"]["chunk_overlap"]

            chunks = split_text_with_overlap(
                text=markdown_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_page=-1,
                chunk_type="text"
            )

            # 把图片描述作为独立 chunk 追加（chunk_type=image_caption）
            from app.core.chunker import TextChunk
            for i, ic in enumerate(image_captions):
                img_chunk = TextChunk(
                    content=f"[图片描述] {ic['description']}",
                    chunk_index=len(chunks) + i,
                    start_char=0,
                    end_char=0,
                    source_page=-1,
                    chunk_type="image_caption"
                )
                chunks.append(img_chunk)

            # ── Step 7: 向量化 + ES 索引 ──
            chunk_dicts = [
                {
                    "content": c.content,
                    "source_page": c.source_page,
                    "chunk_index": c.chunk_index,
                    "chunk_type": c.chunk_type
                }
                for c in chunks
            ]

            indexed = rag_engine.index_chunks(paper_id, chunk_dicts)

            logger.info(
                f"文档处理完成: {input_path.name} → {merged_md_path.name}, "
                f"{len(chunks)} 块, {indexed} 条索引"
            )

            return {
                "markdown_path": str(merged_md_path),
                "total_chunks": len(chunks),
                "indexed": indexed,
                "page_count": page_count
            }

        except DocumentProcessError:
            raise
        except Exception as e:
            logger.error(f"文档处理异常: {e}", exc_info=True)
            raise DocumentProcessError(f"文档处理失败: {str(e)}")
        finally:
            # 清理 LibreOffice 生成的临时 PDF（保留原始上传文件和最终 Markdown）
            if converted_pdf_to_clean and Path(converted_pdf_to_clean).exists():
                try:
                    os.remove(converted_pdf_to_clean)
                except Exception:
                    pass
            # 清理 pypdf 拆分出的临时子 PDF（仅 >200 页的大论文会产生）
            for part in split_parts_to_clean:
                try:
                    if Path(part).exists():
                        os.remove(part)
                except Exception:
                    pass

    def _run_mineru(self, file_path: str, output_dir: Path) -> Optional[Path]:
        """
        调用 mineru-open-api extract 转 Markdown。

        ⚠️ 前置条件：
        1. 当前进程在 `conda activate ocr` 环境内（由启动者保证）
        2. 已执行过 `mineru-open-api auth`（extract 模式必需）

        命令：mineru-open-api extract <input> -o <outdir>
        输出：outdir 下生成 .md 文件 + images/ 等资源目录
        """
        input_path = Path(file_path).resolve()
        mineru_out = output_dir / f"mineru_{input_path.stem}"
        mineru_out.mkdir(parents=True, exist_ok=True)

        # extract 模式：完整 Markdown + 图片 + 表格 + LaTeX 公式
        if not MINERU_CMD_PATH:
            logger.error("mineru-open-api 路径未探测到，跳过")
            return None
        cmd = [
            MINERU_CMD_PATH, "extract",
            str(input_path),
            "-o", str(mineru_out),
            "-f", "md",
            "--timeout", "600",
        ]
        logger.info(f"MinerU 执行: {' '.join(cmd)}")

        # Windows 下 .cmd 包装脚本必须 shell=True
        use_shell = sys.platform == "win32" and MINERU_CMD_PATH.lower().endswith((".cmd", ".bat"))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=900,  # 整体超时 15 分钟（包含云端排队）
                encoding="utf-8",
                shell=use_shell
            )
            if result.returncode != 0:
                logger.warning(
                    f"mineru-open-api extract 失败 (exit {result.returncode}): "
                    f"{result.stderr[:300]}"
                )
                logger.info("尝试 flash-extract 模式兜底...")
                return self._run_mineru_flash(file_path, mineru_out)
        except subprocess.TimeoutExpired:
            logger.error(f"MinerU 超时（15分钟）: {file_path}")
            return None
        except FileNotFoundError:
            logger.error(
                "mineru-open-api 命令未找到！请确认：\n"
                "  1. 已 conda activate ocr\n"
                "  2. 已安装 mineru-open-api（uv tool install mineru-open-api）"
            )
            return None

        # 定位输出的 .md 文件（可能在 outdir 根目录或子目录）
        md_files = sorted(
            mineru_out.rglob("*.md"),
            key=lambda f: f.stat().st_mtime, reverse=True
        )
        return md_files[0] if md_files else None

    def _run_mineru_flash(self, file_path: str, output_dir: Path) -> Optional[Path]:
        """
        flash-extract 模式（兜底，无需 auth）。

        局限：图片/表格是占位符，质量低。仅当 extract 失败时用。
        """
        input_path = Path(file_path).resolve()
        if not MINERU_CMD_PATH:
            logger.error("mineru-open-api 路径未探测到，跳过 flash-extract")
            return None
        use_shell = sys.platform == "win32" and MINERU_CMD_PATH.lower().endswith((".cmd", ".bat"))
        cmd = [MINERU_CMD_PATH, "flash-extract", str(input_path), "-o", str(output_dir)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=300, encoding="utf-8",
                shell=use_shell
            )
            if result.returncode != 0:
                logger.error(f"flash-extract 也失败: {result.stderr[:300]}")
                return None
        except Exception as e:
            logger.error(f"flash-extract 异常: {e}")
            return None

        md_files = sorted(
            output_dir.rglob("*.md"),
            key=lambda f: f.stat().st_mtime, reverse=True
        )
        return md_files[0] if md_files else None

    def _merge_markdowns(self, md_files: List[Path]) -> str:
        """合并多个 Markdown 文件（按子文件顺序）"""
        parts = []
        for i, md_path in enumerate(md_files, 1):
            content = md_path.read_text(encoding="utf-8")
            parts.append(
                f"<!-- === 第 {i} 部分（来源: {md_path.name}）=== -->\n\n{content}\n"
            )
        return "\n".join(parts)


# 全局实例
document_service = DocumentService()

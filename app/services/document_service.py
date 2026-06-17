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


def _clean_markdown(text: str) -> str:
    """轻量排版清洗：让 MinerU 输出的 markdown 对 LLM 更友好。

    处理三类常见噪声（都是 MinerU 抽取扫描/排版文档时的典型产物）：
    1. 被错误断行的句子：行尾不是句末标点，下一行也不是列表/标题开头 → 拼回一行。
       （PDF 换行≠语义换行，硬换行会把一个句子切成多行，影响检索和阅读。）
    2. 3+ 连续空行压成 2 个。
    3. 独占一行的纯数字（常见页码）。

    刻意保守：不动 markdown 语法（#、-、*、`、表格 |）、不动图片、不动 HTML 注释。
    """
    import re

    # 1. 合并错误断行：行尾不是 句末标点/列表符/数字/反引号/竖线，
    #    且下一行首不是 标题#/列表-/数字编号/表格|/引用>/反引号`/空白(缩进代码块)，
    #    且【当前行不是标题行/列表行/代码围栏行】（用回调查 start 所在行首字符）。
    #    用回调避免 (?m)(?<!^) 在不同 regex 引用下行为不一致的问题。
    _line_start_chars = "#-*>"   # 这些字符开头的行是结构行，不参与合并

    def _merge(m):
        start = m.start()
        # 找 start 所在行的行首
        line_head = text.rfind("\n", 0, start) + 1
        first = text[line_head] if line_head < len(text) else ""
        if first in _line_start_chars:
            return m.group(0)   # 当前是结构行，保持原样不合并
        return m.group(1) + " " + m.group(2)   # 把行尾字符、空格、下一行首字符拼回

    text = re.sub(
        r'([^\n。？！：；”"’、•\-\d`|])\n([ \t]*[^\n•\-\d#`|>\s])',
        _merge,
        text,
    )
    # 2. 压缩 3+ 连续空行为 2 个
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 3. 去页码（独占一行的 1-4 位纯数字）
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    return text


class DocumentService:
    """文档处理服务"""

    # 支持的所有输入格式（PDF + LibreOffice 可转的 + 图片）
    SUPPORTED_EXTENSIONS = {".pdf"} | SUPPORTED_NON_PDF | {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def process_document(self, file_path: str, paper_id: int,
                         title: str = "", force_pdf: bool = True,
                         image_mode: str = None) -> Dict:
        """
        处理一个文档：转换 → 拆分 → MinerU → 合并 → 分块 → 索引

        返回: {"markdown_path", "total_chunks", "indexed", "page_count"}

        force_pdf（格式路由参数）：
        - True（默认）：非 PDF 先用 LibreOffice 转 PDF 再交给 MinerU。MinerU 对 PDF 解析最稳。
        - False：DOCX/PPTX 直接给 MinerU（MinerU 原生支持），跳过 LibreOffice。
          适用于"图片少、文本层完整"的可编辑 Word/PPT —— 省一次转换、有时质量更好。

        image_mode（图片理解开关，二态）：
        - "on"：处理全部图片（无上限）。论文/报告适用——架构图/实验图有价值，描述补进全文+ES。
        - "off"：完全跳过图片理解。教材/数学书适用——公式截图描述无意义，省 30+ 分钟。
        - None：用 config.yaml 的 document.image_mode 默认值。
        """
        # 解析 image_mode：None 时回落到 config 默认
        if image_mode is None:
            image_mode = settings.get("document", {}).get("image_mode", "on")
        image_mode = (image_mode or "on").lower()
        if image_mode not in ("on", "off"):
            image_mode = "on"
        logger.info(f"图片理解模式: {image_mode}")
        input_path = Path(file_path)
        papers_dir = Path(settings["document"]["papers_dir"])
        paper_dir = papers_dir / str(paper_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        converted_pdf_to_clean = None  # 记录 LibreOffice 生成的临时 PDF（结束清理）
        split_parts_to_clean = []     # 记录 pypdf 拆分出的临时子 PDF（结束清理）

        try:
            # ── Step 1: 格式路由（force_pdf 决定是否转 PDF）──
            # - PDF 输入：直接用（force_pdf 对 PDF 无意义）
            # - 非 PDF + force_pdf=True：LibreOffice 转 PDF（旧行为，最稳）
            # - 非 PDF + force_pdf=False：原格式直接给 MinerU（图片少的 Word/PPT 更优）
            if is_pdf(file_path):
                pdf_path = file_path
                mineru_input = file_path        # MinerU 直接吃原 PDF
                page_count = get_pdf_page_count(pdf_path)
                logger.info(f"输入是 PDF（{page_count} 页），跳过格式转换: {input_path.name}")
            elif force_pdf:
                logger.info(f"非 PDF（{input_path.suffix}），按 force_pdf 用 LibreOffice 转 PDF...")
                pdf_path = convert_to_pdf(file_path, output_dir=str(paper_dir))
                converted_pdf_to_clean = pdf_path
                mineru_input = pdf_path
                page_count = get_pdf_page_count(pdf_path)
            else:
                # 原格式直解：不转 PDF，MinerU 直接吃 DOCX/PPTX
                logger.info(f"非 PDF（{input_path.suffix}），force_pdf=False，原格式直解 MinerU")
                pdf_path = None
                mineru_input = file_path
                # 页数用 PDF 的概念不适用，记 0（DOCX 无固定页数）
                page_count = 0

            # ── Step 2: PDF 拆分（>200 页）—— 仅 PDF 走这步 ──
            if pdf_path is not None:
                sub_files = split_pdf(pdf_path, max_pages=settings["document"]["max_pdf_pages"])
                # 拆分产物（sub_files 里除原始 pdf_path 外的都是临时文件，结束后清理）
                split_parts_to_clean = [p for p in sub_files if p != pdf_path]
                logger.info(f"PDF {input_path.name}（{page_count} 页）→ {len(sub_files)} 份")
            else:
                # 原格式直解：MinerU 自己处理整文件（DOCX/PPTX 无 200 页限制）
                sub_files = [mineru_input]
                logger.info(f"原格式直解: {input_path.name}（DOCX/PPTX 不拆分）")

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

            # ── Step 5: 图表多模态理解（qwen3.7-plus 视觉能力）──
            # 顺序调整：先理解图片，再把【图片描述补进 markdown】，
            # 这样保存的 .md 和全文任务(笔记/PPT/summary)都能看到图片内容，
            # 不再只靠 ES 里的 image_caption 片段。
            #
            # 二态开关（image_mode），不再按数量截断：
            # - on  ：处理【全部】图片（论文/报告，架构图/实验图有价值）
            # - off ：完全跳过（教材/数学书，公式截图描述无意义，省 30+ 分钟）
            # 为什么不做"上限截断"：前 N 张可能恰好是最不相关的，截断只增噪声不提质。
            image_captions = []
            if image_mode == "off":
                logger.info("图片理解已关闭（image_mode=off），跳过")
            else:
                try:
                    from app.core.multimodal import multimodal_manager
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    # 收集所有 mineru_* 目录下的全部图片（不截断）
                    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                    all_images = []
                    for md_dir in sorted(paper_dir.glob("mineru_*")):
                        images_subdir = md_dir / "images"
                        if images_subdir.exists():
                            all_images.extend(sorted(
                                p for p in images_subdir.rglob("*")
                                if p.suffix.lower() in image_exts
                            ))
                    logger.info(
                        f"待理解图片: {len(all_images)} 张（全开模式，5 并发，"
                        f"预计 {len(all_images) * 33 // 5 // 60 + 1} 分钟）"
                    )
                    if all_images:
                        with ThreadPoolExecutor(max_workers=5) as ex:
                            fut_to_img = {
                                ex.submit(multimodal_manager.understand_image, str(p)): p
                                for p in all_images
                            }
                            done = 0
                            total = len(fut_to_img)
                            for fut in as_completed(fut_to_img):
                                img = fut_to_img[fut]
                                done += 1
                                try:
                                    desc = fut.result()
                                    if desc:
                                        image_captions.append({
                                            "image_path": str(img),
                                            "description": desc
                                        })
                                    if done % 5 == 0 or done == total:
                                        logger.info(f"图片理解进度: {done}/{total}")
                                except Exception as e:
                                    logger.warning(f"图片理解失败 {img.name}: {e}")
                        # 稳定排序
                        order = {str(p): i for i, p in enumerate(all_images)}
                        image_captions.sort(key=lambda r: order.get(r["image_path"], 0))
                    logger.info(f"图片理解完成: {len(image_captions)} 张")
                except Exception as e:
                    logger.warning(f"图片理解失败（不影响主流程）: {e}")

            # ── Step 5.5: 图片描述存为【独立文件】，不污染正文 ──
            # 【重要变更】原来把图片描述 append 到正文末尾，导致超长文档（如 111 万字
            # 数学书）的后 70% 全是英文图片描述块，严重污染正文语义、割裂结构。
            # 现在改为：正文 markdown 保持纯净（只有 MinerU 原始解析 + 排版清洗），
            # 图片描述单独存 {stem}_images.md，LLM 读全文时看到的是干净正文。
            # 图片描述仍会作为独立 chunk 进 ES（Step 7），检索能力不丢。
            if image_captions:
                caption_lines = [
                    f"# {input_path.stem} - 图表说明\n",
                    "> 以下由 qwen3.7-plus 视觉理解生成，供检索参考。\n",
                ]
                for ic in image_captions:
                    img_name = Path(ic["image_path"]).name
                    # 用 [图] 文字标记代替 emoji 📊（emoji 在部分终端/编码下乱码成 ??）
                    caption_lines.append(f"**[图] {img_name}**：{ic['description']}\n")
                images_md_path = paper_dir / f"{input_path.stem}_images.md"
                images_md_path.write_text("\n".join(caption_lines), encoding="utf-8")
                logger.info(f"图片描述存为独立文件: {images_md_path.name}（{len(image_captions)} 条，不污染正文）")

            # ── Step 5.6: 排版清洗（让 markdown 对 LLM 更友好）──
            # MinerU 输出常带：被错误断行的句子、连续空行、纯数字页码行等。
            # 这些对检索和通读都是噪声，清洗后显著提升 LLM 可读性。
            # 注意：清洗在【补图片描述之后】，避免破坏图片块；只改正文排版。
            before_len = len(markdown_content)
            markdown_content = _clean_markdown(markdown_content)
            logger.info(f"排版清洗: {before_len} → {len(markdown_content)} 字符")

            # ── Step 6: 保存纯净正文 Markdown（不含图片描述，描述在 _images.md）──
            merged_md_path = paper_dir / f"{input_path.stem}.md"
            merged_md_path.write_text(markdown_content, encoding="utf-8")

            # ── Step 6.5: 清理中间 part md（保留 images 目录 + _images.md）──
            # 正文 md（{stem}.md）+ 图片描述 md（{stem}_images.md）都在顶层，
            # get_full_markdown 只读 {stem}.md（取排序后第一个，正文先于 _images）。
            # mineru_* 子目录里的 part md 是冗余的，删除省空间。images/ 必须保留。
            cleaned_md = 0
            for md_dir in paper_dir.glob("mineru_*"):
                for part_md in md_dir.glob("*.md"):
                    try:
                        part_md.unlink()
                        cleaned_md += 1
                    except Exception as e:
                        logger.warning(f"删除中间文件失败 {part_md}: {e}")
            if cleaned_md:
                logger.info(f"已清理 {cleaned_md} 个中间 part md（images 目录已保留）")

            # ── Step 7: 分块（文本 + 图片描述独立 chunk）──
            chunk_size = settings["elasticsearch"]["chunk_size"]
            chunk_overlap = settings["elasticsearch"]["chunk_overlap"]

            chunks = split_text_with_overlap(
                text=markdown_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_page=-1,
                chunk_type="text"
            )

            # 图片描述也作为独立 chunk 追加（chunk_type=image_caption），便于检索命中
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
        """合并多个 Markdown 文件（按子文件顺序）

        同时修正图片相对路径：原 mineru 输出里写 `images/xxx.jpg`，
        合并后文件存在上级目录，需改成 `mineru_xxx/images/xxx.jpg` 才能正确渲染。
        """
        import re
        parts = []
        for i, md_path in enumerate(md_files, 1):
            content = md_path.read_text(encoding="utf-8")
            # 修正图片路径：images/ → <mineru目录名>/images/
            mineru_dir_name = md_path.parent.name  # 如 mineru_part1
            content = re.sub(
                r"!\[([^\]]*)\]\(images/",
                lambda m, d=mineru_dir_name: f"![{m.group(1)}]({d}/images/",
                content,
            )
            parts.append(
                f"<!-- === 第 {i} 部分（来源: {md_path.name}）=== -->\n\n{content}\n"
            )
        return "\n".join(parts)

    def get_full_markdown(self, paper_id: int) -> str:
        """读取某篇论文处理后的完整 Markdown（MinerU 解析 + 合并后的 .md）。

        用于复习笔记 / PPT 这类「需要通读全文」的任务 —— 它们不能只看 RAG
        检索出来的 top-5 片段，否则只见树木不见森林。

        文件位置：data/papers/<paper_id>/<stem>.md（process_document 时保存）。

        Returns:
            完整 Markdown 文本。找不到时抛 DocumentProcessError（避免静默失败）。
        """
        papers_dir = Path(settings["document"]["papers_dir"])
        paper_dir = papers_dir / str(paper_id)
        if not paper_dir.exists():
            raise DocumentProcessError(
                f"论文目录不存在: {paper_dir}（paper_id={paper_id} 可能尚未解析完成）"
            )
        # 取正文 .md：排除 _images.md（图片描述独立文件，不喂给全文通读任务）
        # 否则 LLM 读全文会看到一堆英文图片描述，污染正文语义
        md_files = sorted(
            f for f in paper_dir.glob("*.md")
            if not f.name.endswith("_images.md")
        )
        if not md_files:
            raise DocumentProcessError(
                f"论文 {paper_id} 目录下没有正文 Markdown 文件: {paper_dir}"
            )
        return md_files[0].read_text(encoding="utf-8")


# 全局实例
document_service = DocumentService()

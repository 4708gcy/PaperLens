"""文档管理路由"""
import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Query
from app.schemas import APIResponse
from app.services.document_service import document_service
from app.core.rag_engine import rag_engine
from app.core.pdf_splitter import count_file_images
from app.models.orm import Database, Paper
from app.config import settings
from app.exceptions import NotFoundError
from app.logger import logger

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# 智能判断的图片数阈值：非 PDF 且图片数 < 此值 → 原格式直解；否则 → 转 PDF
# 理由：图片少的 Word/PPT 文本层完整，MinerU 直解质量好且省一次转换；
#       图片多（或扫描件转的 Word）转 PDF 更稳，版面识别更准。
DEFAULT_IMAGE_THRESHOLD = 10


def _decide_force_pdf(file_path: str, user_choice: Optional[bool],
                      threshold: int = DEFAULT_IMAGE_THRESHOLD) -> bool:
    """格式路由核心：决定非 PDF 文件是否先转 PDF。

    优先级：
    1. 用户显式选择（force_pdf=true/false）→ 直接采用
    2. 否则按图片数智能判断：< threshold 原格式直解，>= threshold 转 PDF
    3. 数不准（-1）→ 保守转 PDF（最稳）
    """
    if user_choice is not None:
        return user_choice
    img_count = count_file_images(file_path)
    if img_count < 0:
        logger.info(f"图片数无法判断（{file_path}），保守转 PDF")
        return True
    decision = img_count >= threshold
    logger.info(f"格式路由：{Path(file_path).name} 有 {img_count} 张图 "
                f"(阈值 {threshold}) → {'转PDF' if decision else '原格式直解'}")
    return decision


@router.post("/upload", response_model=APIResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    force_pdf: Optional[bool] = Query(None, description="True=强制转PDF；False=原格式直解；不传=智能判断"),
    image_mode: str = Query("on", description="图片理解开关：on=处理全部图片；off=跳过（教材/数学书）"),
):
    """
    上传论文 PDF（或其他格式），后台异步处理：MinerU → 分块 → 索引

    为什么用 BackgroundTasks 而不是同步？
    — MinerU 转换 + 向量化可能 1-3 分钟
    — 先保存文件即刻返回 paper_id，后台异步处理，避免 HTTP 超时

    格式路由（force_pdf 参数）：
    — PDF 输入：参数无意义，直接处理
    — 非 PDF + force_pdf=True：LibreOffice 转 PDF（最稳）
    — 非 PDF + force_pdf=False：原格式直解 MinerU（图片少的 Word/PPT 推荐）
    — 非 PDF + force_pdf=None：按图片数智能判断

    图片理解（image_mode 参数）：
    — "on"（默认）：处理全部图片，描述补进全文+ES（论文/报告适用）
    — "off"：完全跳过图片理解（教材/数学书适用，省 30+ 分钟）
    """
    # 校验文件类型
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in document_service.SUPPORTED_EXTENSIONS:
        return APIResponse(
            code=400,
            msg=f"不支持的格式「{ext}」，支持：{', '.join(sorted(document_service.SUPPORTED_EXTENSIONS))}"
        )

    # 先写数据库拿 paper_id（status=processing），再保存文件
    # 顺序原因：若文件保存失败，可回滚数据库记录，避免孤儿文件
    db = Database.get_session()
    try:
        paper = Paper(
            title=Path(file.filename).stem,
            file_path="",  # 占位，保存文件后回填
            status="processing"
        )
        db.add(paper)
        db.commit()
        db.refresh(paper)
        paper_id = paper.paper_id
    finally:
        db.close()

    # 保存上传文件（失败则回滚数据库记录）
    upload_dir = Path(settings["document"]["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        # 文件保存失败：回滚数据库记录，避免出现无文件的 paper 记录
        logger.error(f"文件保存失败，回滚 paper_id={paper_id}: {e}")
        db = Database.get_session()
        try:
            bad = db.query(Paper).filter(Paper.paper_id == paper_id).first()
            if bad:
                db.delete(bad)
                db.commit()
        finally:
            db.close()
        return APIResponse(code=500, msg=f"文件保存失败：{str(e)}")

    # 回填真实文件路径
    db = Database.get_session()
    try:
        paper_obj = db.query(Paper).filter(Paper.paper_id == paper_id).first()
        if paper_obj:
            paper_obj.file_path = str(file_path)
            db.commit()
    finally:
        db.close()

    logger.info(f"文件已保存: {file_path}, paper_id={paper_id}")

    # 智能判断格式路由（PDF 无所谓；非 PDF 在此决策）
    is_pdf_input = ext == ".pdf"
    if is_pdf_input:
        decided_force_pdf = True   # PDF 本就是 PDF，传 True 不影响（process_document 里 PDF 直接用）
        scan_warning = ""
        # 对 PDF 做一次扫描件检测，仅作日志提示（不拦截）
        try:
            img_n = count_file_images(str(file_path))
            if img_n > 0:
                pages = 0
                try:
                    from app.core.pdf_splitter import get_pdf_page_count
                    pages = get_pdf_page_count(str(file_path))
                except Exception:
                    pass
                if pages > 0 and img_n >= pages * 0.8:
                    scan_warning = (f"⚠️ 该 PDF 内嵌 {img_n} 张图 / {pages} 页，"
                                    f"疑似扫描件。如解析效果差，建议用 WPS 转为「可编辑 PDF」或 DOCX 后重新上传。")
                    logger.warning(scan_warning)
        except Exception:
            pass
    else:
        decided_force_pdf = _decide_force_pdf(str(file_path), force_pdf)
        scan_warning = ""

    # 规范化 image_mode（防非法值；None/异常都回落到 on）
    decided_image_mode = (image_mode or "on").lower()
    if decided_image_mode not in ("on", "off"):
        decided_image_mode = "on"

    # 后台异步处理
    def _process():
        db2 = Database.get_session()
        try:
            result = document_service.process_document(
                file_path=str(file_path),
                paper_id=paper_id,
                title=Path(file.filename).stem,
                force_pdf=decided_force_pdf,
                image_mode=decided_image_mode,
            )
            paper_obj = db2.query(Paper).filter(Paper.paper_id == paper_id).first()
            if paper_obj:
                paper_obj.status = "indexed"
                paper_obj.markdown_path = result["markdown_path"]
                paper_obj.page_count = result.get("page_count", 0)
                paper_obj.chunk_count = result["total_chunks"]
                db2.commit()
            logger.info(f"paper_id={paper_id} 索引完成")
        except Exception as e:
            logger.error(f"paper_id={paper_id} 处理失败: {e}", exc_info=True)
            paper_obj = db2.query(Paper).filter(Paper.paper_id == paper_id).first()
            if paper_obj:
                paper_obj.status = "failed"
                db2.commit()
        finally:
            db2.close()

    background_tasks.add_task(_process)

    return APIResponse(
        msg="文件上传成功，后台处理中（MinerU + 索引约 1-3 分钟）",
        data={
            "paper_id": paper_id,
            "filename": file.filename,
            "force_pdf": decided_force_pdf,
            "image_mode": decided_image_mode,
            "scan_warning": scan_warning,
        }
    )


@router.get("/list", response_model=APIResponse)
async def list_documents():
    """列出所有论文"""
    db = Database.get_session()
    try:
        papers = db.query(Paper).order_by(Paper.create_dt.desc()).all()
        return APIResponse(data=[
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "status": p.status,
                "page_count": p.page_count,
                "chunk_count": p.chunk_count,
                "create_dt": p.create_dt.isoformat() if p.create_dt else None
            }
            for p in papers
        ])
    finally:
        db.close()


@router.get("/{paper_id}/markdown", response_model=APIResponse)
async def get_markdown(paper_id: int):
    """获取某篇论文的完整 Markdown（MinerU 解析结果）。

    用于「查看原文」、以及前端拼装复习笔记/PPT 的全文上下文。
    """
    db = Database.get_session()
    try:
        paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
        if not paper:
            raise NotFoundError(f"论文 {paper_id} 不存在")
        title = paper.title
    finally:
        db.close()
    try:
        content = document_service.get_full_markdown(paper_id)
    except Exception as e:
        return APIResponse(code=404, msg=f"读取全文失败：{e}")
    return APIResponse(data={"paper_id": paper_id, "title": title, "markdown": content})


@router.delete("/{paper_id}", response_model=APIResponse)
async def delete_document(paper_id: int):
    """删除论文（同步删除 ES 索引）"""
    db = Database.get_session()
    try:
        paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
        if not paper:
            raise NotFoundError(f"论文 {paper_id} 不存在")

        # 1. 删除 ES 索引
        rag_engine.delete_index(paper_id)
        # 2. 删除本地文件
        if paper.file_path and os.path.exists(paper.file_path):
            os.remove(paper.file_path)
        papers_dir = Path(settings["document"]["papers_dir"]) / str(paper_id)
        if papers_dir.exists():
            shutil.rmtree(papers_dir)
        # 3. 删除数据库记录
        db.delete(paper)
        db.commit()

        return APIResponse(msg=f"论文 {paper_id} 已删除", data={"paper_id": paper_id})
    finally:
        db.close()

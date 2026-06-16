"""文档管理路由"""
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.schemas import APIResponse
from app.services.document_service import document_service
from app.core.rag_engine import rag_engine
from app.models.orm import Database, Paper
from app.config import settings
from app.exceptions import NotFoundError
from app.logger import logger

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload", response_model=APIResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    上传论文 PDF（或其他格式），后台异步处理：MinerU → 分块 → 索引

    为什么用 BackgroundTasks 而不是同步？
    — MinerU 转换 + 向量化可能 1-3 分钟
    — 先保存文件即刻返回 paper_id，后台异步处理，避免 HTTP 超时
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

    # 后台异步处理
    def _process():
        db2 = Database.get_session()
        try:
            result = document_service.process_document(
                file_path=str(file_path),
                paper_id=paper_id,
                title=Path(file.filename).stem
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
        data={"paper_id": paper_id, "filename": file.filename}
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

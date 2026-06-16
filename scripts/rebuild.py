"""重建所有已入库文档：用新管线（图片描述补进 markdown + 路径修正 + 全文优先）重跑。

对每篇已上传的文档：
1. 从数据库读 file_path（原始上传文件）
2. 重新跑 document_service.process_document（MinerU + MiMo + 新的图片补全 + ES 索引）
   - 不删旧的 ES 索引：process_document 内部会先删再建（见 rag_engine.index_chunks）
3. 更新数据库状态

用法（必须在 ocr 环境）：
    conda run -n ocr python scripts/rebuild.py            # 重建全部
    conda run -n ocr python scripts/rebuild.py --paper-ids 1 3   # 只重建指定 id
    conda run -n ocr python scripts/rebuild.py --dry-run         # 只打印不执行

注意：会重新调用 MinerU（每篇几十秒~几分钟）和 MiMo 图片理解（每张几秒）。
"""
import argparse
import os
import sys
from pathlib import Path

# 让脚本能 import app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.models.orm import Database, Paper
from app.services.document_service import document_service
from app.core.rag_engine import rag_engine
from app.logger import logger


def main():
    parser = argparse.ArgumentParser(description="重建所有已入库文档（新管线）")
    parser.add_argument("--paper-ids", nargs="*", type=int, default=None,
                        help="只重建指定 paper_id（默认全部）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出将重建的文档，不执行")
    args = parser.parse_args()

    # 初始化数据库
    Database.init(settings["document"]["db_path"])
    session = Database.get_session()
    try:
        query = session.query(Paper)
        if args.paper_ids:
            query = query.filter(Paper.paper_id.in_(args.paper_ids))
        papers = query.order_by(Paper.paper_id).all()
    finally:
        session.close()

    if not papers:
        print("没有找到任何文档。")
        return

    print(f"共 {len(papers)} 篇文档待处理：")
    for p in papers:
        print(f"  [{p.paper_id}] {p.title}  (status={p.status}, file={p.file_path})")

    if args.dry_run:
        print("\n--dry-run：仅列出，不执行。")
        return

    print("\n开始重建（会重新跑 MinerU + MiMo，请耐心等待）...\n")
    success, failed = 0, 0
    for p in papers:
        paper_id = p.paper_id
        file_path = p.file_path
        print(f"\n===== [{paper_id}] {p.title} =====")
        if not file_path or not os.path.exists(file_path):
            print(f"  ✗ 原始文件不存在: {file_path}，跳过")
            failed += 1
            continue
        try:
            # 先删旧 ES 索引，避免重复堆积（create_index 不会自动清空）
            try:
                rag_engine.delete_index(paper_id)
                print(f"  · 已清理旧 ES 索引")
            except Exception as de:
                print(f"  · 清理旧索引跳过（可能本就没有）: {de}")

            result = document_service.process_document(
                file_path=file_path,
                paper_id=paper_id,
                title=p.title,
            )
            # 更新数据库
            session = Database.get_session()
            try:
                paper_obj = session.query(Paper).filter(Paper.paper_id == paper_id).first()
                if paper_obj:
                    paper_obj.status = "indexed"
                    paper_obj.markdown_path = result["markdown_path"]
                    paper_obj.page_count = result.get("page_count", 0)
                    paper_obj.chunk_count = result["total_chunks"]
                    session.commit()
            finally:
                session.close()
            print(f"  ✓ 完成：{result['total_chunks']} chunks，markdown={result['markdown_path']}")
            success += 1
        except Exception as e:
            print(f"  ✗ 失败：{e}")
            logger.error(f"重建 paper_id={paper_id} 失败: {e}", exc_info=True)
            # 标记失败
            session = Database.get_session()
            try:
                paper_obj = session.query(Paper).filter(Paper.paper_id == paper_id).first()
                if paper_obj:
                    paper_obj.status = "failed"
                    session.commit()
            finally:
                session.close()
            failed += 1

    print(f"\n========== 重建完成：成功 {success}，失败 {failed} ==========")
    if failed:
        print("失败的文档请查看上方日志，可能是 MinerU/文件问题。")


if __name__ == "__main__":
    main()

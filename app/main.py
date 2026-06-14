"""FastAPI 应用入口"""
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.models.orm import Database
from app.routers import documents, chat
from app.exceptions import register_exception_handlers
from app.logger import logger


def _check_environment():
    """
    启动时检查关键依赖（mineru-open-api / soffice）。

    注意：只做 warning 不阻塞启动。
    原因：mineru-open-api 装在 conda 环境的 Scripts 目录，
    若用 python.exe 全路径启动（而非 conda activate），PATH 中可能找不到。
    但 subprocess 调用时若文档解析真正失败，会有明确报错。
    """
    missing = []
    if not shutil.which("mineru-open-api"):
        missing.append("mineru-open-api")
    if not shutil.which("soffice"):
        missing.append("soffice")
    if missing:
        logger.warning(
            f"⚠️ 环境检查：以下命令不在 PATH：{missing}。"
            f"若文档解析时失败，请确认后端进程在 conda activate ocr 环境启动。"
        )
    else:
        logger.info("✓ 环境检查通过：mineru-open-api + soffice 均可用")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化，关闭时清理"""
    # 环境自检
    _check_environment()
    logger.info("✓ 环境检查通过：mineru-open-api + soffice 均可用")

    # 初始化数据库
    Database.init(settings["document"]["db_path"])
    logger.info(f"数据库已初始化: {settings['document']['db_path']}")

    # 预加载 Embedding 模型（可选：失败不阻塞启动，延迟到首次检索）
    try:
        from app.core.embedding import embedding_manager
        _ = embedding_manager.dimension
        logger.info(f"Embedding 模型已加载，维度: {_}")
    except Exception as e:
        logger.warning(
            f"Embedding 模型预加载失败（不阻塞启动，首次检索时会重试）: {e}"
        )

    yield

    logger.info("应用关闭")


app = FastAPI(
    title=settings["app"]["name"],
    version=settings["app"]["version"],
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册全局异常
register_exception_handlers(app)

# 注册路由
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings["app"]["version"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings["app"]["host"],
        port=settings["app"]["port"],
        reload=settings["app"]["debug"]
    )

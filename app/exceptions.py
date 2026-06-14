"""全局异常处理 —— 这次写好，别再 except:pass"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.logger import logger


class PaperLensError(Exception):
    """项目自定义异常基类"""
    def __init__(self, msg: str, code: int = 500):
        self.msg = msg
        self.code = code
        super().__init__(msg)


class NotFoundError(PaperLensError):
    """资源不存在"""
    def __init__(self, msg: str = "资源不存在"):
        super().__init__(msg, code=404)


class LLMError(PaperLensError):
    """LLM 调用异常"""
    def __init__(self, msg: str = "LLM 服务调用失败"):
        super().__init__(msg, code=502)


class DocumentProcessError(PaperLensError):
    """文档处理异常"""
    def __init__(self, msg: str = "文档处理失败"):
        super().__init__(msg, code=422)


def register_exception_handlers(app: FastAPI):
    """注册全局异常处理器"""

    @app.exception_handler(PaperLensError)
    async def paperlens_handler(request: Request, exc: PaperLensError):
        logger.error(f"业务异常 [{exc.code}]: {exc.msg}")
        return JSONResponse(
            status_code=exc.code,
            content={"code": exc.code, "msg": exc.msg, "data": None}
        )

    @app.exception_handler(Exception)
    async def global_handler(request: Request, exc: Exception):
        logger.error(f"未捕获异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "msg": f"服务器内部错误: {str(exc)}", "data": None}
        )

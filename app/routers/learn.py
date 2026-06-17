"""学习助手 · 分步向导相关接口

当前提供：
- POST /learn/outline  生成笔记/PPT 的大纲（用 fast 模型 + 全文，非流式）
- POST /learn/slides/export  把 Marp Markdown 导出为 PPTX（调 marp-cli）
"""
import os
import tempfile
import subprocess
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.schemas import APIResponse, OutlineRequest
from app.agents.graph import _make_llm
from app.agents.graph import _load_full_markdown
from app.agents.prompts import LEARN_NOTES_OUTLINE_SYSTEM, LEARN_SLIDES_OUTLINE_SYSTEM
from app.config import settings
from app.logger import logger

router = APIRouter(prefix="/api/v1/learn", tags=["learn"])


def _get_outline_llm():
    """现场创建一个【强制关 thinking】的快速 LLM，用于大纲生成。

    为什么不复用全局 fast_llm？
    - 早期版本 fast_llm 继承了全局 enable_thinking=True，导致大纲生成触发
      thinking 模式的 98304 token 输入上限（全书 109 万字必超）报 400。
    - 全局变量在 uvicorn --reload 时可能不重新赋值（模块级语句偶尔不重跑），
      现场创建彻底绕开这个隐患。
    - 大纲是结构化任务，不需要深度推理，关 thinking 还能省 45s 首 token 延迟。
    """
    return _make_llm(
        temperature=0,
        model=settings["llm"]["fast_model"],
        enable_thinking=False,
    )


class SlidesExportRequest(BaseModel):
    """PPT 导出请求"""
    markdown: str
    format: str = "pptx"  # pptx / pdf / html


@router.post("/outline", response_model=APIResponse)
async def generate_outline(request: OutlineRequest):
    """分步向导第 1 步：基于全文生成笔记/PPT 大纲。

    用 fast 模型（qwen-turbo）+ 全文，非流式返回大纲文本。
    前端拿到后让用户编辑，再带着确认后的大纲调 /chat/stream 生成成品。
    """
    mode = request.mode
    if mode not in ("notes", "slides"):
        return APIResponse(code=400, msg=f"mode 必须是 notes 或 slides，收到：{mode}")

    # 选大纲模板
    template = LEARN_NOTES_OUTLINE_SYSTEM if mode == "notes" else LEARN_SLIDES_OUTLINE_SYSTEM

    try:
        # 大纲用关 thinking 的 LLM（_get_outline_llm），输入上限回到 1M token，
        # 所以全文预算用 60 万字（thinking=False 的预算），而非默认的 18 万。
        # 否则超长书只会看到头尾 18 万字，中间内容缺失，大纲质量打折。
        full_text = _load_full_markdown(request.paper_ids, max_chars=600000)
    except Exception as e:
        logger.error(f"读取全文失败: {e}", exc_info=True)
        return APIResponse(code=500, msg=f"读取全文失败：{e}")

    placeholders = {
        "context": full_text,
        "focus": request.focus or "",
    }
    if mode == "notes":
        placeholders["detail_level"] = request.detail_level or "标准"
    else:
        placeholders["page_count"] = str(request.page_count or 10)

    try:
        prompt = template.format(**placeholders)
    except KeyError as e:
        return APIResponse(code=500, msg=f"模板缺占位符：{e}")

    try:
        # 现场创建关 thinking 的 LLM（绕开全局 fast_llm 可能未刷新的隐患）
        outline_llm = _get_outline_llm()
        resp = outline_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="请输出大纲。"),
        ])
        outline = resp.content.strip()
    except Exception as e:
        logger.error(f"生成大纲失败: {e}", exc_info=True)
        return APIResponse(code=500, msg=f"生成大纲失败：{e}")

    logger.info(f"大纲生成完成（mode={mode}, 长度={len(outline)}）")
    return APIResponse(data={"outline": outline, "mode": mode})


@router.post("/slides/export")
async def export_slides(request: SlidesExportRequest):
    """把 Marp Markdown 导出为 PPTX（调 marp-cli，需 Node.js + npx）。

    返回二进制文件流（PPTX）。
    """
    fmt = request.format if request.format in ("pptx", "pdf", "html") else "pptx"
    md = (request.markdown or "").strip()
    if not md:
        return APIResponse(code=400, msg="markdown 内容为空")

    # 写临时 .md 文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(md)
        md_path = f.name
    out_path = md_path.rsplit(".", 1)[0] + f".{fmt}"

    try:
        # 调 marp-cli：npx @marp-team/marp-cli md -> 目标格式
        cmd = [
            "npx", "@marp-team/marp-cli@latest",
            md_path, "-o", out_path,
        ]
        logger.info(f"导出 PPT 执行: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=240,
            shell=True,  # Windows 下 npx 需要 shell=True 才能找到 .cmd
        )
        if result.returncode != 0:
            logger.error(f"marp-cli 失败: {result.stderr[:500]}")
            return APIResponse(
                code=500,
                msg=f"导出失败（marp-cli 错误）：{result.stderr[:300] or result.stdout[:300]}"
            )
        if not os.path.exists(out_path):
            return APIResponse(code=500, msg="导出失败：未生成输出文件")

        with open(out_path, "rb") as f:
            content = f.read()
        logger.info(f"PPT 导出成功：{fmt}, {len(content)} 字节")
        media = {
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "pdf": "application/pdf",
            "html": "text/html",
        }[fmt]
        filename = f"PPT.{fmt}"
        return Response(
            content=content,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except subprocess.TimeoutExpired:
        return APIResponse(code=500, msg="导出超时（marp-cli 首次安装可能较慢，请重试）")
    except FileNotFoundError:
        return APIResponse(code=500, msg="未找到 npx（需要安装 Node.js）")
    except Exception as e:
        logger.error(f"PPT 导出异常: {e}", exc_info=True)
        return APIResponse(code=500, msg=f"导出异常：{e}")
    finally:
        # 清理临时文件
        for p in (md_path, out_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

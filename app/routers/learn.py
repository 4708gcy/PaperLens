"""学习助手 · 分步向导相关接口

当前提供：
- POST /learn/outline  生成笔记/PPT 的大纲（用 fast 模型 + 全文，非流式）
"""
from fastapi import APIRouter
from langchain_core.messages import SystemMessage, HumanMessage

from app.schemas import APIResponse, OutlineRequest
from app.agents.graph import fast_llm
from app.agents.graph import _load_full_markdown
from app.agents.prompts import LEARN_NOTES_OUTLINE_SYSTEM, LEARN_SLIDES_OUTLINE_SYSTEM
from app.logger import logger

router = APIRouter(prefix="/api/v1/learn", tags=["learn"])


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
        full_text = _load_full_markdown(request.paper_ids)
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
        resp = fast_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="请输出大纲。"),
        ])
        outline = resp.content.strip()
    except Exception as e:
        logger.error(f"生成大纲失败: {e}", exc_info=True)
        return APIResponse(code=500, msg=f"生成大纲失败：{e}")

    logger.info(f"大纲生成完成（mode={mode}, 长度={len(outline)}）")
    return APIResponse(data={"outline": outline, "mode": mode})

"""智能体对话路由 —— 普通响应 + SSE 流式响应"""
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from app.schemas import ChatRequest, ChatResponse
from app.agents.graph import study_graph
from app.models.orm import Database, Conversation, Message
from app.exceptions import LLMError
from app.logger import logger

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _save_message(thread_id: str, role: str, content: str, intent: str = ""):
    """保存消息到数据库"""
    db = Database.get_session()
    try:
        conv = db.query(Conversation).filter(Conversation.thread_id == thread_id).first()
        if not conv:
            conv = Conversation(thread_id=thread_id, title=content[:50])
            db.add(conv)
            db.commit()
            db.refresh(conv)
        msg = Message(
            conversation_id=conv.conversation_id,
            role=role, content=content, intent=intent
        )
        db.add(msg)
        db.commit()
    finally:
        db.close()


def _build_input_state(request: ChatRequest) -> dict:
    """从请求构造初始状态"""
    return {
        "messages": [HumanMessage(content=request.message)],
        "paper_ids": request.paper_ids,
        "intent": "",
        "context": "",
        "topic": "",
        "sections": [],
        "outline": [],
        "final_report": "",
        "learn_mode": request.learn_mode or "",
        "learn_outline": request.learn_outline or "",
        "learn_config": request.learn_config or {},
        "retrieval_mode": request.retrieval_mode or "",
        "images": request.images or [],
    }


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    普通对话（一次性返回完整响应）

    流程：
    1. 接收 message + paper_ids + thread_id
    2. 构造初始状态，调 StateGraph
    3. LangGraph 自动 triage → retrieve → agent
    4. 提取 AI 回复返回
    """
    input_state = _build_input_state(request)
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        result = study_graph.invoke(input_state, config)
    except Exception as e:
        logger.error(f"Agent 执行失败: {e}", exc_info=True)
        raise LLMError(f"Agent 处理失败: {str(e)}")

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = ai_messages[-1].content if ai_messages else "抱歉，处理出错了"
    intent = result.get("intent", "unknown")

    # 持久化
    _save_message(request.thread_id, "user", request.message)
    _save_message(request.thread_id, "assistant", reply, intent)

    return ChatResponse(
        data={"reply": reply, "intent": intent, "thread_id": request.thread_id}
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话（SSE 逐 token 返回）

    为什么需要流式？
    — LLM 生成长回复时（如结构化解读），用户等待体验差
    — SSE 逐字推送，用户在 AI "想" 的时候就能开始阅读
    """
    async def generate():
        input_state = _build_input_state(request)
        config = {"configurable": {"thread_id": request.thread_id}}

        current_intent = ""
        full_reply = ""

        try:
            # astream_events 捕获 LangGraph 内部事件
            async for event in study_graph.astream_events(input_state, config, version="v2"):
                kind = event.get("event", "")

                # 捕获 triage 节点的分类结果
                if kind == "on_chain_end" and event.get("name") == "triage":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "intent" in output:
                        current_intent = output["intent"]
                        yield f"data: {json.dumps({'type': 'intent', 'intent': current_intent}, ensure_ascii=False)}\n\n"

                # 捕获 LLM token 流（打字机效果）
                # 只捕获"最终回答阶段"的 LLM 调用：
                # - qa/analyze/general: 对应 agent 节点
                # - synthesize: 只有 assembler 节点是最终输出
                # 过滤掉 triage（意图分类）、planner（大纲生成）、section_worker（章节写作）
                if kind == "on_chat_model_stream":
                    tags = event.get("tags", [])
                    metadata = event.get("metadata", {})
                    node_name = str(metadata.get("langgraph_node", ""))
                    # 这些节点的 LLM 输出不应作为最终 token 推送
                    skip_nodes = {"triage", "planner", "section_worker"}
                    if any(n in tags or n == node_name for n in skip_nodes):
                        continue
                    chunk = event.get("data", {}).get("chunk", {})
                    if hasattr(chunk, "content") and chunk.content:
                        full_reply += chunk.content
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content}, ensure_ascii=False)}\n\n"

            # 流结束
            yield f"data: {json.dumps({'type': 'done', 'intent': current_intent}, ensure_ascii=False)}\n\n"

            # 持久化
            _save_message(request.thread_id, "user", request.message)
            _save_message(request.thread_id, "assistant", full_reply, current_intent)

        except Exception as e:
            logger.error(f"流式 Agent 失败: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'msg': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

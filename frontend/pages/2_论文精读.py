"""论文精读页面 —— 单论文多轮问答 + 流式"""
import time
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import list_documents, chat, chat_stream

st.title("💬 论文精读")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 设置")
    result = list_documents()
    papers = result.get("data", []) if result.get("code") == 200 else []
    indexed_papers = [p for p in papers if p["status"] == "indexed"]

    if not indexed_papers:
        st.warning("请先在「论文管理」上传并等待处理完成")
        st.stop()

    paper_options = {p["paper_id"]: p["title"] for p in indexed_papers}
    selected_id = st.selectbox(
        "选择论文",
        options=list(paper_options.keys()),
        format_func=lambda x: f"[{x}] {paper_options[x][:30]}"
    )
    use_stream = st.toggle("流式输出", value=True, help="逐字显示 AI 回复")

    st.divider()
    # ── 对话管理：新建 / 清空 ──
    # thread_id 默认带时间戳，保证「新建对话」后历史不串、checkpointer 不复用旧上下文
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"paper_{selected_id}_{int(time.time())}"

    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button("🆕 新建对话", use_container_width=True,
                     help="开一个全新对话（清空当前显示，并换一个新的对话上下文，不再记住之前内容）"):
            st.session_state.messages = []
            st.session_state.thread_id = f"paper_{selected_id}_{int(time.time())}"
            st.toast("已开启新对话", icon="🆕")
            st.rerun()
    with col_clear:
        if st.button("🗑️ 清空显示", use_container_width=True,
                     help="只清空当前页面的对话显示，不改对话上下文（下次提问仍可记住上文）"):
            st.session_state.messages = []
            st.toast("已清空显示", icon="🗑️")
            st.rerun()

    st.caption(f"当前对话ID：`{st.session_state.thread_id}`")

    st.divider()
    st.caption("🎯 AI 会自动识别你的意图：")
    st.caption("- 📖 知识问答（基于论文内容回答）")
    st.caption("- 💡 概念解释")
    st.caption("- 🔗 多篇综述（需多篇论文）")

# 聊天状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 切换论文时重置对话（换论文 = 换新对话上下文）
# 必须在显示之前执行 + st.rerun，否则右侧还停留在旧论文的对话上
if st.session_state.get("last_paper_id") != selected_id:
    old = st.session_state.get("last_paper_id")
    st.session_state.last_paper_id = selected_id
    st.session_state.messages = []
    st.session_state.thread_id = f"paper_{selected_id}_{int(time.time())}"
    st.toast(f"已切换到论文 [{selected_id}]，对话已重置（原 {old}）", icon="🔄")
    st.rerun()

# 主区域顶部：显示当前选中论文信息
selected_paper = next((p for p in indexed_papers if p["paper_id"] == selected_id), None)
if selected_paper:
    st.info(
        f"📄 **当前论文**：[{selected_paper['paper_id']}] {selected_paper['title']}　|　"
        f"{selected_paper.get('page_count', 0)} 页　|　"
        f"{selected_paper.get('chunk_count', 0)} 个语义块　|　"
        f"在左侧切换论文"
    )

# 图片上传区（多模态：qwen3.7-plus 可看图）—— 顶层调用，不包 with
import base64

def _encode_image_to_data_url(uploaded) -> str:
    """把 Streamlit UploadedFile 转成 data URL（base64），供 qwen3.7-plus 视觉输入。"""
    raw = uploaded.getvalue()
    ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else "jpeg"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "webp": "webp", "gif": "gif", "bmp": "bmp"}.get(ext, "jpeg")
    b64 = base64.b64encode(raw).decode()
    return f"data:image/{mime};base64,{b64}"

uploaded_images = st.file_uploader(
    "📎 上传图片提问（可选，qwen3.7-plus 会看图）",
    type=["png", "jpg", "jpeg", "webp", "gif", "bmp"],
    accept_multiple_files=True,
    key="read_img_uploader",
)
pending_images = [_encode_image_to_data_url(img) for img in (uploaded_images or [])]

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for img_url in msg.get("images", []):
            st.image(img_url, width=200)
        if msg.get("intent"):
            st.caption(f"🎯 意图：{msg['intent']}")

# 接收输入
if prompt := st.chat_input("输入你的问题...（上方可附图）"):
    st.session_state.messages.append({
        "role": "user", "content": prompt, "images": pending_images
    })
    with st.chat_message("user"):
        st.markdown(prompt)
        for img_url in pending_images:
            st.image(img_url, width=200)

    with st.chat_message("assistant"):
        if use_stream:
            placeholder = st.empty()
            full_response = ""
            detected_intent = ""

            for event in chat_stream(prompt, [selected_id], st.session_state.thread_id,
                                      images=pending_images or None):
                if event.get("type") == "intent":
                    detected_intent = event.get("intent", "")
                elif event.get("type") == "token":
                    full_response += event.get("content", "")
                    placeholder.markdown(full_response + "▌")
                elif event.get("type") == "done":
                    placeholder.markdown(full_response)
                    if event.get("intent"):
                        detected_intent = event.get("intent")
                elif event.get("type") == "error":
                    placeholder.error(f"错误：{event.get('msg')}")

            if detected_intent:
                st.caption(f"🎯 意图：{detected_intent}")
        else:
            with st.spinner("思考中..."):
                result = chat(prompt, [selected_id], st.session_state.thread_id,
                              images=pending_images or None)
            full_response = result.get("data", {}).get("reply", "出错了")
            detected_intent = result.get("data", {}).get("intent", "")
            st.markdown(full_response)
            if detected_intent:
                st.caption(f"🎯 意图：{detected_intent}")

    st.session_state.messages.append({
        "role": "assistant", "content": full_response, "intent": detected_intent
    })

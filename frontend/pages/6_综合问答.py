"""综合问答页面 —— 跨多文档检索问答（ES 检索，区别于论文精读的全文直喂）

适用场景：已上传一批文档（课件/论文），想「在这堆资料里找讲过 X 的地方」、
「对比这几篇关于 Y 的说法」这类跨文档定位问题。

语料范围：可选「全部已上传」或「自选若干篇」。
回答形式：对话式问答（流式，可连续追问）。
"""
import time
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import list_documents, rag_chat_stream

st.title("🔎 综合问答")
st.caption("跨文档检索问答。从大量资料里捞出相关片段来回答 —— 适合「哪些资料讲过 X」这类问题。")

# 侧边栏：语料范围 + 对话管理
with st.sidebar:
    st.header("⚙️ 语料范围")
    result = list_documents()
    papers = result.get("data", []) if result.get("code") == 200 else []
    indexed_papers = [p for p in papers if p["status"] == "indexed"]

    if not indexed_papers:
        st.warning("请先在「论文管理」上传并等待处理完成")
        st.stop()

    all_ids = [p["paper_id"] for p in indexed_papers]
    paper_map = {p["paper_id"]: p["title"] for p in indexed_papers}

    # 语料模式：全部 / 自选
    scope_mode = st.radio(
        "语料范围",
        options=["全部已上传", "自选若干篇"],
        index=0,
        help="「全部」会在所有已处理文档里检索；「自选」只在勾选的文档里检索。"
    )

    if scope_mode == "全部已上传":
        selected_ids = all_ids
        st.success(f"📂 将在全部 {len(all_ids)} 篇文档里检索")
    else:
        chosen = st.multiselect(
            f"选择文档（共 {len(all_ids)} 篇）",
            options=all_ids,
            default=all_ids[:1] if all_ids else [],
            format_func=lambda x: f"[{x}] {paper_map[x][:30]}",
        )
        selected_ids = chosen
        if not selected_ids:
            st.warning("请至少选择 1 篇文档")
        else:
            st.info(f"📂 将在选中的 {len(selected_ids)} 篇文档里检索")

    st.divider()
    # 语料变化检测：切换「全部/自选」或改变勾选文档时，清空旧对话 + 换新 thread_id
    # 否则右侧还显示上一个语料范围的问答，与当前语料不一致
    _scope_key = f"{scope_mode}_{'_'.join(map(str, sorted(selected_ids)))}"
    if st.session_state.get("rag_last_scope_key") != _scope_key:
        st.session_state["rag_last_scope_key"] = _scope_key
        st.session_state.rag_messages = []
        st.session_state.rag_thread_id = f"rag_{int(time.time())}"
        st.rerun()  # 必须重跑，否则旧消息还在屏幕上

    # 对话管理
    if "rag_thread_id" not in st.session_state:
        st.session_state.rag_thread_id = f"rag_{int(time.time())}"

    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button("🆕 新建对话", use_container_width=True,
                     help="开一个全新对话（清空显示 + 换新上下文）"):
            st.session_state.rag_messages = []
            st.session_state.rag_thread_id = f"rag_{int(time.time())}"
            st.toast("已开启新对话", icon="🆕")
            st.rerun()
    with col_clear:
        if st.button("🗑️ 清空显示", use_container_width=True,
                     help="只清空页面显示，不改对话上下文"):
            st.session_state.rag_messages = []
            st.toast("已清空显示", icon="🗑️")
            st.rerun()

    st.caption(f"当前对话ID：`{st.session_state.rag_thread_id}`")
    st.divider()
    st.caption("💡 本页用 ES 检索（跨文档定位），与论文精读的全文直喂不同。")
    st.caption("适合：在多份资料里找/对比某主题。")

# 聊天状态
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

# 图片上传区（多模态：qwen3.7-plus 可看图）
import base64

def _encode_image_to_data_url(uploaded) -> str:
    """把 Streamlit UploadedFile 转成 data URL（base64），供 qwen3.7-plus 视觉输入。"""
    raw = uploaded.getvalue()
    ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else "jpeg"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "webp": "webp", "gif": "gif", "bmp": "bmp"}.get(ext, "jpeg")
    b64 = base64.b64encode(raw).decode()
    return f"data:image/{mime};base64,{b64}"

# 待发送图片（顶层调用，不包 with container，避免 chat_input 嵌套报错）
uploaded_images = st.file_uploader(
    "📎 上传图片提问（可选，qwen3.7-plus 会看图）",
    type=["png", "jpg", "jpeg", "webp", "gif", "bmp"],
    accept_multiple_files=True,
    key="rag_img_uploader",
)
pending_images = [_encode_image_to_data_url(img) for img in (uploaded_images or [])]

# 显示历史
for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 历史里也回显用户当时上传的图
        for img_url in msg.get("images", []):
            st.image(img_url, width=200)

# 接收输入
if prompt := st.chat_input("跨文档提问，例如：哪些资料讲到了注意力机制？（上方可附图）"):
    if not selected_ids:
        st.warning("请先在左侧选择语料范围。")
        st.stop()
    st.session_state.rag_messages.append({
        "role": "user", "content": prompt, "images": pending_images
    })
    with st.chat_message("user"):
        st.markdown(prompt)
        for img_url in pending_images:
            st.image(img_url, width=200)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for event in rag_chat_stream(
            prompt, selected_ids, st.session_state.rag_thread_id,
            images=pending_images or None,
        ):
            t = event.get("type")
            if t == "token":
                full_response += event.get("content", "")
                placeholder.markdown(full_response + "▌")
            elif t == "done":
                placeholder.markdown(full_response)
            elif t == "error":
                placeholder.error(f"错误：{event.get('msg')}")

    st.session_state.rag_messages.append({
        "role": "assistant", "content": full_response
    })

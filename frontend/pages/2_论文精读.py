"""论文精读页面 —— 单论文多轮问答 + 流式"""
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
    st.caption("🎯 AI 会自动识别你的意图：")
    st.caption("- 📖 知识问答（基于论文内容回答）")
    st.caption("- 💡 概念解释")
    st.caption("- 🔗 多篇综述（需多篇论文）")

# 聊天状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"paper_{selected_id}"

# 切换论文时重置对话
if st.session_state.get("last_paper_id") != selected_id:
    st.session_state.last_paper_id = selected_id
    st.session_state.messages = []
    st.session_state.thread_id = f"paper_{selected_id}"

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("intent"):
            st.caption(f"🎯 意图：{msg['intent']}")

# 接收输入
if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if use_stream:
            placeholder = st.empty()
            full_response = ""
            detected_intent = ""

            for event in chat_stream(prompt, [selected_id], st.session_state.thread_id):
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
                result = chat(prompt, [selected_id], st.session_state.thread_id)
            full_response = result.get("data", {}).get("reply", "出错了")
            detected_intent = result.get("data", {}).get("intent", "")
            st.markdown(full_response)
            if detected_intent:
                st.caption(f"🎯 意图：{detected_intent}")

    st.session_state.messages.append({
        "role": "assistant", "content": full_response, "intent": detected_intent
    })

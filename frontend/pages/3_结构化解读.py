"""结构化解读页面"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import list_documents, analyze

st.title("🔍 一键结构化解读")
st.markdown("选择一篇论文，AI 自动生成 5 段式解读（背景 / 方法 / 贡献 / 实验 / 局限）。")

result = list_documents()
papers = result.get("data", []) if result.get("code") == 200 else []
indexed_papers = [p for p in papers if p["status"] == "indexed"]

if not indexed_papers:
    st.warning("请先上传并等待论文处理完成")
    st.stop()

paper_options = {p["paper_id"]: p["title"] for p in indexed_papers}
selected_id = st.selectbox(
    "选择论文",
    options=list(paper_options.keys()),
    format_func=lambda x: f"[{x}] {paper_options[x][:30]}"
)

if st.button("🚀 生成结构化解读", type="primary"):
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        for event in analyze(selected_id, thread_id=f"analyze_{selected_id}"):
            if event.get("type") == "intent":
                st.caption(f"🎯 意图：{event.get('intent')}")
            elif event.get("type") == "token":
                full += event.get("content", "")
                placeholder.markdown(full + "▌")
            elif event.get("type") == "done":
                placeholder.markdown(full)
            elif event.get("type") == "error":
                st.error(f"错误：{event.get('msg')}")
                break

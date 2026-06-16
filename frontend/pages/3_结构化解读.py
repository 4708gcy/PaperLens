"""结构化解读页面"""
import time
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import list_documents, analyze

st.title("🔍 一键结构化解读")
st.markdown("选择一篇论文，AI 自动生成 5 段式解读（背景 / 方法 / 贡献 / 实验 / 局限）。")

result = list_documents()
papers = result.get("data", []) if result.get("code") == 200 else []
indexed_papers = [p for p in papers if p.get("status") == "indexed"]

if not indexed_papers:
    st.warning("请先上传并等待论文处理完成")
    st.stop()

paper_options = {p["paper_id"]: p["title"] for p in indexed_papers}
selected_id = st.selectbox(
    "选择论文",
    options=list(paper_options.keys()),
    format_func=lambda x: f"[{x}] {paper_options[x][:30]}"
)

# 切换论文时清掉旧结果（避免显示别篇的解读）
if st.session_state.get("analyze_last_id") != selected_id:
    st.session_state["analyze_last_id"] = selected_id
    st.session_state.pop("analyze_result", None)
    st.session_state.pop("analyze_intent", None)

if st.button("🚀 生成结构化解读", type="primary"):
    placeholder_block = st.chat_message("assistant")
    placeholder = placeholder_block.empty()
    full = ""
    intent = ""
    try:
        for event in analyze(selected_id, thread_id=f"analyze_{selected_id}_{int(time.time())}"):
            if event.get("type") == "intent":
                intent = event.get("intent", "")
            elif event.get("type") == "token":
                full += event.get("content", "")
                placeholder.markdown(full + "▌")
            elif event.get("type") == "done":
                placeholder.markdown(full)
            elif event.get("type") == "error":
                st.error(f"错误：{event.get('msg')}")
                st.stop()
    except Exception as e:
        st.error(f"请求失败（后端可能未启动或超时）：{e}")
        st.stop()
    # 持久化结果，重跑脚本（点任何控件）时不会消失
    if full:
        st.session_state["analyze_result"] = full
        st.session_state["analyze_intent"] = intent
        st.rerun()  # 清掉按钮按下态，用持久化结果重新渲染

# 复用已持久化的结果
if "analyze_result" in st.session_state:
    with st.chat_message("assistant"):
        intent = st.session_state.get("analyze_intent", "")
        if intent:
            st.caption(f"🎯 意图：{intent}")
        st.markdown(st.session_state["analyze_result"])
    st.divider()
    st.download_button(
        "📥 下载结构化解读 (.md)",
        data=st.session_state["analyze_result"],
        file_name=f"结构化解读_{paper_options[selected_id][:30]}.md",
        mime="text/markdown",
    )

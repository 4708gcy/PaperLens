"""多篇综述页面"""
import time
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import list_documents, synthesize

st.title("📝 多篇综述生成")
st.markdown("选择 2-5 篇论文 + 输入主题，AI 自动规划大纲、**并行检索写作**、生成综述报告。")

result = list_documents()
papers = result.get("data", []) if result.get("code") == 200 else []
indexed_papers = [p for p in papers if p.get("status") == "indexed"]

if len(indexed_papers) < 2:
    st.warning("综述需要至少 2 篇已处理的论文，请先上传")
    st.stop()

paper_options = {p["paper_id"]: p["title"] for p in indexed_papers}
selected_ids = st.multiselect(
    "选择论文（2-5 篇）",
    options=list(paper_options.keys()),
    format_func=lambda x: f"[{x}] {paper_options[x][:30]}"
)
if len(selected_ids) > 5:
    st.warning("建议选 2-5 篇，过多会显著变慢。")

topic = st.text_input("综述主题", placeholder="如：大模型推理加速方法对比")

# 选篇/主题变化时清掉旧结果
_state_key = f"synth_{'_'.join(map(str, selected_ids))}_{topic.strip()}"
if st.session_state.get("synth_last_key") != _state_key:
    st.session_state["synth_last_key"] = _state_key
    st.session_state.pop("synth_result", None)
    st.session_state.pop("synth_intent", None)

if st.button("🚀 生成综述", type="primary"):
    if len(selected_ids) < 2:
        st.error("至少选 2 篇论文")
        st.stop()
    if not topic.strip():
        st.error("请输入综述主题")
        st.stop()
    placeholder_block = st.chat_message("assistant")
    placeholder = placeholder_block.empty()
    full = ""
    intent = ""
    # 排序保证不同点击顺序生成相同的 thread_id
    sid_sorted = sorted(selected_ids)
    thread_id = f"synth_{'_'.join(map(str, sid_sorted))}_{int(time.time())}"
    try:
        for event in synthesize(selected_ids, topic, thread_id=thread_id):
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
    if full:
        st.session_state["synth_result"] = full
        st.session_state["synth_intent"] = intent
        st.rerun()

# 复用已持久化的结果
if "synth_result" in st.session_state:
    with st.chat_message("assistant"):
        intent = st.session_state.get("synth_intent", "")
        if intent:
            st.caption(f"🎯 意图：{intent}")
        st.markdown(st.session_state["synth_result"])
    st.divider()
    st.download_button(
        "📥 下载综述报告 (.md)",
        data=st.session_state["synth_result"],
        file_name=f"综述_{topic[:20] or 'report'}.md",
        mime="text/markdown",
    )

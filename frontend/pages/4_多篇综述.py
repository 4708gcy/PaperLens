"""多篇综述页面"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import list_documents, synthesize

st.title("📝 多篇综述生成")
st.markdown("选择 2-5 篇论文 + 输入主题，AI 自动规划大纲、**并行检索写作**、生成综述报告。")

result = list_documents()
papers = result.get("data", []) if result.get("code") == 200 else []
indexed_papers = [p for p in papers if p["status"] == "indexed"]

if len(indexed_papers) < 2:
    st.warning("综述需要至少 2 篇已处理的论文，请先上传")
    st.stop()

paper_options = {p["paper_id"]: p["title"] for p in indexed_papers}
selected_ids = st.multiselect(
    "选择论文（2-5 篇）",
    options=list(paper_options.keys()),
    format_func=lambda x: f"[{x}] {paper_options[x][:30]}"
)

topic = st.text_input("综述主题", placeholder="如：大模型推理加速方法对比")

if st.button("🚀 生成综述", type="primary"):
    if len(selected_ids) < 2:
        st.error("至少选 2 篇论文")
    elif not topic.strip():
        st.error("请输入综述主题")
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            thread_id = f"synth_{'_'.join(map(str, selected_ids))}"
            for event in synthesize(selected_ids, topic, thread_id=thread_id):
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

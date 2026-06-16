"""论文管理页面"""
import streamlit as st
import time
import sys
from pathlib import Path

# 添加项目根目录到 path（让 api_client 可导入）
sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import upload_document, list_documents, delete_document

st.title("📚 论文管理")
st.markdown("上传论文 PDF（或 DOC/DOCX/PPT/PPTX），AI 用 MinerU 解析、MiMo 理解图表、ES 建立检索索引。")

# 上传区
uploaded_file = st.file_uploader(
    "选择文件",
    type=["pdf", "docx", "doc", "pptx", "ppt", "png", "jpg", "jpeg"]
)

if uploaded_file and st.button("📤 上传并处理", type="primary"):
    with st.spinner("正在上传..."):
        try:
            result = upload_document(uploaded_file)
        except Exception as e:
            st.error(f"❌ 上传失败（后端可能未启动）：{e}")
            st.stop()
    if result.get("code") == 200:
        st.success(f"✅ {uploaded_file.name} 上传成功！paper_id={result['data']['paper_id']}")
        st.info("⏳ 后台正在处理（MinerU 解析 + MiMo 图表理解 + 向量化索引），约 1-3 分钟")
    else:
        st.error(f"❌ 上传失败：{result.get('msg')}")

st.divider()

# 论文列表
st.subheader("已上传的论文")
if st.button("🔄 刷新列表"):
    st.rerun()

result = list_documents()
papers = result.get("data", []) if result.get("code") == 200 else []

if not papers:
    st.info("还没有论文，先上传一个吧。")
else:
    for p in papers:
        status_emoji = {
            "processing": "⏳", "indexed": "✅", "failed": "❌", "pending": "⏸️"
        }.get(p.get("status", ""), "❓")
        col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
        with col1:
            st.write(f"{status_emoji} **[{p['paper_id']}]** {p['title']}")
        with col2:
            st.caption(f"{p.get('page_count', 0)} 页")
        with col3:
            st.caption(f"{p.get('chunk_count', 0)} 块")
        with col4:
            if st.button("🗑️", key=f"del_{p['paper_id']}", help="删除"):
                try:
                    r = delete_document(p["paper_id"])
                except Exception as e:
                    st.error(f"删除失败（后端可能未启动）：{e}")
                    st.stop()
                if r.get("code") == 200:
                    st.success("已删除")
                    time.sleep(0.5)
                    st.rerun()

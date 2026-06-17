"""论文管理页面"""
import streamlit as st
import time
import sys
from pathlib import Path

# 添加项目根目录到 path（让 api_client 可导入）
sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import upload_document, list_documents, delete_document

st.title("📚 论文管理")
st.markdown(
    "上传论文 / 课件（PDF / DOC / DOCX / PPT / PPTX / 图片），AI 用 MinerU 解析、"
    "qwen3.7-plus 理解图表、ES 建立检索索引。"
)

# 上传提示（可编辑 PDF 优先）
st.info(
    "💡 **格式建议**\n"
    "- **PDF**：请上传**可编辑 PDF**（有文本层）。扫描件 PDF（整页是图片）解析质量差，"
    "建议先用 WPS 等工具转为「可编辑 PDF」或「DOCX」后再上传。\n"
    "- **DOC/DOCX/PPT/PPTX**：图片少的文档建议用原格式直解（下方选项），质量通常更好且更快。"
)

# 上传区
uploaded_file = st.file_uploader(
    "选择文件",
    type=["pdf", "docx", "doc", "pptx", "ppt", "png", "jpg", "jpeg"]
)

# 高级选项：格式路由（仅对 Word/PPT 生效）+ 图片理解开关（所有格式生效）
is_non_pdf = uploaded_file is not None and not uploaded_file.name.lower().endswith(".pdf")
with st.expander("⚙️ 高级选项", expanded=False):
    st.markdown("**① 格式路由**（仅对 Word/PPT 生效，PDF 无关）")
    route_option = st.radio(
        "非 PDF 文档如何解析？",
        options=["auto", "direct", "pdf"],
        format_func=lambda x: {
            "auto": "🤖 智能判断（推荐）：按图片数自动决定",
            "direct": "📄 原格式直解：DOCX/PPTX 直接给 MinerU（图片少时更优）",
            "pdf": "🔄 强制转 PDF：先用 LibreOffice 转 PDF（图片多/扫描件时更稳）",
        }[x],
        index=0,
        disabled=not is_non_pdf,
        help="PDF 文件此选项无效。Word/PPT 图片少（<10张）时原格式直解质量通常更好；图片多或扫描件转的 Word 建议转 PDF。",
        key="route_option_radio",
    )
    force_pdf_choice = {"auto": None, "direct": False, "pdf": True}[route_option]

    st.markdown("**② 图片理解**（qwen3.7-plus 给图表生成描述，所有格式生效）")
    image_mode_choice = st.radio(
        "是否对图表做多模态理解？",
        options=["on", "off"],
        format_func=lambda x: {
            "on": "🖼️ 开启（默认）：处理全部图片，描述补进全文+索引（论文/报告适用）",
            "off": "🚫 关闭：完全跳过（教材/数学书适用，公式截图描述无意义，省 30+ 分钟）",
        }[x],
        index=0,
        help="论文里的架构图/实验图有价值，开启后能看图说话；数学书里的公式截图用文字描述价值低，关闭可大幅省时。",
        key="image_mode_radio",
    )

if uploaded_file and st.button("📤 上传并处理", type="primary"):
    with st.spinner("正在上传..."):
        try:
            result = upload_document(uploaded_file, force_pdf=force_pdf_choice,
                                     image_mode=image_mode_choice)
        except Exception as e:
            st.error(f"❌ 上传失败（后端可能未启动）：{e}")
            st.stop()
    if result.get("code") == 200:
        data = result.get("data", {}) or {}
        st.success(f"✅ {uploaded_file.name} 上传成功！paper_id={data.get('paper_id')}")
        # 展示格式路由决策
        if data.get("force_pdf") is True and is_non_pdf:
            st.caption("🔄 已按「转 PDF」处理（LibreOffice 转换 → MinerU）")
        elif is_non_pdf:
            st.caption("📄 已按「原格式直解」处理（DOCX/PPTX 直接给 MinerU）")
        # 展示图片理解决策
        if data.get("image_mode") == "off":
            st.caption("🚫 已跳过图片理解（教材/数学书模式）")
        else:
            st.caption("🖼️ 已开启图片理解（描述将补进全文+索引）")
        # 扫描件警告（后端检测到时返回）
        if data.get("scan_warning"):
            st.warning(data["scan_warning"])
        st.info("⏳ 后台正在处理（MinerU 解析 + qwen3.7-plus 图表理解 + 向量化索引），约 1-3 分钟")
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

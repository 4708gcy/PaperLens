"""PaperLens 前端主入口"""
import streamlit as st

st.set_page_config(
    page_title="PaperLens — 论文研究 Agent",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PaperLens — 论文研究智能体")

st.markdown("""
上传你的论文 PDF（或 DOC/PPTX），也可以上传**课件 / 知识文档**，然后用自然语言：

- **💬 论文精读**：针对单篇论文深度问答（流式响应 + 引用追溯）
- **🔍 结构化解读**：一键生成论文 5 段式解读（背景/方法/贡献/实验/局限）
- **📝 多篇综述**：跨多篇论文自动生成综述报告
- **🎓 学习助手**：上传课件/讲义，AI 辅导问答、要点总结、知识卡片、自测练习、**复习笔记**、**PPT 生成**

AI 会自动理解你的意图，所有回答都可追溯引用来源。
""")

pg = st.navigation([
    st.Page("pages/1_论文管理.py", title="论文管理", icon="📚"),
    st.Page("pages/2_论文精读.py", title="论文精读", icon="💬"),
    st.Page("pages/3_结构化解读.py", title="结构化解读", icon="🔍"),
    st.Page("pages/4_多篇综述.py", title="多篇综述", icon="📝"),
    st.Page("pages/5_学习助手.py", title="学习助手", icon="🎓"),
])
pg.run()

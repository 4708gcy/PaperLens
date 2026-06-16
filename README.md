# 📄 PaperLens — 论文与课件学习智能体

> 基于 **LangGraph 多分支 Agent** 的学习助手：上传论文 / 课件（PDF/DOCX/PPTX/图片）→ LibreOffice 转 PDF → MinerU 解析为 Markdown + MiMo 理解图表 → 两条数据通路：**论文精读走 ES 检索（BM25+KNN+RRF+Rerank）**，**学习助手走全文通读**（辅导问答 / 要点总结 / 知识卡片 / 自测练习 / 复习笔记 / PPT 生成）。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-orange)
![Elasticsearch](https://img.shields.io/badge/ES-9.3-yellow)

## ✨ 核心功能

### 📖 论文精读（ES 检索通路）
- **💬 论文精读问答**：针对单篇论文深度多轮问答（流式响应 + 引用追溯）
- **🔍 结构化解读**：一键生成论文 5 段式解读（研究背景 / 核心方法 / 主要贡献 / 实验结论 / 局限展望）
- **📝 多篇综述**：跨多篇论文自动规划大纲 → **Send 并行检索写作** → 合并综述报告

### 🎓 学习助手（全文通读通路）
- **📖 辅导问答**：苏格拉底式讲解，基于全文答 + 反问引导
- **📋 要点总结**：通读全文生成结构化大纲摘要
- **🗂️ 知识卡片**：抽取核心概念生成 Anki 风格闪卡（JSON，可下载）
- **📝 自测练习**：基于全文出选择题并可交互作答（JSON，可下载）
- **📒 复习笔记**：分步向导（配置→大纲→你编辑→生成），知识框架 + 易错点 + 记忆口诀
- **📽️ PPT 生成**：分步向导（选主题/页数→大纲→你编辑→生成），Marp 格式可一键导出 PPT

### 📎 通用能力
- **多格式输入**：PDF / DOC / DOCX / PPT / PPTX / 图片，非 PDF 自动经 LibreOffice 转 PDF
- **图表理解**：MiMo-v2-omni 给每张图表生成文字描述，补进 markdown 全文 + 进 ES 索引
- **多文档综合**：学习助手支持一次选多篇课件，综合跨资料学习

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit 前端 (8501)                        │
│  论文管理 │ 论文精读 │ 结构化解读 │ 多篇综述 │ 学习助手(6模式) │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI 后端 (8000)                           │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           LangGraph StateGraph（核心）                   ││
│  │  triage → [qa/analyze(走检索) | synthesize | learn(走全文)││
│  │             ↓                    ↓                       ││
│  │         retrieve(ES)          learn_agent                ││
│  │             ↓              读全文markdown                ││
│  │      qa/analyze/Send并行                                ││
│  └─────────────────────────────────────────────────────────┘│
│  DocumentService: LibreOffice → MinerU → MiMo图表 → 分块     │
│  LearnRouter: /learn/outline 大纲生成（分步向导）             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌───────────────────┐             ┌─────────────────────┐
│  Elasticsearch    │             │  Qwen3.7-plus       │
│  论文精读/解读/综述 │             │  + MiMo-v2-omni     │
│  BM25+KNN+Rerank  │             │  (思考模式可配置)    │
└───────────────────┘             └─────────────────────┘
```

**两条数据通路**：
- **论文精读 / 结构化解读 / 多篇综述**：走 ES top-k 检索（BM25+KNN+RRF+Rerank），适合长论文的局部精准问答
- **学习助手（6 模式）**：走磁盘上的完整 Markdown（含图表描述），LLM 通读全文后回答，避免"只见片段"

## 📊 评测结果（简历核心数据）

### 4 种检索策略对比（20 条问答对）

| 策略 | MRR | Recall@3 | Recall@5 |
|------|-----|----------|----------|
| A. 纯 BM25 | 0.283 | 35.0% | 45.0% |
| **B. 纯向量（BGE-m3）** | **0.700** | **85.0%** | **85.0%** |
| C. BM25+向量+RRF | 0.472 | 50.0% | 60.0% |
| D. C+Reranker | 0.717 | 80.0% | 80.0% |

**关键发现**：中英跨语言场景下，纯向量 BGE-m3 优于 BM25+向量混合——BM25 对中文 query 检索英文文档效果差，反而拉低 RRF 融合。

### LLM-as-Judge 答案质量（8 条）

| 指标 | 分数 |
|------|------|
| 无 RAG | 2.38 / 5 |
| **有 RAG** | **4.00 / 5** |
| **提升** | **+68.4%** |

![检索策略对比](eval/results_chart.png)

## 🔧 环境要求

- **Python 环境**：`conda activate ocr`（MinerU 所在环境，全程必须）
- **Elasticsearch 9.3**：本地安装 + IK 中文分词器
- **MinerU**：`mineru-open-api`（首次 extract 模式需 `mineru-open-api auth`）
- **LibreOffice**：`soffice` 命令（非 PDF 转 PDF）
- **BGE 模型**：本地路径 `E:/ai八斗学院学习/models/BAAI/bge-m3` + `bge-reranker-v2-m3`
- **API Keys**：DashScope（Qwen3.7-plus）+ 小米 MiMo

## 🚀 快速启动

```bash
# 0. 全程在 ocr 环境
conda activate ocr

# 1. 启动 Elasticsearch（本地安装）
cd E:\elasticsearch-9.3.1\elasticsearch-9.3.1
./bin/elasticsearch

# 2. 配置密钥
cp .env.example .env  # 填 DASHSCOPE_API_KEY, MIMO_API_KEY

# 3. 启动后端
cd E:\其他\大模型项目\PaperLens
start_backend.bat
# 或手动：python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 4. 启动前端（新终端，同样 conda activate ocr）
cd frontend
streamlit run app.py --server.port 8501

# 浏览器访问 http://localhost:8501
```

## 📁 项目结构

```
PaperLens/
├── app/
│   ├── main.py              # FastAPI 入口 + 环境自检
│   ├── config.py            # YAML + 环境变量替换
│   ├── schemas.py           # Pydantic 请求/响应（含 learn_outline/config）
│   ├── exceptions.py        # 全局异常处理
│   ├── core/
│   │   ├── chunker.py       # 滑动窗口分块
│   │   ├── pdf_splitter.py  # PDF 拆分 + LibreOffice 转换
│   │   ├── embedding.py     # BGE-m3 单例
│   │   ├── reranker.py      # BGE-reranker-v2-m3
│   │   ├── rag_engine.py    # BM25+KNN+RRF+Rerank
│   │   └── multimodal.py    # MiMo-v2-omni 图表理解
│   ├── agents/
│   │   ├── state.py         # LangGraph 状态（含 learn_outline/config）
│   │   ├── prompts.py       # 集中 Prompt（含学习助手6模式 + 大纲提示词）
│   │   └── graph.py         # StateGraph（triage→4分支，learn直达全文）
│   ├── routers/
│   │   ├── documents.py     # 上传/列表/删除 + 获取全文markdown API
│   │   ├── chat.py          # 对话 + SSE 流式
│   │   └── learn.py         # 学习助手大纲生成接口
│   ├── services/
│   │   └── document_service.py  # MinerU→MiMo→图片描述补全文→分块→索引
│   └── models/orm.py        # SQLAlchemy ORM
├── scripts/
│   └── rebuild.py           # 重建所有文档（新管线重跑）
├── eval/                    # 评测脚本 + 数据集 + 结果
├── tests/                   # pytest 测试（graph路由/chunker/rag_engine/retrieve/synthesize）
├── frontend/                # Streamlit 5 页面
├── config.yaml              # 配置（含 enable_thinking / max_images_per_dir）
├── .env.example / requirements.txt
└── start_backend.bat        # Windows 启动脚本（ocr 环境）
```

## 🛠️ 技术栈

LangGraph · FastAPI · Elasticsearch · Streamlit · BGE-m3 · BGE-reranker-v2-m3 · mineru-open-api · LibreOffice · MiMo-v2-omni · Qwen3.7-plus · SQLite · pytest

## 🧪 测试

```bash
conda activate ocr
python -m pytest tests/test_graph.py -v
# graph 路由测试（13 passed，含 learn 直达 learn_agent 的跳过检索测试）
```

## 📝 关键工程决策

1. **两条数据通路**：论文精读走 ES top-k 检索（长文档精准局部问答），学习助手走全文通读（笔记/PPT/总结需全局视野）。learn 意图直达 learn_agent，跳过冗余的 ES 检索。
2. **图表理解补进全文**：MiMo 给每张图生成描述，既写进 markdown 全文（供学习助手理解图表），又作为独立 ES chunk（供检索命中），图片不再"隐形"。
3. **分步向导生成**：笔记和 PPT 不是黑箱——先生成大纲让用户编辑确认，再按确认的大纲生成成品，全程 `session_state` 持久化，点控件不会丢内容。
4. **思考模式可配置**：qwen3.7-plus 是混合思考模型，`config.yaml` 的 `enable_thinking` 控制开关（true 质量高但首 token 慢，false 快）。
5. **多文档全文防爆**：学习助手支持多选，`_load_full_markdown` 对超长情况按篇均摊截断（每篇留头尾），保证每篇都能被看到。
6. **每篇论文独立 ES 索引**：综述时并行查 N 索引无需 filter；删除=删索引；BM25 词频不受干扰。
7. **Send API 并行综述**：planner 生成大纲后，Send 为每个章节启动并行 worker，比串行快 3-5 倍。
8. **重建脚本**：`scripts/rebuild.py` 用新管线重跑所有文档（图片描述补全文等），无需重新上传。

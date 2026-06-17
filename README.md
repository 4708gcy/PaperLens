# 📄 PaperLens — 论文与课件学习智能体

> 基于 **LangGraph 多分支 Agent** 的学习助手：上传论文 / 课件（PDF/DOCX/PPTX/图片）→ 智能格式路由 → MinerU 解析为 Markdown + qwen3.7-plus 视觉理解图表 → **三级素材策略**（能通读就全文通读，超长文档降级检索/截断）→ 论文精读/解读/综述走 **ES 检索 RAG**（BM25+KNN+RRF+Rerank），学习助手 6 模式走**全文通读**（辅导问答 / 要点总结 / 知识卡片 / 自测练习 / 复习笔记 / PPT 生成）。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-orange)
![Elasticsearch](https://img.shields.io/badge/ES-9.3-yellow)

## ✨ 核心功能

### 📖 论文精读（ES 检索通路）
- **💬 论文精读问答**：针对单篇论文深度**多轮问答**（流式响应 + 引用追溯 + **历史记忆**）
- **🔍 结构化解读**：一键生成论文 5 段式解读（研究背景 / 核心方法 / 主要贡献 / 实验结论 / 局限展望）
- **📝 多篇综述**：跨多篇论文自动规划大纲 → **Send 并行检索写作** → 合并综述报告
- **🔎 综合问答**：跨多文档 ES 检索，在大量资料里定位/对比某主题（适合"哪些资料讲过 X"）

### 🎓 学习助手（全文通读通路）
- **📖 辅导问答**：苏格拉底式讲解，基于全文答 + 反问引导
- **📋 要点总结**：通读全文生成结构化大纲摘要
- **🗂️ 知识卡片**：抽取核心概念生成 Anki 风格闪卡（JSON，可下载）
- **📝 自测练习**：基于全文出选择题并可交互作答（JSON，可下载）
- **📒 复习笔记**：分步向导（配置→大纲→你编辑→生成），支持单文件/分章节两种模式
- **📽️ PPT 生成**：分步向导（选主题/页数→大纲→你编辑→生成），Marp 格式 Markdown

### 📎 通用能力
- **智能格式路由**：上传时可选「智能判断/原格式直解/强制转 PDF」。非 PDF 文档按图片数自动决策（图片少走原格式直解 MinerU，更优更快；图片多走 LibreOffice 转 PDF，更稳）。PDF 上传时检测扫描件并提示。
- **图片理解开关**：上传时可选「全开/全关」。论文/报告（架构图有价值）开；教材/数学书（公式截图无意义）关，省 30+ 分钟。
- **图表理解**：qwen3.7-plus 的视觉能力给每张图表生成文字描述（并发处理），补进 markdown 全文 + 进 ES 索引
- **多轮记忆**：qa/general/learn-qa 节点注入最近 N 轮历史，模型"记得刚才聊了什么"
- **多文档综合**：学习助手支持一次选多篇课件，综合跨资料学习

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit 前端 (8501)                        │
│  论文管理 │ 论文精读 │ 结构化解读 │ 多篇综述 │ 学习助手(6模式) │ 综合问答 │
└─────────────────┬───────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI 后端 (8000)                           │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           LangGraph StateGraph（核心）                   ││
│  │  triage → [qa/analyze(走检索) | synthesize | learn      ││
│  │             ↓                    ↓                       ││
│  │         retrieve(ES)       learn_agent                   ││
│  │             ↓            三级素材策略：                   ││
│  │      qa/analyze/Send    ①全文通读 ②超限检索 ③超限截断     ││
│  └─────────────────────────────────────────────────────────┘│
│  DocumentService: 格式路由→MinerU→qwen3.7-plus图表→排版清洗→分块│
│  LearnRouter: /learn/outline 大纲生成（fast模型，关thinking） │
└─────────────────┬───────────────────────────────────────────┘
                  ▼
        ┌─────────┴─────────────┐
        ▼                       ▼
┌───────────────────┐   ┌─────────────────────┐
│  Elasticsearch    │   │  Qwen3.7-plus       │
│  检索/综述/章节笔记 │   │  主LLM(开thinking)  │
│  BM25+KNN+Rerank  │   │  fast_llm(关thinking)│
└───────────────────┘   └─────────────────────┘
```

**两条数据通路 + 三级素材策略**：
- **论文精读 / 结构化解读 / 多篇综述**：走 ES top-k 检索（BM25+KNN+RRF+Rerank），适合长论文的局部精准问答
- **学习助手（6 模式）**：走磁盘上的完整 Markdown（含图表描述），LLM 通读全文后回答，避免"只见片段"
- **三级素材降级**（学习助手内部）：① 全文 ≤ 上下文预算 → **通读全文**（论文/短课件最优）；② 超限 + 分章节笔记 → **ES 检索降级**（精确相关片段）；③ 超限 + 其他模式 → **头尾截断全文**（保视野）

## 📊 评测结果（简历核心数据）

### 4 种检索策略对比（20 条问答对抽样）

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
- **LibreOffice**：`soffice` 命令（非 PDF 转 PDF，仅在「强制转 PDF」模式触发）
- **BGE 模型**：本地路径 `E:/ai八斗学院学习/models/BAAI/bge-m3` + `bge-reranker-v2-m3`
- **API Keys**：DashScope（Qwen3.7-plus 文本 + qwen-turbo 快速模型 + 图表视觉，一个 Key 即可）

## 🚀 快速启动

```bash
# 0. 全程在 ocr 环境
conda activate ocr

# 1. 启动 Elasticsearch（本地安装）
cd E:\elasticsearch-9.3.1\elasticsearch-9.3.1
./bin/elasticsearch

# 2. 配置密钥
cp .env.example .env  # 填 DASHSCOPE_API_KEY

# 3. 启动后端
cd E:\其他\大模型项目\PaperLens
start_backend.bat
# 或手动：python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 4. 启动前端（新终端，同样 conda activate ocr）
cd frontend
streamlit run app.py --server.port 8501

# 浏览器访问 http://localhost:8501
```

### 上传文档建议

- **PDF**：请上传**可编辑 PDF**（有文本层）。扫描件 PDF 解析质量差，建议先用 WPS 转为「可编辑 PDF」或「DOCX」。
- **DOC/DOCX/PPT/PPTX**：图片少的文档选「原格式直解」（MinerU 直读 Word 更好更快）；图片多或扫描件转的 Word 选「强制转 PDF」。
- **图片理解**：论文/报告开（架构图有价值）；教材/数学书关（公式截图无意义，省时）。

## 📁 项目结构

```
PaperLens/
├── app/
│   ├── main.py              # FastAPI 入口 + 环境自检
│   ├── config.py            # YAML + 环境变量替换
│   ├── schemas.py           # Pydantic 请求/响应（含 learn_outline/config/image_mode）
│   ├── exceptions.py        # 全局异常处理
│   ├── core/
│   │   ├── chunker.py       # 滑动窗口分块
│   │   ├── pdf_splitter.py  # PDF 拆分 + LibreOffice 转换 + 图片计数
│   │   ├── embedding.py     # BGE-m3 单例
│   │   ├── reranker.py      # BGE-reranker-v2-m3
│   │   ├── rag_engine.py    # BM25+KNN+RRF+Rerank
│   │   └── multimodal.py    # qwen3.7-plus 图表理解（并发）
│   ├── agents/
│   │   ├── state.py         # LangGraph 状态（含 learn_outline/config）
│   │   ├── prompts.py       # 集中 Prompt（含学习助手6模式 + 大纲提示词）
│   │   └── graph.py         # StateGraph + 三级素材策略 + 章节感知分批 + 多轮记忆
│   ├── routers/
│   │   ├── documents.py     # 上传(格式路由+图片模式)/列表/删除
│   │   ├── chat.py          # 对话 + SSE 流式
│   │   └── learn.py         # 学习助手大纲生成(关thinking) + PPT导出
│   ├── services/
│   │   └── document_service.py  # 格式路由→MinerU→图表理解→排版清洗→中间文件清理→分块→索引
│   └── models/orm.py        # SQLAlchemy ORM
├── scripts/
│   └── rebuild.py           # 重建所有文档（新管线重跑）
├── eval/                    # 评测脚本 + 数据集 + 结果
├── tests/                   # pytest 测试（graph路由/chunker/多轮记忆/章节分批）
├── frontend/                # Streamlit 6 页面
├── config.yaml              # 配置（含 enable_thinking / image_mode）
├── .env.example / requirements.txt
└── start_backend.bat        # Windows 启动脚本（ocr 环境）
```

## 🛠️ 技术栈

LangGraph · FastAPI · Elasticsearch · Streamlit · BGE-m3 · BGE-reranker-v2-m3 · mineru-open-api · LibreOffice · Qwen3.7-plus（文本+视觉）· qwen-turbo（快速模型）· SQLite · pytest

## 🧪 测试

```bash
conda activate ocr
python -m pytest tests/test_graph.py tests/test_chunker.py -v
# 21 passed：13 路由 + 3 多轮记忆 + 3 章节分批 + 2 chunker/splitter
```

## 📝 关键工程决策

1. **两条数据通路**：论文精读走 ES top-k 检索（长文档精准局部问答），学习助手走全文通读（笔记/PPT/总结需全局视野）。learn 意图直达 learn_agent，跳过冗余的 ES 检索。
2. **三级素材降级**：学习助手内部按文档长度自适应——能通读就通读（论文最优），超长文档的分章节笔记降级检索（精确），其他模式降级截断（保视野）。兼顾质量与上下文限制。
3. **图表理解补进全文**：qwen3.7-plus 的视觉能力给每张图生成描述（并发处理），既写进 markdown 全文（供学习助手理解图表），又作为独立 ES chunk（供检索命中），图片不再"隐形"。
4. **分步向导生成**：笔记和 PPT 不是黑箱——先生成大纲让用户编辑确认，再按确认的大纲生成成品，全程 `session_state` 持久化，点控件不会丢内容。
5. **思考模型分级**：qwen3.7-plus 是混合思考模型，深度问答/综述开 thinking（重质量，但输入上限 98K），大纲/意图分类用 qwen-turbo 关 thinking（重速度，输入上限 1M，绕开 98K 限制）。
6. **智能格式路由**：非 PDF 文档按图片数自动决策——图片少走原格式直解（DOCX/PPTX 直接给 MinerU，质量更好），图片多走 LibreOffice 转 PDF（更稳）。用户可手动覆盖。
7. **多轮记忆**：qa/general/learn-qa 节点注入最近 N 轮历史（`_build_chat_messages`），模型能"记得刚才聊了什么"，不再回答"这是我们第一次交流"。
8. **排版清洗 + 中间文件清理**：MinerU 输出后做轻量清洗（合并错误断行、压空行、删页码），清理中间 part md（保留 images 目录），最终只留一份高质量顶层 Markdown。
9. **每篇论文独立 ES 索引**：综述时并行查 N 索引无需 filter；删除=删索引。
10. **Send API 并行综述**：planner 生成大纲后，Send 为每个章节启动并行 worker，比串行快 3-5 倍。

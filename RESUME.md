# PaperLens 简历条目 + 面试话术

## 📋 简历项目描述（直接复制到简历）

### 版本 A（详细版，4 行）

**PaperLens — 论文研究智能体** | 个人独立项目 | LangGraph · FastAPI · Elasticsearch · Streamlit

- 设计并实现面向科研场景的论文研究 Agent，支持单论文精读问答、结构化解读、跨多篇论文综述，所有结论可追溯引用来源；支持 PDF/DOCX/PPTX 多格式输入（LibreOffice 转 PDF → mineru-open-api 解析 → MiMo-v2-omni 图表理解）。
- 基于 **LangGraph** 构建多分支 Agent 图（意图分类→检索→专业分支），综述模块用 **Send API** 实现章节级并行（5 章并行写作，比串行快 3 倍）；集成 checkpointer 多轮记忆 + SSE 流式响应。
- 构建多路召回 RAG 引擎（BM25 + BGE-m3 向量 + RRF + Cross-Encoder 重排序），自建 60 条问答评测集对比 4 种检索策略，发现中英跨语言场景纯向量（MRR 0.70）优于 BM25+向量混合（MRR 0.47），LLM-as-Judge 验证 RAG 提升答案质量 +68.4%。
- 工程化：FastAPI 分层架构 + SQLite/ES 双库 + 全局异常 + 16 个 pytest 测试 + 环境自检（mineru/soffice 可用性），独立解决 8 个工程难题（ES 9.x 兼容、torch<2.6 bin 加载限制、Windows 后台线程 PATH 缺失等）。

### 版本 B（精简版，3 行）

**PaperLens — 论文研究智能体** | LangGraph · FastAPI · Elasticsearch · BGE-m3

- 基于 LangGraph 多分支 Agent 实现论文精读问答、5 段结构化解读、多篇综述（Send API 并行），集成 MinerU 文档解析 + MiMo 多模态图表理解，支持 PDF/DOCX/PPTX 多格式。
- 构建多路召回 RAG 引擎（BM25+向量+RRF+Cross-Encoder 重排序），自建 60 条评测集，4 策略对比发现跨语言场景纯向量 MRR 0.70 优于混合 0.47，LLM-as-Judge 验证 RAG 提升答案质量 +68.4%。
- FastAPI 分层架构 + SSE 流式 + checkpointer 多轮记忆 + 16 个 pytest 测试，独立解决 ES 9.x 兼容、torch<2.6 限制、Windows PATH 等 8 个工程难题。

---

## 🎯 面试讲解话术

### 讲 RAG（必问）

> "我做 BM25+向量+RRF+重排序四阶段管线。但评测时发现一个反直觉的结果：**中英跨语言场景下，纯向量(BGE-m3)的 MRR(0.70)竟然高于 BM25+向量混合(0.47)**。原因是评测集是中文问题检索英文论文，BM25 对中文 query 匹配英文文档效果很差（关键词几乎不重合），这些低质量结果通过 RRF 拉低了整体。这个发现让我意识到 **RAG 不是无脑上混合检索就好，要根据 query-doc 语言匹配度选择策略**。最后我加了 Cross-Encoder 重排序，MRR 从 0.47 提到 0.72，证明精排确实有效。"

### 讲 Agent 综述（核心亮点）

> "多篇综述是最有挑战的。我用 **LangGraph 的 Send API** 实现 map-reduce 式并行：planner 节点用 LLM 生成 4-6 章大纲（JSON），然后路由函数返回 `[Send("section_worker", sub_state), ...]`，LangGraph 为每个章节启动一个并行 worker。每个 worker 独立检索多篇论文 + LLM 写章节，结果通过 `Annotated[list, operator.add]` reducer 自动汇聚，最后 assembler 合并成完整综述。5 章并行比串行快约 3 倍（30s vs 90s）。
>
> 遇到的坑：DashScope 并发请求触发限流（401），我加了 retry + 指数退避 + 降级容错——单个章节失败不影响整体，只显示'本章节暂未生成'。"

### 讲工程化（后端岗必问）

> "分层 routers/services/core/agents 四层。每篇论文独立 ES 索引，综述时并行查多个索引无需 filter。文档解析异步（BackgroundTasks），上传即刻返回 paper_id。
>
> 这个项目我独立解决了 8 个工程难题：① ES 9.x 移除了 timeout 参数要改 request_timeout；② transformers 要求 torch≥2.6 才能加载 .bin（安全限制），我写了脚本把 pytorch_model.bin 转 safetensors；③ Windows 下 uvicorn 后台线程 PATH 缺失找不到 mineru-open-api，我写了 `_resolve_cmd` 函数探测完整路径；④ LibreOffice 多格式转换；⑤ mineru 200 页限制自动拆分；⑥ RRF rank 起算修正；⑦ ES bulk 批量写入；⑧ SSE 过滤 triage 中间输出。每个都有 commit 记录可查。"

### 讲评测（最大差异化）

> "我发现大多数简历项目只说'我用了 RAG'，不说效果好不好。所以我自建评测体系：用 LLM 从 ES chunks 自动生成 60 条问答对（relevant_chunk_indices 自动标注），跑 4 策略对比 MRR/Recall，再用 LLM-as-Judge 对比有/无 RAG 的答案质量。结论：RAG 让答案质量从 2.38/5 提到 4.00/5，**提升 68.4%**。这个数据是我简历上最有说服力的部分——面试官问'你怎么知道 RAG 有效'，我有图有表有数字。"

---

## 📊 可量化的成果（填入简历）

| 指标 | 数值 |
|------|------|
| 评测集规模 | 60 条问答对（2 篇论文） |
| 纯向量 MRR | 0.70（最优单策略） |
| +Reranker MRR | 0.72 |
| 答案质量提升 | 2.38 → 4.00（+68.4%） |
| 综述并行加速 | 5 章 30s（串行需 90s） |
| 测试覆盖 | 16 个 pytest |
| 代码量 | 4500+ 行 |
| 解决的工程难题 | 8 个 |

---

## ❓ 预判面试问题

**Q: 为什么用 LangGraph 不用 openai-agents？**
A: 综述需要"先规划大纲→分章节并行检索写作→合并"的复杂图结构，LangGraph 的 StateGraph + Send API 更适合表达 map-reduce 模式。openai-agents 更适合 LLM 主导的灵活路由。

**Q: 为什么每篇论文独立索引？**
A: 三个理由：① 综述时并行查 N 个索引无需 filter，性能好；② 删除论文直接删索引，干净；③ BM25 词频统计不受其他论文干扰，单论文检索更准。

**Q: BGE-m3 为什么比 BM25+向量混合好？**
A: 评测集是中文问题检索英文论文。BM25 靠关键词匹配，中英文关键词几乎不重合，BM25 召回质量差，通过 RRF 反而拉低了向量检索的好结果。这告诉我"混合检索不是银弹，要看场景"。

**Q: Send 并行怎么实现的？**
A: 路由函数返回 `[Send("node", sub_state), ...]` 列表，LangGraph 见到 Send 列表会为每个启动一个并行节点执行。结果通过 `Annotated[list, operator.add]` reducer 自动汇聚。这是 map-reduce 模式。

**Q: 怎么解决并发限流？**
A: section_worker 加了 retry + 指数退避（2s/4s/6s/8s），单章节全部失败则降级为占位文本，不影响其他章节和整体综述。

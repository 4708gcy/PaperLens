# PaperLens 简历条目 + 面试话术

## 📋 简历项目描述（直接复制到简历）

### 版本 A（详细版，4 行）

**PaperLens — 论文与课件学习智能体** | 个人独立项目 | LangGraph · FastAPI · Elasticsearch · Streamlit

- 设计并实现面向学习/科研场景的智能体，覆盖**两条数据通路**：论文精读/结构化解读/多篇综述/综合问答走 **ES 多路召回 RAG**（BM25+BGE-m3+RRF+Cross-Encoder 重排），学习助手 6 种模式（辅导问答/要点总结/知识卡片/自测练习/复习笔记/PPT 生成）走**全文通读**。支持 PDF/DOCX/PPTX 多格式（LibreOffice 转 PDF → MinerU 解析 → qwen3.7-plus 视觉理解图表，描述补进全文+索引）。
- 基于 **LangGraph** 构建多分支 Agent 图：意图分类后 learn 直达全文 agent（跳过冗余检索），synthesize 用 **Send API** 章节级并行（5 章并行写作，比串行快 3 倍）；笔记/PPT 采用**分步向导**（生成大纲→用户编辑→按确认大纲生成），全程 session_state 持久化。
- 构建多路召回 RAG 引擎（BM25 + BGE-m3 向量 + RRF + Cross-Encoder 重排序），自建 60 条问答评测集对比 4 种检索策略，发现中英跨语言场景纯向量（MRR 0.70）优于 BM25+向量混合（MRR 0.47），LLM-as-Judge 验证 RAG 提升答案质量 +68.4%。
- 工程化：FastAPI 分层架构 + SQLite/ES 双库 + 全局异常 + 13 个 pytest 测试 + 环境自检（mineru/soffice 可用性），独立解决工程难题（ES 9.x 兼容、思考模型流式静默期、Windows 后台线程 PATH 缺失等）。

### 版本 B（精简版，3 行）

**PaperLens — 论文与课件学习智能体** | LangGraph · FastAPI · Elasticsearch · BGE-m3

- 基于 LangGraph 多分支 Agent 实现论文精读问答、结构化解读、多篇综述（Send API 并行）；并扩展为学习助手（辅导问答/闪卡/自测/复习笔记/PPT 分步向导），区分两条通路：RAG 检索（论文精读）与全文通读（学习助手），支持 PDF/DOCX/PPTX 多格式 + qwen3.7-plus 视觉图表理解（一个模型搞定文本+多模态）。
- 构建多路召回 RAG 引擎（BM25+向量+RRF+Cross-Encoder 重排序），自建 60 条评测集，4 策略对比发现跨语言场景纯向量 MRR 0.70 优于混合 0.47，LLM-as-Judge 验证 RAG 提升答案质量 +68.4%。
- FastAPI 分层架构 + SSE 流式 + checkpointer 多轮记忆 + 思考模式可配置 + 多文档全文截断保护 + 13 个 pytest 测试，独立解决 ES 9.x 兼容、思考模型流式、Windows PATH 等工程难题。

---

## 🎯 面试讲解话术

### 讲 RAG（必问）

> "我做 BM25+向量+RRF+重排序四阶段管线。但评测时发现一个反直觉的结果：**中英跨语言场景下，纯向量(BGE-m3)的 MRR(0.70)竟然高于 BM25+向量混合(0.47)**。原因是评测集是中文问题检索英文论文，BM25 对中文 query 匹配英文文档效果很差（关键词几乎不重合），这些低质量结果通过 RRF 拉低了整体。这个发现让我意识到 **RAG 不是无脑上混合检索就好，要根据 query-doc 语言匹配度选择策略**。最后我加了 Cross-Encoder 重排序，MRR 从 0.47 提到 0.72，证明精排确实有效。"

### 讲"两条数据通路"（核心架构决策）

> "项目最有意思的架构决策是**区分两条通路**。最初所有功能都走 RAG top-k 检索，但我发现复习笔记/PPT 这类任务质量很差——它们需要全局视野（整篇讲了什么、章节怎么排布），而 top-5 检索片段（约2500字）只能看到局部。所以我改成：**论文精读/解读/综述走 ES 检索**（长文档局部精准问答），**学习助手 6 种模式走全文 markdown**（LLM 通读全文后回答）。learn 意图直达 learn_agent，跳过冗余的 ES 检索计算。多文档场景还做了全文截断保护（按篇均摊，每篇留头尾），避免上下文爆炸。"

### 讲学习助手分步向导（产品思维）

> "学习助手的复习笔记和 PPT 不是黑箱一键生成。我做了**分步向导**：第一步用 fast 模型基于全文出大纲，第二步用户在大纲上自由增删改，第三步按确认的大纲生成成品。全程用 Streamlit 的 session_state 持久化——这里有个经典坑：生成内容如果只放在 `if st.button` 块里，用户点任何控件都会触发脚本重跑，按钮变 False，内容就消失了。我用 session_state + rerun 模式解决了这个，所有 tab（问答/总结/闪卡/自测/笔记/PPT）都持久化。"

### 讲图表理解补全文（工程深度）

> "MinerU 解析 PDF 会提取图片，但有个问题：markdown 里只有图片路径，LLM 读全文时根本不知道图表画了什么。我用 **qwen3.7-plus 的视觉能力**给每张图生成文字描述，然后**补进 markdown 全文末尾**——这样学习助手读全文时能真正理解图表。同时这些描述也作为独立的 image_caption chunk 进 ES，检索也能命中。论文3 之前一张图描述都没有，现在有 16 条。而且复用主力模型，不用再单独维护一个多模态服务。"

### 讲思考模型流式坑（Debug 能力）

> "qwen3.7-plus 是混合思考模型，我遇到一个很隐蔽的 bug：流式输出时模型先吐 600+ 个 reasoning_content token（content 字段为空），再吐正式回答。前 45 秒前端收不到任何数据，流就超时断了。我写了探针逐层定位：直接调 API、独立 LangGraph、完整 graph、复刻 httpx 流，最终确认是思考静默期。解决方案是 `enable_thinking` 开关（做成 config 可配置），关掉后首 token 从 45 秒降到 1 秒。"

### 讲 Agent 综述（核心亮点）

> "多篇综述是最有挑战的。我用 **LangGraph 的 Send API** 实现 map-reduce 式并行：planner 节点用 LLM 生成 4-6 章大纲（JSON），然后路由函数返回 `[Send("section_worker", sub_state), ...]`，LangGraph 为每个章节启动一个并行 worker。每个 worker 独立检索多篇论文 + LLM 写章节，结果通过 `Annotated[list, operator.add]` reducer 自动汇聚，最后 assembler 合并成完整综述。5 章并行比串行快约 3 倍（30s vs 90s）。
>
> 遇到的坑：DashScope 并发请求触发限流（401），我加了 retry + 指数退避 + 降级容错——单个章节失败不影响整体，只显示'本章节暂未生成'。"

### 讲工程化（后端岗必问）

> "分层 routers/services/core/agents 四层。每篇论文独立 ES 索引，综述时并行查多个索引无需 filter。文档解析异步（BackgroundTasks），上传即刻返回 paper_id。
>
> 这个项目我独立解决了多个工程难题：① ES 9.x 移除了 timeout 参数要改 request_timeout；② 思考模型流式 reasoning_content 静默期导致前端超时；③ Windows 下 uvicorn 后台线程 PATH 缺失找不到 mineru-open-api，写了 `_resolve_cmd` 探测完整路径；④ LibreOffice 多格式转换；⑤ mineru 200 页限制自动拆分；⑥ RRF rank 起算修正；⑦ Streamlit session_state 持久化（点控件内容消失的系统性修复）；⑧ ES bulk 批量写入 + 重建脚本避免 chunk 重复堆积。每个都有代码可查。"

### 讲评测（最大差异化）

> "我发现大多数简历项目只说'我用了 RAG'，不说效果好不好。所以我自建评测体系：用 LLM 从 ES chunks 自动生成 60 条问答对（relevant_chunk_indices 自动标注），跑 4 策略对比 MRR/Recall，再用 LLM-as-Judge 对比有/无 RAG 的答案质量。结论：RAG 让答案质量从 2.38/5 提到 4.00/5，**提升 68.4%**。这个数据是我简历上最有说服力的部分——面试官问'你怎么知道 RAG 有效'，我有图有表有数字。"

---

## 📊 可量化的成果（填入简历）

| 指标 | 数值 |
|------|------|
| 功能覆盖 | 论文精读/解读/综述/综合问答 + 学习助手6模式（共 6 页面） |
| 数据通路 | 2 条（ES 检索 / 全文通读） |
| 评测集规模 | 60 条问答对（2 篇论文） |
| 纯向量 MRR | 0.70（最优单策略） |
| +Reranker MRR | 0.72 |
| 答案质量提升 | 2.38 → 4.00（+68.4%） |
| 综述并行加速 | 5 章 30s（串行需 90s） |
| 测试覆盖 | 13 个 pytest（graph 路由） |
| 解决的工程难题 | 8+ 个 |

---

## ❓ 预判面试问题

**Q: 为什么用 LangGraph 不用 openai-agents？**
A: 综述需要"先规划大纲→分章节并行检索写作→合并"的复杂图结构，LangGraph 的 StateGraph + Send API 更适合表达 map-reduce 模式。openai-agents 更适合 LLM 主导的灵活路由。

**Q: 为什么每篇论文独立索引？**
A: 三个理由：① 综述时并行查 N 个索引无需 filter，性能好；② 删除论文直接删索引，干净；③ BM25 词频统计不受其他论文干扰，单论文检索更准。

**Q: BGE-m3 为什么比 BM25+向量混合好？**
A: 评测集是中文问题检索英文论文。BM25 靠关键词匹配，中英文关键词几乎不重合，BM25 召回质量差，通过 RRF 反而拉低了向量检索的好结果。这告诉我"混合检索不是银弹，要看场景"。

**Q: 学习助手为什么走全文不走检索？**
A: 复习笔记/PPT 这类任务需要全局视野，top-5 检索片段（约2500字）只见树木不见森林。而辅导问答是局部精准问答，走检索反而更快更省。所以我按任务特性分了两条通路，learn 意图直达全文 agent 跳过冗余检索。

**Q: 多文档全文不会超上下文吗？**
A: 我做了截断保护。`_load_full_markdown` 默认上限 60 万字（≈30 万 tokens，qwen3.7-plus 上下文 1M）。超限时按篇均摊，每篇保留开头+结尾各一半，保证每篇都能被看到，同时打明确标记让 LLM 知道是节选。

**Q: Send 并行怎么实现的？**
A: 路由函数返回 `[Send("node", sub_state), ...]` 列表，LangGraph 见到 Send 列表会为每个启动一个并行节点执行。结果通过 `Annotated[list, operator.add]` reducer 自动汇聚。这是 map-reduce 模式。

**Q: 思考模型流式那个坑怎么发现的？**
A: 用户报"点出题按钮返回空"。我写了 5 个探针逐层定位：直接打 API（确认是思考模型）、独立 LangChain（确认 content 能收到）、完整 graph（确认路由对）、复刻 httpx 流（确认 50 秒首 token）。根因是 45 秒思考静默期前端流超时。修复用 enable_thinking 开关，首 token 从 45 秒降到 1 秒。

**Q: 怎么解决并发限流？**
A: section_worker 加了 retry + 指数退避（2s/4s/6s/8s），单章节全部失败则降级为占位文本，不影响其他章节和整体综述。

"""学习助手页面 —— 上传课件/知识文档，辅助日常学习

六种模式（复用后端的 上传→MinerU→分块→向量化→RAG 检索 链路）：
- 📖 辅导问答：苏格拉底式讲解（基于课件答 + 反问/举例引导）
- 📋 要点总结：一键生成结构化大纲摘要
- 🗂️ 知识卡片：抽取概念生成 Anki 风格闪卡
- 📝 自测练习：基于课件出选择题并可交互作答
- 📒 复习笔记：知识框架 + 易错点 + 记忆口诀，适合考前复习
- 📽️ PPT 生成：Marp 格式 Markdown，可一键导出 PPT

提示：课件/知识文档和论文走同一条入库路径，先在「论文管理」页面上传，
处理完成（status=indexed）后在这里选择它即可。
"""
import json
import time
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api_client import (
    list_documents,
    learn_qa, learn_summary, learn_flashcard, learn_quiz,
    learn_notes, learn_slides,
    generate_outline,
)


def _collect_stream(stream, key="token"):
    """把流式 token 拼成完整文本，同时捕获 intent/error。返回 (full_text, intent)"""
    full = ""
    intent = ""
    for event in stream:
        t = event.get("type")
        if t == "intent":
            intent = event.get("intent", "")
        elif t == "token":
            full += event.get("content", "")
        elif t == "error":
            st.error(f"错误：{event.get('msg')}")
            return None, intent
    return full, intent


def _extract_json_candidates(text: str):
    """从 LLM 文本里尽量多地捞出可能是 JSON 数组/对象的片段，按可靠度排序返回。"""
    if not text:
        return []
    candidates = []

    # 策略 1：剥离 markdown 代码块，取最长一段
    if "```" in text:
        for chunk in text.split("```"):
            c = chunk.strip()
            # 去掉语言标记 json / jsonc 等
            if c[:5].lower().startswith("json"):
                c = c[4:].lstrip()
            if c and (c[0] in "[{"):
                candidates.append(c)

    # 策略 2：直接把整段去空白后当作候选
    stripped = text.strip()
    if stripped and (stripped[0] in "[{"):
        candidates.append(stripped)

    # 策略 3：用括号配平提取所有 [..] 与 {..} 片段
    pairs = {"]": "[", "}": "{"}
    opens = set("[{")
    for i, ch in enumerate(text):
        if ch not in opens:
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(text)):
            cj = text[j]
            if in_str:
                if esc:
                    esc = False
                elif cj == "\\":
                    esc = True
                elif cj == '"':
                    in_str = False
                continue
            if cj == '"':
                in_str = True
            elif cj in opens:
                depth += 1
            elif cj in pairs:
                depth -= 1
                if depth == 0:
                    candidates.append(text[i:j + 1])
                    break

    # 去重保序
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    # 按长度从长到短排（越完整越优先），但保持同长度原序
    uniq.sort(key=len, reverse=True)
    return uniq


def _try_repair_json(s: str):
    """对常见 LLM 产生的非法 JSON 做修复尝试，返回修复后的字符串。"""
    if not s:
        return s
    # 1) 去掉行尾/数组元素间的悬挂逗号：,] 或 ,}（允许中间空白）
    import re
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # 2) 把字符串值里裸露的单引号成对替换为双引号（保守起见只处理像 'key':'val' 这种）
    # 不做激进转换，避免破坏英文撇号
    # 3) 去掉控制字符（除 \n \t）
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    return s


def _parse_json_lenient(text: str):
    """容错解析 LLM 输出的 JSON 数组/对象。支持代码块包裹、对象包裹、
    悬挂逗号、前后多余文字、嵌套结构等多种异常。返回 python 对象或 None。"""
    if not text:
        return None

    candidates = _extract_json_candidates(text)

    for raw in candidates:
        for attempt in (raw, _try_repair_json(raw)):
            if not attempt:
                continue
            try:
                data = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            # 直接就是数组
            if isinstance(data, list):
                return data
            # 对象包裹：取第一个 list 值
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return v
                # 单对象也接受（少见但宽容）
                if data:
                    return [data]
    return None


def _format_quiz_markdown(quizzes) -> str:
    """把 quiz JSON 格式化成便于打印/复习的 Markdown 文档。
    正确答案单独放在文末，方便先做题后对答案。"""
    if not quizzes:
        return ""
    lines = ["# 自测练习", "", f"> 共 {len(quizzes)} 道题，先作答再看文末答案。", ""]
    letters = ["A", "B", "C", "D", "E", "F"]
    answers = []
    for i, q in enumerate(quizzes, 1):
        question = q.get("question", "")
        options = q.get("options", [])
        lines.append(f"## {i}. {question}")
        lines.append("")
        for k, opt in enumerate(options):
            label = letters[k] if k < len(letters) else str(k)
            lines.append(f"- **{label}.** {opt}")
        lines.append("")
        answers.append((i, str(q.get("answer", "")).strip().upper(), q.get("explanation", "")))
    lines.append("---")
    lines.append("")
    lines.append("## 答案与解析")
    lines.append("")
    for i, ans, exp in answers:
        lines.append(f"**{i}. {ans}**　{exp}")
        lines.append("")
    return "\n".join(lines)


def _format_flashcard_markdown(cards) -> str:
    """把闪卡 JSON 格式化成便于复习的 Markdown（问答对照表）。"""
    if not cards:
        return ""
    lines = ["# 知识卡片", "", f"> 共 {len(cards)} 张。", ""]
    for i, c in enumerate(cards, 1):
        front = c.get("front", "")
        back = c.get("back", "")
        tag = c.get("tag", "")
        lines.append(f"## {i}. {front}")
        if tag:
            lines.append(f"`{tag}`")
        lines.append("")
        lines.append(f"> {back}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


st.title("🎓 学习助手")
st.markdown(
    "上传**课件 / 知识文档 / 讲义**（在「论文管理」页面上传，处理完成后回到这里），"
    "AI 会基于你的资料辅助学习：辅导问答、要点总结、知识卡片、自测练习、复习笔记、PPT 生成。"
)

# ── 选文档（支持多选）──
result = list_documents()
papers = result.get("data", []) if result.get("code") == 200 else []
indexed_papers = [p for p in papers if p["status"] == "indexed"]

if not indexed_papers:
    st.warning("请先在「论文管理」页面 上传一个课件/知识文档，等待处理完成后再回来。")
    st.stop()

paper_options = {p["paper_id"]: p["title"] for p in indexed_papers}
selected_ids = st.multiselect(
    "选择资料（可多选，跨资料综合学习）",
    options=list(paper_options.keys()),
    format_func=lambda x: f"[{x}] {paper_options[x][:30]}",
    help="选多篇资料可综合多份课件内容生成笔记/PPT；只选一篇则深入单份资料。"
)

if not selected_ids:
    st.info("👆 请至少选择一份资料。")
    st.stop()

# 选中资料的标题列表（用于文件名、显示）
selected_titles = [paper_options[i] for i in selected_ids]

# thread_id 默认带时间戳，支持「新建对话」
_thread_key = "_".join(map(str, selected_ids))
if "learn_thread_id" not in st.session_state:
    st.session_state.learn_thread_id = f"learn_{_thread_key}_{int(time.time())}"

# 切换资料选择时换新对话
if st.session_state.get("learn_last_paper_key") != _thread_key:
    st.session_state.learn_last_paper_key = _thread_key
    st.session_state.learn_thread_id = f"learn_{_thread_key}_{int(time.time())}"

# ── 六种模式 Tab ──
tab_qa, tab_sum, tab_card, tab_quiz, tab_notes, tab_slides = st.tabs(
    ["📖 辅导问答", "📋 要点总结", "🗂️ 知识卡片", "📝 自测练习", "📒 复习笔记", "📽️ PPT 生成"]
)

# ── Tab 1: 辅导问答 ──
with tab_qa:
    if "learn_messages" not in st.session_state:
        st.session_state.learn_messages = []

    # 切换资料选择时清空聊天显示
    if st.session_state.get("learn_msg_paper_key") != _thread_key:
        st.session_state.learn_msg_paper_key = _thread_key
        st.session_state.learn_messages = []

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🆕 新建对话"):
            st.session_state.learn_messages = []
            st.session_state.learn_thread_id = f"learn_{_thread_key}_{int(time.time())}"
            st.rerun()
    with col2:
        if st.button("🗑️ 清空显示"):
            st.session_state.learn_messages = []
            st.rerun()

    for msg in st.session_state.learn_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("针对所选资料提问，AI 会辅导式讲解…"):
        st.session_state.learn_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            for event in learn_qa(prompt, selected_ids, st.session_state.learn_thread_id):
                t = event.get("type")
                if t == "token":
                    full += event.get("content", "")
                    placeholder.markdown(full + "▌")
                elif t == "done":
                    placeholder.markdown(full)
                elif t == "error":
                    placeholder.error(f"错误：{event.get('msg')}")
            st.caption(f"🎯 意图：learn")
        st.session_state.learn_messages.append({"role": "assistant", "content": full})

# ── Tab 2: 要点总结 ──
with tab_sum:
    if st.button("🚀 生成要点总结", type="primary"):
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("正在通读资料并提炼…"):
                full, _ = _collect_stream(
                    learn_summary(selected_ids, st.session_state.learn_thread_id)
                )
            if full is not None:
                placeholder.markdown(full)
                # 持久化，避免重跑脚本时（点任何控件）整段总结消失
                st.session_state["summary_result"] = full
        if full is None:
            st.stop()
    elif "summary_result" in st.session_state:
        # 复用已生成的总结（避免按钮变 False 后内容消失）
        with st.chat_message("assistant"):
            st.markdown(st.session_state["summary_result"])
    if "summary_result" in st.session_state:
        st.divider()
        st.download_button(
            "📥 下载要点总结 (.md)",
            data=st.session_state["summary_result"],
            file_name=f"要点总结_{'_'.join(selected_titles)[:30]}.md",
            mime="text/markdown",
        )

# ── Tab 3: 知识卡片 ──
with tab_card:
    if st.button("🗂️ 生成知识闪卡", type="primary"):
        with st.spinner("正在抽取知识点生成闪卡…"):
            raw, _ = _collect_stream(
                learn_flashcard(selected_ids, st.session_state.learn_thread_id)
            )
        if raw is None:
            st.stop()
        cards = _parse_json_lenient(raw)
        if not cards:
            if not raw or not raw.strip():
                st.error(
                    "后端没有返回任何内容（raw 为空）。常见原因：模型流式超时、"
                    "检索为空或后端报错。请打开后端终端查看日志，或重试一次。"
                )
            else:
                st.warning("未能解析出闪卡 JSON，原始输出如下，可手动查看：")
                st.code(raw, language="json")
            st.stop()
        st.session_state["flashcards"] = cards
        st.rerun()  # 持久化后重跑，清掉按钮按下态

    cards = st.session_state.get("flashcards")
    if cards:
        st.success(f"生成 {len(cards)} 张闪卡，点击「显示答案」可翻转 👇")
        # 卡片网格：每行 2 张
        for i in range(0, len(cards), 2):
            cols = st.columns(2)
            for j, card in enumerate(cards[i:i + 2]):
                with cols[j]:
                    front = card.get("front", "")
                    back = card.get("back", "")
                    tag = card.get("tag", "")
                    with st.container(border=True):
                        if tag:
                            st.caption(f"🏷️ {tag}")
                        st.markdown(f"**❓ {front}**")
                        if st.toggle("显示答案", key=f"flip_{i}_{j}"):
                            st.info(back)
        # 保存到本地
        st.divider()
        _card_md = _format_flashcard_markdown(cards)
        c_json, c_md = st.columns(2)
        with c_json:
            st.download_button(
                "下载 JSON",
                data=json.dumps(cards, ensure_ascii=False, indent=2),
                file_name=f"知识卡片_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        with c_md:
            st.download_button(
                "下载 Markdown",
                data=_card_md,
                file_name=f"知识卡片_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
        if st.button("🔄 重新生成（清空当前卡片）"):
            st.session_state.pop("flashcards", None)
            st.rerun()

# ── Tab 4: 自测练习 ──
with tab_quiz:
    # 题目用 session_state 持久化：否则点任何 radio/button 都会重跑脚本，
    # 「出题」按钮变回 False，整块题目就消失了。
    if st.button("📝 出一份自测题", type="primary"):
        with st.spinner("正在基于资料出题…"):
            raw, _ = _collect_stream(
                learn_quiz(selected_ids, st.session_state.learn_thread_id)
            )
        if raw is None:
            st.stop()
        quizzes = _parse_json_lenient(raw)
        if not quizzes:
            if not raw or not raw.strip():
                st.error(
                    "后端没有返回任何内容（raw 为空）。常见原因：模型流式超时、"
                    "检索为空或后端报错。请打开后端终端查看日志，或重试一次。"
                )
            else:
                st.warning("未能解析出题目 JSON，原始输出如下，可手动查看：")
                st.code(raw, language="json")
            st.stop()
        # 存进 session_state：题目、标准答案、每题作答状态、是否已提交
        st.session_state["quiz_questions"] = quizzes
        st.session_state["quiz_submitted"] = {i: False for i in range(len(quizzes))}
        st.session_state["quiz_choices"] = {}
        st.rerun()  # 清掉「出题」按钮的按下态，直接展示题目区

    quizzes = st.session_state.get("quiz_questions")
    if quizzes:
        st.success(f"已生成 {len(quizzes)} 道题，作答后可看答案与解析 👇")
        submitted = st.session_state.get("quiz_submitted", {})
        letters = ["A", "B", "C", "D", "E", "F"]
        score = 0
        answered = 0
        for idx, q in enumerate(quizzes):
            question = q.get("question", "")
            options = q.get("options", [])
            answer = str(q.get("answer", "")).strip().upper()
            explanation = q.get("explanation", "")
            opt_labels = [
                f"{letters[k]}. {opt}" if k < len(letters) else opt
                for k, opt in enumerate(options)
            ]
            with st.container(border=True):
                st.markdown(f"**{idx + 1}. {question}**")
                # 选项 radio：选择本身不再触发「提交」，只记录到 session_state
                prev_choice = st.session_state["quiz_choices"].get(idx)
                choice = st.radio(
                    "你的选择",
                    options=range(len(opt_labels)),
                    format_func=lambda k, ol=opt_labels: ol[k],
                    key=f"quiz_choice_{idx}",
                    index=prev_choice,
                    horizontal=False,
                )
                st.session_state["quiz_choices"][idx] = choice
                # 提交按钮：点了才判分，判分结果同样存 session_state，避免重跑后消失
                if st.button("提交答案", key=f"submit_{idx}"):
                    st.session_state["quiz_submitted"][idx] = True
                if submitted.get(idx):
                    answered += 1
                    picked_letter = letters[choice] if choice is not None else ""
                    if picked_letter == answer:
                        score += 1
                        st.success(f"✅ 正确！({answer})")
                    else:
                        st.error(f"❌ 错误，正确答案是 {answer}，你选了 {picked_letter or '空'}")
                    if explanation:
                        st.info(f"解析：{explanation}")
        if answered > 0:
            st.divider()
            st.metric("本次得分", f"{score} / {answered}")

        # —— 保存到本地，方便复习 ——
        st.divider()
        st.markdown("**📥 保存这份题目到本地**")
        _quiz_md = _format_quiz_markdown(quizzes)
        c_json, c_md = st.columns(2)
        with c_json:
            st.download_button(
                "下载 JSON",
                data=json.dumps(quizzes, ensure_ascii=False, indent=2),
                file_name=f"自测题_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        with c_md:
            st.download_button(
                "下载 Markdown",
                data=_quiz_md,
                file_name=f"自测题_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
        with st.expander("预览 Markdown"):
            st.code(_quiz_md, language="markdown")
        if st.button("🔄 重新出题（清空当前题目）"):
            st.session_state.pop("quiz_questions", None)
            st.session_state.pop("quiz_submitted", None)
            st.session_state.pop("quiz_choices", None)
            st.rerun()


# ── Tab 5: 复习笔记（分步向导：配置→大纲→生成）──
with tab_notes:
    st.caption("📒 基于**全文**生成考前复习笔记。先出大纲你确认，再生成成品，全程可调。")
    step = st.session_state.get("notes_step", 1)

    # Step 1：配置 + 生成大纲
    if step == 1:
        focus = st.text_input(
            "本次重点（可选）",
            value="",
            placeholder="如：重点记公式和易错点 / 重点记第3章",
            key="notes_focus_input",
        )
        detail = st.selectbox("详细程度", ["精简", "标准", "详尽"], index=1, key="notes_detail_input")
        if st.button("🪄 生成笔记大纲", type="primary"):
            if not selected_ids:
                st.warning("请先在左侧选择至少一篇文档。")
                st.stop()
            with st.spinner("正在通读全文并规划大纲…"):
                try:
                    outline = generate_outline(
                        selected_ids, "notes",
                        focus=focus, detail_level=detail,
                    )
                except Exception as e:
                    st.error(f"生成大纲失败：{e}")
                    st.stop()
            if not outline.strip():
                st.error("大纲为空，请重试或检查文档是否已解析完成。")
                st.stop()
            st.session_state["notes_outline"] = outline
            st.session_state["notes_focus"] = focus
            st.session_state["notes_detail"] = detail
            st.session_state["notes_step"] = 2
            st.rerun()

    # Step 2：编辑大纲
    if step == 2:
        st.success("✅ 大纲已生成，可自由增删改后再生成完整笔记。")
        edited = st.text_area(
            "笔记大纲（每行一个章节/主题，可编辑）",
            value=st.session_state.get("notes_outline", ""),
            height=280,
            key="notes_outline_edit",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 确认大纲，生成完整笔记", type="primary"):
                st.session_state["notes_outline_confirmed"] = edited
                st.session_state["notes_step"] = 3
                st.rerun()
        with c2:
            if st.button("🔄 重新生成大纲"):
                st.session_state["notes_step"] = 1
                st.rerun()

    # Step 3：流式生成成品 + 下载
    if step == 3:
        outline_confirmed = st.session_state.get("notes_outline_confirmed", "")
        focus = st.session_state.get("notes_focus", "")
        detail = st.session_state.get("notes_detail", "标准")
        # 只有「还没生成过」才真正跑 LLM；已有结果则直接复用（避免重跑重复生成）
        if "notes_result" not in st.session_state:
            st.info(f"📋 按确认的大纲生成中（详细程度：{detail}）…")
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full = ""
                for event in learn_notes(
                    selected_ids, st.session_state.learn_thread_id,
                    outline=outline_confirmed, detail_level=detail, focus=focus,
                ):
                    t = event.get("type")
                    if t == "token":
                        full += event.get("content", "")
                        placeholder.markdown(full + "▌")
                    elif t == "done":
                        placeholder.markdown(full)
                    elif t == "error":
                        placeholder.error(f"错误：{event.get('msg')}")
            # 持久化生成结果，后续重跑不再重新调 LLM
            st.session_state["notes_result"] = full
        else:
            full = st.session_state["notes_result"]
            with st.chat_message("assistant"):
                st.markdown(full)

        if full:
            st.divider()
            st.download_button(
                "📥 下载复习笔记 (.md)",
                data=full,
                file_name=f"复习笔记_{'_'.join(selected_titles)[:30]}.md",
                mime="text/markdown",
            )
        if st.button("✏️ 回去改大纲"):
            # 改大纲后清掉旧结果，确保重新生成
            st.session_state.pop("notes_result", None)
            st.session_state["notes_step"] = 2
            st.rerun()


# ── Tab 6: PPT 生成（分步向导：配置→大纲→生成，Marp 格式）──
with tab_slides:
    st.caption("📽️ 基于**全文**生成 Marp PPT。先选主题/页数出大纲，你确认后再生成。")
    with st.expander("❓ 如何把 Marp Markdown 变成 PPT？"):
        st.markdown("""
        **方法 1（推荐）：VS Code + Marp 插件**
        1. 安装 VS Code 扩展 **Marp for VS Code**
        2. 打开下载的 `.md` 文件
        3. 点击右上角「导出」按钮，选择 PDF / PPTX / HTML

        **方法 2：命令行**
        `npx @marp-team/marp-cli@latest 文件.md -o 输出.pptx`

        **方法 3：Marp Web**
        粘贴到 [marp.app](https://marp.app/) 在线预览和导出
        """)
    step = st.session_state.get("slides_step", 1)

    # Step 1：配置 + 生成大纲
    if step == 1:
        col_t, col_p = st.columns(2)
        with col_t:
            theme = st.selectbox("Marp 主题", ["default", "gaia", "uncover"], key="slides_theme_input")
        with col_p:
            page_count = st.number_input("页数", min_value=5, max_value=40, value=10, step=1, key="slides_pages_input")
        focus = st.text_input(
            "侧重点（可选）",
            value="",
            placeholder="如：偏架构，少讲背景 / 重点讲实验",
            key="slides_focus_input",
        )
        if st.button("🪄 生成 PPT 大纲", type="primary"):
            if not selected_ids:
                st.warning("请先在左侧选择至少一篇文档。")
                st.stop()
            with st.spinner("正在通读全文并规划 PPT 大纲…"):
                try:
                    outline = generate_outline(
                        selected_ids, "slides",
                        focus=focus, page_count=int(page_count), theme=theme,
                    )
                except Exception as e:
                    st.error(f"生成大纲失败：{e}")
                    st.stop()
            if not outline.strip():
                st.error("大纲为空，请重试或检查文档是否已解析完成。")
                st.stop()
            st.session_state["slides_outline"] = outline
            st.session_state["slides_theme"] = theme
            st.session_state["slides_pages"] = int(page_count)
            st.session_state["slides_focus"] = focus
            st.session_state["slides_step"] = 2
            st.rerun()

    # Step 2：编辑大纲
    if step == 2:
        st.success("✅ PPT 大纲已生成，可增删页、改标题后再生成。")
        edited = st.text_area(
            "PPT 大纲（一行一页：页码. 标题 — 内容说明）",
            value=st.session_state.get("slides_outline", ""),
            height=280,
            key="slides_outline_edit",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 确认大纲，生成完整 PPT", type="primary"):
                st.session_state["slides_outline_confirmed"] = edited
                st.session_state["slides_step"] = 3
                st.rerun()
        with c2:
            if st.button("🔄 重新生成大纲"):
                st.session_state["slides_step"] = 1
                st.rerun()

    # Step 3：流式生成成品 + 下载
    if step == 3:
        outline_confirmed = st.session_state.get("slides_outline_confirmed", "")
        theme = st.session_state.get("slides_theme", "default")
        pages = st.session_state.get("slides_pages", 10)
        focus = st.session_state.get("slides_focus", "")
        # 只有「还没生成过」才真正跑 LLM；已有结果则直接复用（避免重跑重复生成）
        if "slides_result" not in st.session_state:
            st.info(f"📋 按确认的大纲生成中（主题：{theme}，约 {pages} 页）…")
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full = ""
                for event in learn_slides(
                    selected_ids, st.session_state.learn_thread_id,
                    outline=outline_confirmed, theme=theme, page_count=pages, focus=focus,
                ):
                    t = event.get("type")
                    if t == "token":
                        full += event.get("content", "")
                        placeholder.markdown(full + "▌")
                    elif t == "done":
                        placeholder.markdown(full)
                    elif t == "error":
                        placeholder.error(f"错误：{event.get('msg')}")
            # 持久化生成结果，后续重跑不再重新调 LLM
            st.session_state["slides_result"] = full
        else:
            full = st.session_state["slides_result"]
            with st.chat_message("assistant"):
                st.markdown(full)

        if full:
            st.divider()
            st.download_button(
                "📥 下载 PPT Markdown (.md)",
                data=full,
                file_name=f"PPT_{'_'.join(selected_titles)[:30]}.md",
                mime="text/markdown",
            )
            st.info("💡 下载后用 VS Code Marp 插件预览/导出为 PPTX（见上方说明）")
        if st.button("✏️ 回去改大纲"):
            st.session_state.pop("slides_result", None)
            st.session_state["slides_step"] = 2
            st.rerun()

"""Streamlit 界面：streamlit run app.py"""

import html
import re
import time
import uuid
from pathlib import Path

import streamlit as st

import config
import rag

MODES = {"rag": "RAG", "plain": "无 RAG", "compare": "对比"}
PROJECT_COLORS = {"Obsidian": "violet", "Dataview": "blue", "Tasks": "green"}
PROJECT_LABELS = {
    "Obsidian": "Obsidian 官方帮助",
    "Dataview": "Dataview 官方文档",
    "Tasks": "Tasks 官方文档",
}
PROJECT_ICONS = {"Obsidian": "🟣", "Dataview": "🔵", "Tasks": "🟢"}
EXAMPLES = [
    ("Obsidian", "如何链接到另一篇笔记里的某个标题？"),
    ("Dataview", "怎么列出带 #book 标签的笔记，并按评分从高到低排序？"),
    ("Tasks", "如何查询未来 7 天内到期且未完成的任务？"),
    ("Obsidian", "Properties 支持哪些属性类型？"),
    ("Dataview", "TABLE 和 LIST 两种查询有什么区别？"),
    ("Tasks", "怎么创建每周一重复的任务？"),
]
LLM_NAME = Path(config.QWEN_MODEL_PATH).name

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.hero { margin-bottom: 1.2rem; }
.hero-title {
    font-size: 2.1rem; font-weight: 750; letter-spacing: -0.02em; line-height: 1.2;
    background: linear-gradient(90deg, #6D28D9 0%, #A855F7 60%, #EC4899 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: #6B6680; font-size: 0.98rem; margin-top: 0.35rem; }
.pipeline { color: #6B6680; font-size: 0.82rem; line-height: 1.9; }
.pipeline b { color: #6D28D9; font-weight: 600; }
.panel-label { font-size: 0.8rem; font-weight: 650; letter-spacing: 0.04em; margin-bottom: 0.3rem; }
.panel-label.rag { color: #6D28D9; }
.panel-label.plain { color: #8A8599; }
.source-kicker { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.35rem; }
.source-number {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 1.65rem; height: 1.65rem; padding: 0 0.4rem; border-radius: 999px;
    color: white; background: #7C3AED; font-size: 0.76rem; font-weight: 750;
}
.source-project {
    color: #6D28D9; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.025em;
}
.source-title { font-size: 1.08rem; font-weight: 720; line-height: 1.35; }
.source-section { color: #777184; font-size: 0.84rem; margin-top: 0.28rem; }
.source-score { text-align: center; padding: 0.5rem 0.25rem; }
.source-score-value { color: #6D28D9; font-size: 1.35rem; font-weight: 760; line-height: 1.1; }
.source-score-label { color: #8A8599; font-size: 0.7rem; margin-top: 0.2rem; }
.source-meta {
    color: #6B6680; font-size: 0.78rem; line-height: 1.7;
    padding: 0.5rem 0.7rem; border-radius: 0.45rem;
    background: rgba(128, 128, 128, 0.07); margin: 0.45rem 0 0.65rem;
}
.source-context {
    border: 1px solid rgba(128, 128, 128, 0.22); border-radius: 0.55rem;
    overflow: hidden; margin-top: 0.55rem; background: rgba(128, 128, 128, 0.04);
}
.source-context pre {
    margin: 0; padding: 0.7rem 0.85rem; white-space: pre-wrap; overflow-wrap: anywhere;
    font-size: 0.82rem; line-height: 1.55; font-family: var(--font-monospace);
}
.source-context .context-muted { color: #777184; }
.source-context .context-hit {
    color: inherit; background: rgba(168, 85, 247, 0.14);
    border-top: 1px solid rgba(168, 85, 247, 0.25);
    border-bottom: 1px solid rgba(168, 85, 247, 0.25);
}
</style>
"""

st.set_page_config(
    page_title="Obsidian 文档助手",
    page_icon=":material/menu_book:",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)


# ---------- 资源加载（整个应用只加载一次） ----------


@st.cache_resource(show_spinner="正在加载向量库和 Embedding 模型…")
def load_index() -> dict:
    stats = rag.index_stats()
    rag.get_vector_store()
    return stats


@st.cache_resource(show_spinner=f"正在加载 {LLM_NAME}…")
def load_llm() -> bool:
    rag.get_llm()
    return True


index_stats, index_error, llm_error = None, None, None
try:
    index_stats = load_index()
except (rag.IndexNotReady, rag.ModelNotReady) as error:
    index_error = str(error)
try:
    load_llm()
except Exception as error:  # 路径错误、显存不足等都直接显示在界面上
    llm_error = str(error)


# ---------- 渲染 ----------

CODE_BLOCK = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)
CITATION = re.compile(r"\[(\d+)\](?!\()")


def source_anchor(source: rag.Source, anchor_prefix: str) -> str:
    return f"source-{anchor_prefix}-{source.chunk_id[:12]}"


def linkify_citations(
    text: str, sources: list[rag.Source], anchor_prefix: str
) -> str:
    """把回答里的 [1] 变成页面内证据链接；代码里的内容不动。"""
    urls = {
        source.index: f"#{source_anchor(source, anchor_prefix)}"
        for source in sources
    }

    def replace(match: re.Match) -> str:
        number = int(match.group(1))
        if number not in urls:
            return match.group(0)
        return f"[\\[{number}\\]]({urls[number]})"

    parts = CODE_BLOCK.split(text)
    return "".join(
        part if index % 2 else CITATION.sub(replace, part)
        for index, part in enumerate(parts)
    )


def render_answer(turn: dict, kind: str, live: bool) -> None:
    """kind 为 'rag' 或 'plain'。live=True 时现场生成并流式显示。"""
    use_rag = kind == "rag"
    label = "RAG · 基于官方文档回答" if use_rag else f"无 RAG · {LLM_NAME} 直接回答"
    st.markdown(
        f'<div class="panel-label {kind}">{label}</div>', unsafe_allow_html=True
    )

    if live and kind not in turn["errors"]:
        slot = st.empty()
        started = time.perf_counter()
        try:
            messages = rag.build_messages(
                turn["question"], turn["sources"] if use_rag else None
            )
            with slot.container():
                turn[kind] = st.write_stream(rag.generate_stream(messages))
            turn["timing"][kind] = time.perf_counter() - started
        except rag.ModelNotReady as error:
            turn["errors"][kind] = f"生成模型未就绪：{error}"
        except Exception as error:
            turn["errors"][kind] = f"生成失败：{error}"
        slot.empty()

    if kind in turn["errors"]:
        st.warning(turn["errors"][kind])
        return

    answer = turn[kind] or ""
    st.markdown(
        linkify_citations(answer, turn["sources"], turn["id"])
        if use_rag
        else answer
    )
    if use_rag:
        st.caption(
            f"检索 {len(turn['sources'])} 个片段用时 {turn['timing']['retrieve']:.2f} 秒"
            f" · 生成用时 {turn['timing'][kind]:.1f} 秒"
        )
    else:
        st.caption(f"未检索文档 · 生成用时 {turn['timing'][kind]:.1f} 秒")


def render_sources(sources: list[rag.Source], anchor_prefix: str) -> None:
    if not sources:
        return
    st.markdown(f"**参考来源**（{len(sources)}）")
    for source in sources:
        anchor = source_anchor(source, anchor_prefix)
        st.markdown(f'<span id="{anchor}"></span>', unsafe_allow_html=True)
        project_label = PROJECT_LABELS.get(source.project, source.project)
        icon = PROJECT_ICONS.get(source.project, "📄")
        section_parts = [part.strip() for part in source.section.split(" > ")]
        current_section = section_parts[-1] if section_parts else source.title
        relevance = round(max(0.0, min(1.0, source.score)) * 100)
        with st.container(border=True):
            title_column, score_column = st.columns(
                [5, 1], gap="small", vertical_alignment="center"
            )
            with title_column:
                st.markdown(
                    '<div class="source-kicker">'
                    f'<span class="source-number">{source.index}</span>'
                    f'<span class="source-project">{icon} '
                    f'{html.escape(project_label)}</span>'
                    "</div>"
                    f'<div class="source-title">{html.escape(source.title)}</div>'
                    f'<div class="source-section">当前章节 · '
                    f'{html.escape(current_section)}</div>',
                    unsafe_allow_html=True,
                )
            with score_column:
                st.markdown(
                    '<div class="source-score">'
                    f'<div class="source-score-value">{relevance}%</div>'
                    '<div class="source-score-label">检索相关度</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="source-meta">'
                f'📍 原文位置：{html.escape(source.source)}，'
                f'第 {source.start_line}–{source.end_line} 行<br>'
                f'📚 章节路径：{html.escape(source.section)}'
                "</div>",
                unsafe_allow_html=True,
            )
            st.link_button(
                "打开原文准确位置",
                source.url,
                icon=":material/open_in_new:",
            )
            with st.expander("查看命中原文与上下文"):
                before = html.escape(source.context_before)
                hit = html.escape(source.raw_content)
                after = html.escape(source.context_after)
                st.markdown(
                    '<div class="source-context">'
                    f'<pre class="context-muted">{before}</pre>'
                    f'<pre class="context-hit">{hit}</pre>'
                    f'<pre class="context-muted">{after}</pre>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"上下文范围：第 {source.context_start_line}–"
                    f"{source.context_end_line} 行；紫色区域为实际命中原文"
                )


def render_turn(turn: dict, live: bool = False) -> None:
    turn.setdefault("id", uuid.uuid4().hex[:10])
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant"):
        mode = turn["mode"]

        if live and mode != "plain":
            started = time.perf_counter()
            try:
                with st.spinner("正在检索官方文档…"):
                    turn["sources"] = rag.retrieve(
                        turn["question"], turn["projects"], turn["k"]
                    )
                turn["timing"]["retrieve"] = time.perf_counter() - started
            except (rag.IndexNotReady, rag.ModelNotReady) as error:
                turn["errors"]["rag"] = f"无法检索：{error}"

        if mode == "compare":
            left, right = st.columns(2, gap="medium")
            with left.container(border=True):
                render_answer(turn, "plain", live)
            with right.container(border=True):
                render_answer(turn, "rag", live)
        else:
            render_answer(turn, mode, live)

        render_sources(turn["sources"], turn["id"])


# ---------- 侧边栏 ----------

with st.sidebar:
    st.markdown("### 设置")
    mode = st.segmented_control(
        "回答模式",
        options=list(MODES),
        format_func=MODES.get,
        default="compare",
        required=True,
        help="对比模式会让模型分别在不检索和检索文档的情况下回答同一个问题。",
    )
    projects = st.pills(
        "数据源",
        options=config.PROJECTS,
        selection_mode="multi",
        default=config.PROJECTS,
        help="只在选中的文档里检索。",
    )
    if not projects:
        st.caption("未选择数据源，将检索全部文档。")
    top_k = st.slider("检索片段数", min_value=3, max_value=8, value=config.TOP_K)

    st.divider()
    st.markdown("### 系统状态")
    if index_stats:
        counts = " · ".join(
            f"{project} {count}" for project, count in index_stats["per_project"].items()
        )
        st.markdown(
            '<div class="pipeline">'
            f'官方 Markdown 文档<br>→ <b>{index_stats["total"]}</b> 个文本块<br>'
            f'<span style="font-size:0.76rem">（{counts}）</span><br>'
            f'→ <b>{index_stats["embedding_model"]}</b> 向量化<br>'
            "→ <b>ChromaDB</b> 相似度检索<br>"
            f"→ <b>{LLM_NAME}</b> 生成带引用的回答"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.error(f"向量库未就绪：{index_error}")
    if llm_error:
        st.warning(f"生成模型未就绪：{llm_error}")
    else:
        st.caption(f"{LLM_NAME} 已加载（{config.resolve_device()}）")

    st.divider()
    if st.button("清空对话", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.turns = []
        st.rerun()


# ---------- 主区域 ----------

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">Obsidian 文档助手</div>'
    '<div class="hero-sub">从 Obsidian、Dataview、Tasks 的官方文档中检索依据，'
    f"再由本地 {LLM_NAME} 生成带来源引用的回答</div>"
    "</div>",
    unsafe_allow_html=True,
)

st.session_state.setdefault("turns", [])


def ask_example(question: str) -> None:
    st.session_state.pending_question = question


# chat_input 固定在页面底部，先取到问题，才能决定是否还显示示例问题
typed = st.chat_input("问一个关于 Obsidian、Dataview 或 Tasks 的问题")
question = typed or st.session_state.pop("pending_question", None)

if not st.session_state.turns and not question:
    st.markdown("##### 试试这些问题")
    columns = st.columns(3)
    for position, (project, example) in enumerate(EXAMPLES):
        columns[position % 3].button(
            f":{PROJECT_COLORS[project]}[**{project}**]  \n{example}",
            key=f"example-{position}",
            on_click=ask_example,
            args=(example,),
            width="stretch",
        )

for turn in st.session_state.turns:
    render_turn(turn)

if question:
    turn = {
        "question": question.strip(),
        "mode": mode,
        "projects": list(projects),
        "k": top_k,
        "plain": None,
        "rag": None,
        "sources": [],
        "timing": {},
        "errors": {},
    }
    render_turn(turn, live=True)
    st.session_state.turns.append(turn)

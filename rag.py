"""RAG 核心：向量检索 + 提示词构造 + 本地 Qwen 流式生成。

命令行自测（不需要启动网页）：
    python rag.py "如何查询未来 7 天内到期的任务？"
    python rag.py --no-rag "如何查询未来 7 天内到期的任务？"
"""

import argparse
import hashlib
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import config


class IndexNotReady(RuntimeError):
    """向量库不存在、为空，或与当前 Embedding 模型不匹配。"""


class ModelNotReady(RuntimeError):
    """模型路径还是占位符，或者指向的目录不存在。"""


@dataclass
class Source:
    index: int  # 回答中引用的编号，从 1 开始
    project: str
    title: str
    section: str
    url: str
    source: str
    start_line: int
    end_line: int
    chunk_id: str
    section_id: str
    doc_chunk_index: int
    doc_chunk_end_index: int
    content: str
    raw_content: str
    context_before: str
    context_after: str
    context_start_line: int
    context_end_line: int
    score: float  # 余弦相似度，越接近 1 越相关


RAG_SYSTEM_PROMPT = """你是 Obsidian 及其社区插件 Dataview、Tasks 的使用助手。请严格依据下面的参考资料回答用户的问题。

规则：
1. 只使用参考资料中的信息，资料里没有的内容不要补充，也不要猜测。
2. 如果参考资料不足以回答问题，直接说明“根据现有文档无法确定”，并说明缺少哪方面的信息。
3. 每个结论后面用 [1]、[2] 这样的编号标注出处，编号对应参考资料的序号。
4. 语法、命令、函数名和设置项名称保留英文原文，示例代码放进代码块。
5. 使用与用户提问相同的语言回答，条理清晰，简明扼要。

参考资料：
{context}"""

PLAIN_SYSTEM_PROMPT = (
    "你是 Obsidian 及其社区插件 Dataview、Tasks 的使用助手。"
    "请回答用户的问题，使用与用户提问相同的语言，条理清晰，简明扼要。"
)


def _check_local_path(path: str, name: str) -> None:
    """本地路径必须存在；不像路径的字符串（如 BAAI/bge-m3）当作 HuggingFace 模型名。"""
    looks_local = path.startswith(("/", ".", "~"))
    if looks_local and not Path(path).expanduser().exists():
        raise ModelNotReady(f"{name} 指向的路径不存在：{path}。请在 .env 中填写实际路径。")


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    _check_local_path(config.EMBEDDING_MODEL_PATH, "EMBEDDING_MODEL_PATH")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_PATH,
        model_kwargs={"device": config.resolve_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb

    hint = "请先运行 python ingest.py 建立向量库。"
    if not Path(config.CHROMA_PATH).exists():
        raise IndexNotReady(f"向量库目录不存在：{config.CHROMA_PATH}。{hint}")

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
    except Exception as error:
        raise IndexNotReady(f"向量库中没有集合 {config.COLLECTION_NAME}。{hint}") from error
    if collection.count() == 0:
        raise IndexNotReady(f"向量库是空的。{hint}")

    indexed_with = (collection.metadata or {}).get("embedding_model")
    current = Path(config.EMBEDDING_MODEL_PATH).name
    if indexed_with and indexed_with != current:
        raise IndexNotReady(
            f"向量库是用 {indexed_with} 建立的，当前 Embedding 模型是 {current}，"
            "两者必须一致。请重新运行 python ingest.py。"
        )
    schema_version = (collection.metadata or {}).get("schema_version")
    if schema_version != config.INDEX_SCHEMA_VERSION:
        raise IndexNotReady(
            f"向量库结构版本是 {schema_version or '旧版'}，当前代码需要版本 "
            f"{config.INDEX_SCHEMA_VERSION}。请重新运行 python ingest.py。"
        )
    return client, collection


@lru_cache(maxsize=1)
def get_vector_store():
    from langchain_chroma import Chroma

    client, _ = _get_collection()
    return Chroma(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        create_collection_if_not_exists=False,
    )


def index_stats() -> dict:
    """每个数据源的文本块数量，供界面展示。"""
    _, collection = _get_collection()
    per_project = {
        project: len(collection.get(where={"project": project}, include=[])["ids"])
        for project in config.PROJECTS
    }
    return {
        "total": collection.count(),
        "per_project": per_project,
        "embedding_model": (collection.metadata or {}).get("embedding_model", "未知"),
    }


@lru_cache(maxsize=128)
def _read_source_lines(project: str, relative_source: str) -> tuple[str, ...]:
    """读取用于入库的原始 Markdown；路径只能落在对应数据仓库内。"""
    source_config = next(
        (item for item in config.SOURCES if item["project"] == project), None
    )
    if source_config is None:
        return ()
    repo_dir = (config.DATA_DIR / source_config["dir"]).resolve()
    path = (repo_dir / relative_source).resolve()
    if path != repo_dir and repo_dir not in path.parents:
        return ()
    try:
        return tuple(
            path.read_text(encoding="utf-8", errors="ignore")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .split("\n")
        )
    except OSError:
        return ()


def _source_excerpt(
    project: str, relative_source: str, start_line: int, end_line: int
) -> tuple[str, str, str, int, int]:
    """返回命中原文、前后文，以及整段上下文的起止行号。"""
    lines = _read_source_lines(project, relative_source)
    if not lines:
        return "", "", "", start_line, end_line

    hit_start = max(0, start_line - 1)
    hit_end = min(len(lines), end_line)
    context_start = max(0, hit_start - config.SOURCE_CONTEXT_LINES)
    context_end = min(len(lines), hit_end + config.SOURCE_CONTEXT_LINES)
    return (
        "\n".join(lines[hit_start:hit_end]),
        "\n".join(lines[context_start:hit_start]),
        "\n".join(lines[hit_end:context_end]),
        context_start + 1,
        context_end,
    )


def _merge_adjacent_sources(sources: list[Source], limit: int) -> list[Source]:
    """合并同一小节内连续命中的块，避免重叠内容占用多个引用编号。"""
    buckets: dict[tuple[str, str], list[Source]] = {}
    for source in sources:
        buckets.setdefault((source.source, source.section_id), []).append(source)

    groups: list[list[Source]] = []
    for bucket in buckets.values():
        for source in sorted(bucket, key=lambda item: item.doc_chunk_index):
            if (
                groups
                and groups[-1][0].source == source.source
                and groups[-1][0].section_id == source.section_id
                and source.doc_chunk_index <= groups[-1][-1].doc_chunk_end_index + 1
            ):
                groups[-1].append(source)
            else:
                groups.append([source])

    merged: list[Source] = []
    for members in groups:
        best = max(members, key=lambda item: item.score)
        if len(members) == 1:
            merged.append(best)
            continue

        start_line = min(item.start_line for item in members)
        end_line = max(item.end_line for item in members)
        raw, before, after, context_start, context_end = _source_excerpt(
            best.project, best.source, start_line, end_line
        )
        base_url = best.url.split("#L", 1)[0]
        combined_ids = "|".join(sorted(item.chunk_id for item in members))
        merged.append(
            Source(
                index=0,
                project=best.project,
                title=best.title,
                section=best.section,
                url=f"{base_url}#L{start_line}-L{end_line}",
                source=best.source,
                start_line=start_line,
                end_line=end_line,
                chunk_id=hashlib.sha1(combined_ids.encode()).hexdigest(),
                section_id=best.section_id,
                doc_chunk_index=min(item.doc_chunk_index for item in members),
                doc_chunk_end_index=max(
                    item.doc_chunk_end_index for item in members
                ),
                content=raw or "\n\n".join(item.content for item in members),
                raw_content=raw or "\n\n".join(
                    item.raw_content for item in members
                ),
                context_before=before,
                context_after=after,
                context_start_line=context_start,
                context_end_line=context_end,
                score=best.score,
            )
        )

    merged.sort(key=lambda item: item.score, reverse=True)
    merged = merged[:limit]
    for index, source in enumerate(merged, start=1):
        source.index = index
    return merged


def retrieve(
    question: str, projects: list[str] | None = None, k: int = config.TOP_K
) -> list[Source]:
    """检索与问题最相关的 k 个文本块；projects 用于限定数据源。"""
    search_filter = None
    if projects and set(projects) != set(config.PROJECTS):
        search_filter = {"project": {"$in": list(projects)}}

    # 多取一些候选，合并重叠/相邻块后仍尽量保留 k 个独立证据组。
    results = get_vector_store().similarity_search_with_score(
        question, k=max(k * 2, k), filter=search_filter
    )

    sources = []
    for index, (document, distance) in enumerate(results, start=1):
        # 入库时每块的第一行是 "[项目 docs: 章节]"，展示时去掉，避免和标题重复
        content = document.page_content.split("\n", 1)[-1]
        metadata = document.metadata
        start_line = int(metadata["start_line"])
        end_line = int(metadata["end_line"])
        line_url = f"{metadata['url']}#L{start_line}-L{end_line}"
        raw_content, context_before, context_after, context_start, context_end = (
            _source_excerpt(
                metadata["project"], metadata["source"], start_line, end_line
            )
        )
        sources.append(
            Source(
                index=index,
                project=metadata["project"],
                title=metadata["title"],
                section=metadata["section"],
                url=line_url,
                source=metadata["source"],
                start_line=start_line,
                end_line=end_line,
                chunk_id=metadata["chunk_id"],
                section_id=metadata["section_id"],
                doc_chunk_index=int(metadata["doc_chunk_index"]),
                doc_chunk_end_index=int(metadata["doc_chunk_index"]),
                content=content,
                raw_content=raw_content or content,
                context_before=context_before,
                context_after=context_after,
                context_start_line=context_start,
                context_end_line=context_end,
                score=1.0 - distance,  # 集合使用余弦距离
            )
        )
    return _merge_adjacent_sources(sources, k)


def build_messages(question: str, sources: list[Source] | None = None) -> list[dict]:
    """sources 为 None 表示不使用 RAG，让模型直接回答（用于对比）。"""
    if sources is None:
        system = PLAIN_SYSTEM_PROMPT
    else:
        context = "\n\n".join(
            f"[{source.index}] {source.project} 文档 - {source.section}\n{source.content}"
            for source in sources
        )
        system = RAG_SYSTEM_PROMPT.format(context=context or "（没有检索到相关资料）")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


@lru_cache(maxsize=1)
def get_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _check_local_path(config.QWEN_MODEL_PATH, "QWEN_MODEL_PATH")
    device = config.resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(config.QWEN_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.QWEN_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else device,
    )
    model.eval()
    return tokenizer, model


def generate_stream(messages: list[dict]) -> Iterator[str]:
    """流式生成回答，逐段产出文本。"""
    from transformers import TextIteratorStreamer

    tokenizer, model = get_llm()
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # 关闭 Qwen3 的思考模式，响应更快
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    errors: list[BaseException] = []

    def run() -> None:
        try:
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
            )
        except BaseException as error:  # 例如显存不足
            errors.append(error)
            streamer.end()  # 否则下面的 for 循环会一直等待

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    for piece in streamer:
        if piece:
            yield piece
    thread.join()
    if errors:
        raise errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="命令行自测 RAG 流程")
    parser.add_argument("question")
    parser.add_argument("--no-rag", action="store_true", help="不检索，模型直接回答")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    sources = None
    if not args.no_rag:
        sources = retrieve(args.question, k=args.k)
        print("检索结果：")
        for source in sources:
            print(f"  [{source.index}] {source.score:.3f}  {source.project} - {source.section}")
        print()

    try:
        for piece in generate_stream(build_messages(args.question, sources)):
            print(piece, end="", flush=True)
        print()
    except ModelNotReady as error:
        raise SystemExit(f"无法生成回答：{error}")


if __name__ == "__main__":
    main()

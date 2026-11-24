"""集中管理配置：所有路径都从 .env / 环境变量读取。"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 模型：可以是本地目录，也可以是 HuggingFace 模型名
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/path/to/Qwen3-4B")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3")

# auto = 优先 cuda，其次 mps（Apple 芯片），最后 cpu
DEVICE = os.getenv("DEVICE", "auto")

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_PATH = str(os.getenv("CHROMA_PATH", BASE_DIR / "chroma_db"))
COLLECTION_NAME = "obsidian_docs"
INDEX_SCHEMA_VERSION = 2  # v2 开始保存原文行号、块序号和稳定 chunk ID

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "768"))
SOURCE_CONTEXT_LINES = int(os.getenv("SOURCE_CONTEXT_LINES", "8"))

# 三个数据源。docs 是仓库内的文档目录，branch 用于生成 GitHub 来源链接。
SOURCES = [
    {
        "project": "Obsidian",
        "repo": "obsidianmd/obsidian-help",
        "dir": "obsidian-help",
        "docs": "en",
        "branch": "master",
        # 面向文档贡献者的写作规范，不是软件使用说明
        "skip_dirs": {"Contributing to Obsidian"},
        "skip_files": set(),
    },
    {
        "project": "Dataview",
        "repo": "blacksmithgu/obsidian-dataview",
        "dir": "obsidian-dataview",
        "docs": "docs/docs",
        "branch": "master",
        "skip_dirs": set(),
        "skip_files": {"changelog.md", "friends.md"},
    },
    {
        "project": "Tasks",
        "repo": "obsidian-tasks-group/obsidian-tasks",
        "dir": "obsidian-tasks",
        "docs": "docs",
        "branch": "main",
        # 文档模板、更新日志、被其他页面嵌入的 HTML 片段
        "skip_dirs": {
            "_meta",
            ".obsidian",
            "What is New",
            "snippets-embedded-in-multiple-pages",
        },
        # 这两个文件在官方站点上标记为 publish: false
        "skip_files": {"README.md", "migration.md"},
    },
]

PROJECTS = [source["project"] for source in SOURCES]


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

import os
from typing import Optional

FILEAGENT_LOG_LEVEL = "INFO"
FILEAGENT_LOG_QUIET = "1"
FILEAGENT_UVICORN_LOG_LEVEL = "info"

os.environ["FILEAGENT_LOG_LEVEL"] = FILEAGENT_LOG_LEVEL
os.environ["FILEAGENT_LOG_QUIET"] = FILEAGENT_LOG_QUIET
os.environ["FILEAGENT_UVICORN_LOG_LEVEL"] = FILEAGENT_UVICORN_LOG_LEVEL

from utils.logger import get_logger
logger = get_logger()
def _parse_bool(raw: Optional[str], default: bool = False) -> bool:
    """
    Parse env bool with common variants.
    Treat unknown values as default (and warn).
    """
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    # Common typo
    if s == "flase":
        return False
    logger.warning(f"⚠️ 无法解析布尔环境变量值：{raw!r}，将按默认值 {default} 处理")
    return default


OFFLINE_MODE = False

DEV_NO_MODEL_LOAD = False
logger.info(f"DEV_NO_MODEL_LOAD={DEV_NO_MODEL_LOAD}")

if OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.info("🔌 离线模式已启用，模型将从本地缓存加载")
else:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HF_HOME

os.environ.setdefault("MODELSCOPE_DOMAIN", "www.modelscope.ai")

HOME_DIR = os.path.expanduser("~")
WATCH_DIR = HOME_DIR

DB_PATH = os.getenv("DB_PATH", "./chroma_db")


LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4o-mini"

LOCAL_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
LOCAL_LLM_MODEL = "local-model"
LOCAL_LLM_API_KEY = "not-needed"

_CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DATA_DIR = (os.getenv("FILEAGENT_DATA_DIR") or "").strip()
if _DATA_DIR.startswith("~"):
    _DATA_DIR = os.path.expanduser(_DATA_DIR)
if _DATA_DIR:
    _DATA_DIR = os.path.abspath(_DATA_DIR)

RERANKER_GGUF_FILE = (os.getenv("RERANKER_GGUF_FILE", "bge-reranker-v2-m3-Q5_K_M.gguf") or "").strip() or "bge-reranker-v2-m3-Q5_K_M.gguf"
EMBEDDING_GGUF_FILE = "bge-m3-Q8_0.gguf"

if _DATA_DIR:
    LOCAL_EMBEDDING_MODEL_PATH = os.path.join(_DATA_DIR, "models", EMBEDDING_GGUF_FILE)
    LOCAL_RERANKER_MODEL_PATH = os.path.join(_DATA_DIR, "models", RERANKER_GGUF_FILE)
else:
    LOCAL_EMBEDDING_MODEL_PATH = os.path.join(_CURRENT_DIR, "models", EMBEDDING_GGUF_FILE)
    LOCAL_RERANKER_MODEL_PATH = os.path.join(_CURRENT_DIR, "models", RERANKER_GGUF_FILE)

EMBEDDING_MODEL = "gpustack/bge-m3-GGUF"
EMBEDDING_REPO_ID = "gpustack/bge-m3-GGUF"
RERANKER_MODEL = "gpustack/bge-reranker-v2-m3-GGUF"
RERANKER_OPTIONAL = False

RELEVANCE_THRESHOLD = 0.0

VECTOR_SEARCH_TOP_K = 30

RERANK_TOP_K = 15

USE_WHITELIST_MODE = True

_include_paths_raw = os.getenv("INCLUDE_PATHS", "").strip()
if _include_paths_raw:
    INCLUDE_PATHS = {
        os.path.expanduser(p.strip())
        for p in _include_paths_raw.split(",")
        if p.strip()
    }
else:
    INCLUDE_PATHS = set()

IGNORE_TOP_LEVEL_DIRS = {
    "Library", "Applications", "anaconda3", "miniconda3", "opt", 
    "Public", "Music", "Movies", "Pictures"
}

IGNORE_PATTERNS = {
    "node_modules", "__pycache__", "target", "build", "dist", 
    ".git", ".vscode", ".cursor", ".idea", ".DS_Store", "site-packages",
    ".app",
    "venv", "env", ".env"
}

EXCLUDE_PATHS = {
    "Arduino",
    "node_modules",
}

EXCLUDE_FILENAMES = {
    "CMakeLists.txt",
    "Makefile",
    "Dockerfile",
    "Vagrantfile",
    "Procfile",          # Heroku
    "Brewfile",          # Homebrew
    
    "README.md",
    "README.txt",
    "CHANGELOG.md",
    "HISTORY.md",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "COPYING",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "AUTHORS",
    "AUTHORS.md",
    "MAINTAINERS",
    "CREDITS",
    "THANKS",
    "INSTALL",
    "INSTALL.md",
    "INSTALL.txt",
    
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "Gemfile",
    "Gemfile.lock",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "MANIFEST.in",
    "VERSION",
    
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    "CODEOWNERS",
    ".cz.yaml",
    
    # ===== Docker =====
    ".dockerignore",
    "docker-compose.yml",
    "docker-compose.yaml",
    
    ".editorconfig",
    ".prettierrc",
    ".prettierrc.json",
    ".prettierignore",
    ".eslintrc",
    ".eslintrc.json",
    ".eslintignore",
    ".stylelintrc",
    ".babelrc",
    
    ".coveragerc",
    "pytest.ini",
    "tox.ini",
    ".nycrc",
    "phpunit.xml",
    
    ".pylintrc",
    ".flake8",
    ".isort.cfg",
    ".rubocop.yml",
    ".clang-format",
    
    # ===== npm/yarn =====
    ".npmrc",
    ".npmignore",
    ".yarnrc",
    ".nvmrc",
    
    ".env.example",
    ".env.sample",
    ".env.template",
    "Thumbs.db",
    "ThirdPartyNotices.txt",
    "NOTICE",
    "NOTICE.txt",
    "NOTICE.md",
}

EXCLUDE_FILENAME_PREFIXES = {
    "requirements",      # requirements.txt, requirements-dev.txt
    "constraint",        # constraints.txt
    "pip-",              # pip-requirements.txt
    "dev-requirements",  # dev-requirements.txt
    "test-requirements", # test-requirements.txt
}

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".rtf",
    ".csv", ".tsv",
    ".yml", ".yaml",
    ".org", ".rst", ".tex",
    
    ".pdf",
    ".doc", ".docx",                        # Word
    ".xls", ".xlsx",                        # Excel
    ".ppt", ".pptx",                        # PowerPoint
    ".odt", ".ods", ".odp",                 # LibreOffice / OpenDocument
    
    ".pages", ".numbers", ".key",
    
    ".epub", ".mobi",
    
    ".eml", ".msg",
    
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".tiff", ".tif",
    ".heic", ".heif",                       # Mobile HEIF photos
    ".svg",
    
    ".mp3", ".wav", ".flac", ".aac", ".ogg",
    ".m4a", ".wma", ".aiff", ".ape",

    ".mp4", ".m4v", ".mov", ".avi", ".mkv",
    ".webm", ".flv", ".wmv",

    ".jsonl", ".xml", ".sql",
}


import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Any, Dict, List
from langchain_core.tools import tool
from config import settings
from .registry import ToolRegistry



_kb_instance: Optional[Any] = None

_active_paths: Optional[list] = None
_active_session_id: Optional[str] = None

_MAX_OPENED_FILES_PER_SESSION = 20
_opened_files_cache: Dict[str, Dict[str, str]] = {}
_opened_files_last: Dict[str, str] = {}




def _db_search_by_filename(
    kb: Any,
    keyword: str = "",
    file_extensions: Optional[List[str]] = None,
    allowed_paths: Optional[list] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    kw = (keyword or "").strip().lower()

    if kw and hasattr(kb, "indexed_keyword_search"):
        indexed = kb.indexed_keyword_search(
            keyword,
            allowed_paths=allowed_paths,
            file_extensions=file_extensions,
            limit=limit,
        )
        out: List[Dict[str, Any]] = []
        for item in indexed[: max(1, int(limit or 20))]:
            fp = str(item.get("file_path") or "").strip()
            name = str(item.get("file_name") or os.path.basename(fp) or "").strip()
            if not fp and not name:
                continue
            out.append(
                {
                    "file_name": name,
                    "file_path": fp,
                    "doc_summary": item.get("doc_summary") or "",
                    "doc_category": item.get("doc_category", "other") or "other",
                    "hit_chunks": 1,
                }
            )
        return out

    if file_extensions and hasattr(kb, "indexed_file_inventory"):
        inventory = kb.indexed_file_inventory(
            allowed_paths=allowed_paths,
            file_extensions=file_extensions,
            limit=limit,
            hydrate=True,
        )
        if not inventory.get("ready"):
            return []
        out: List[Dict[str, Any]] = []
        for item in inventory.get("files") or []:
            fp = str(item.get("file_path") or "").strip()
            name = str(item.get("file_name") or os.path.basename(fp) or "").strip()
            if not fp and not name:
                continue
            out.append(
                {
                    "file_name": name,
                    "file_path": fp,
                    "doc_summary": item.get("doc_summary") or "",
                    "doc_category": item.get("doc_category", "other") or "other",
                    "hit_chunks": int(item.get("hit_chunks") or 1),
                }
            )
        return out

    return []


def set_kb_instance(kb: Any) -> None:
    global _kb_instance
    _kb_instance = kb
    print("[Tools] Knowledge base instance set")


def get_kb_instance() -> Any:
    global _kb_instance
    if _kb_instance is None:
        raise RuntimeError("Knowledge base is not initialized; call set_kb_instance() first")
    return _kb_instance


def set_active_paths(paths: list) -> None:
    global _active_paths
    _active_paths = paths


def get_active_paths() -> Optional[list]:
    return _active_paths


def set_active_session_id(session_id: Optional[str]) -> None:
    global _active_session_id
    _active_session_id = session_id


def get_active_session_id() -> Optional[str]:
    return _active_session_id


def cache_opened_file(session_id: Optional[str], file_path: str, content: str) -> None:
    sid = (session_id or "").strip()
    fp = (file_path or "").strip()
    if not sid or not fp:
        return
    sess = _opened_files_cache.get(sid)
    if sess is None:
        sess = {}
        _opened_files_cache[sid] = sess
    sess[fp] = content or ""
    _opened_files_last[sid] = fp
    if len(sess) > _MAX_OPENED_FILES_PER_SESSION:
        oldest_key = next(iter(sess))
        del sess[oldest_key]


def get_last_opened_file_path(session_id: Optional[str]) -> str:
    sid = (session_id or "").strip()
    if not sid:
        return ""
    return (_opened_files_last.get(sid) or "").strip()


@tool
def get_opened_file_text(file_path: str, max_chars: int = 12000) -> str:
    """
    Return the text of a file that was opened in the current session.

    Use this after `open_file` when a follow-up answer, summary, or question
    needs the opened document text again.

    - Content comes from the `open_file` session cache and is not written to
      chat history.
    - The returned text is truncated to `max_chars` to avoid oversized model
      context.

    Returns:
        JSON string: {ok, file_path, content, truncated, error?}
    """
    sid = get_active_session_id()
    fp = (file_path or "").strip()
    if not sid:
        return json.dumps({"ok": False, "file_path": fp, "error": "missing_session_id"}, ensure_ascii=False)

    if not fp:
        fp = get_last_opened_file_path(sid)
        if not fp:
            return json.dumps({"ok": False, "file_path": "", "error": "missing_file_path"}, ensure_ascii=False)

    sess = _opened_files_cache.get(sid) or {}
    if fp not in sess:
        return json.dumps({"ok": False, "file_path": fp, "error": "not_opened_in_session"}, ensure_ascii=False)

    content = sess.get(fp) or ""
    mc = max(200, int(max_chars))
    truncated = False
    if len(content) > mc:
        content = content[:mc]
        truncated = True
    return json.dumps({"ok": True, "file_path": fp, "content": content, "truncated": truncated}, ensure_ascii=False)



@tool
def search_files(keyword: str, file_extensions: str = None, limit: int = 20) -> str:
    """
    Find files by file name or path in the indexed Chroma database.

    Use this when the assistant needs to find a file first, then open it or
    perform a follow-up operation.

    Implementation details:
    - Queries database metadata directly: file_name, file_name_no_ext, and file_path.
    - Does not scan the physical disk, run shell commands, or depend on memory caches.
    - Respects the current request's active_paths, so unchecked sources are not returned.
    - Returns JSON: {count, files:[{file_name,file_path,doc_summary,doc_category,hit_chunks}]}.
    """
    kb = get_kb_instance()
    allowed = get_active_paths()
    kw = (keyword or "").strip()

    exts_list: Optional[List[str]] = None
    if file_extensions:
        _el: List[str] = []
        for e in file_extensions.split(','):
            e = e.strip()
            if not e:
                continue
            if not e.startswith('.'):
                e = '.' + e
            _el.append(e.lower())
        if _el:
            exts_list = _el

    if not kw and not exts_list:
        return json.dumps({"count": 0, "files": [], "keyword": kw}, ensure_ascii=False)

    matched = _db_search_by_filename(
        kb,
        keyword=kw,
        file_extensions=exts_list,
        allowed_paths=allowed,
        limit=max(1, int(limit)),
    )
    payload = {"count": len(matched), "files": matched, "keyword": kw}
    return json.dumps(payload, ensure_ascii=False)


@tool
def count_documents(category: str = None, keyword: str = None, file_extensions: str = None) -> str:
    """
    Count indexed documents that match optional category, keyword, and extension filters.

    Use this for counting questions such as "how many resumes are there",
    "count all reports", or "how many database files are indexed".

    Args:
        category: Document category. Prefer categories already present in the
            database, but dynamic categories such as invoices or courseware are
            also supported. If omitted, count all documents.
        keyword: Keyword filter used to select documents containing a term,
            often from the file name or metadata.
        file_extensions: Comma-separated file extensions, for example
            ".db,.sql,.sqlite,.csv" or "pdf,docx". Provide this when the user
            asks about a specific file type.

    Returns:
        Human-readable count result.
    """
    kb = get_kb_instance()
    allowed = get_active_paths()
    scope_desc = "ALL" if allowed is None else str(len(allowed))
    print(f"[Tool] count_documents called: category={category}, keyword={keyword}, ext={file_extensions}, allowed_paths={scope_desc}")
    
    exts = []
    if file_extensions:
        for e in file_extensions.split(','):
            e = e.strip()
            if not e: continue
            if not e.startswith('.'): e = '.' + e
            exts.append(e.lower())

    result = kb.count_by_category(category, keyword, file_extensions=exts, allowed_paths=allowed)
    
    count = result.get("count", 0)
    files = result.get("files", [])
    
    desc_parts = []
    if category: desc_parts.append(f"{category}")
    if keyword: desc_parts.append(f'containing "{keyword}"')
    if exts: desc_parts.append(f"with extension {'|'.join(exts)}")
    desc = ", ".join(desc_parts) or "all documents"

    if count == 0:
        return f"No matching documents found ({desc})."
        
    response = f"Found {count} matching documents ({desc}):\n\n"
    
    for i, f in enumerate(files[:15], 1):
        name = f.get('file_name', '')
        path = f.get('file_path', '')
        summary = f.get('doc_summary', '')
        hit_chunks = f.get("hit_chunks")
        response += f"{i}. {name}"
        if hit_chunks is not None:
            response += f" ({hit_chunks} chunks)"
        if path:
            response += f"\n   Path: {path}"
        if summary:
            response += f"\n   Summary: {summary}"
        response += "\n\n"
    
    if count > 15:
        response += f"... {count - 15} more not shown"
    
    return response


@tool
def count_documents_files(category: str = None, keyword: str = None, file_extensions: str = None, limit: int = 50) -> str:
    """
    Count indexed documents and return a structured file list for the UI Sources panel.

    - Counts at file level, deduplicated by file_path, not at chunk level.
    - Returns JSON: {count, raw_count, files:[...]}.

    Args:
        category: Category name such as resume, paper, or book. If omitted with
            no other filter, returns all category stats instead of a file list.
        keyword: Keyword filter.
        file_extensions: Comma-separated file extensions, for example ".db,.sql,.csv".
        limit: Maximum number of files to return. Defaults to 50.
    """
    kb = get_kb_instance()
    allowed = get_active_paths()
    
    exts = []
    if file_extensions:
        for e in file_extensions.split(','):
            e = e.strip()
            if not e: continue
            if not e.startswith('.'): e = '.' + e
            exts.append(e.lower())
            
    if category or keyword or exts:
        result = kb.count_by_category(category, keyword, file_extensions=exts, allowed_paths=allowed)
        files = result.get("files", []) or []
        cleaned_files = [
            {
                "file_name": f.get("file_name", ""),
                "file_path": f.get("file_path", ""),
                "doc_summary": (f.get("doc_summary", "") or "")[:200],
                "doc_category": f.get("doc_category", category or "other"),
                "hit_chunks": f.get("hit_chunks"),
            }
            for f in files[: max(1, int(limit))]
        ]
        payload = {
            "count": int(result.get("count", 0)),
            "raw_count": int(result.get("raw_count", 0)),
            "files": cleaned_files,
            "category": category,
            "keyword": keyword,
        }
        return json.dumps(payload, ensure_ascii=False)
    else:
        stats = kb.count_all_categories(allowed_paths=allowed) or {}
        files = []
        for cat, cnt in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            files.append(
                {
                    "file_name": f"{cat}（{cnt}）",
                    "file_path": f"category:{cat}",
                    "doc_summary": "",
                    "doc_category": cat,
                    "hit_chunks": int(cnt),
                    "type": "folder",
                    "iconType": "folder",
                }
            )

        payload = {"total": int(sum(stats.values())), "stats": stats, "files": files[: max(1, int(limit))]}
        return json.dumps(payload, ensure_ascii=False)


@tool
def search_documents(query: str, category: str = None, keyword: str = None, file_extensions: str = None, top_k: int = 10) -> str:
    """
    Search indexed document content semantically.

    Use this for questions that require finding specific information inside
    documents, such as "what is DPO", "who is party A in the contract", or
    "who is the design director".

    Args:
        query: Search query describing the information to find.
        category: Document category. Prefer existing database categories such
            as resume or report. Provide it when the user asks within a category.
        keyword: Keyword filter used to match file names.
        file_extensions: Comma-separated file extensions, for example ".pdf,.docx".
        top_k: Number of results to return. Defaults to 10.

    Returns:
        Relevant document snippets, or "[NO_RELEVANT_DOCS]" when nothing relevant is found.
    """
    kb = get_kb_instance()
    allowed = get_active_paths()
    scope_desc = "ALL" if allowed is None else str(len(allowed))
    print(f"[Tool] search_documents called: query={query}, category={category}, keyword={keyword}, extensions={file_extensions}, top_k={top_k}, allowed_paths={scope_desc}")
    
    exts = []
    if file_extensions:
        for e in file_extensions.split(','):
            e = e.strip()
            if not e: continue
            if not e.startswith('.'): e = '.' + e
            exts.append(e.lower())

    results = kb.vector_search(query, n_results=settings.VECTOR_SEARCH_TOP_K, allowed_paths=allowed, category_filter=category, keyword=keyword, file_extensions=exts)
    
    if not results:
        print("[Tool] vector search returned no results")
        return "[NO_RELEVANT_DOCS]"
    
    print(f"[Tool] vector search returned {len(results)} results")

    _GATE2_VECTOR_MIN = 0.40
    _GATE2_BM25_MIN = 5.0
    pre_filtered = [
        r for r in results
        if float(r.get("score", 0) or 0) >= _GATE2_VECTOR_MIN
        or float(r.get("_bm25_score", 0) or 0) >= _GATE2_BM25_MIN
    ]
    if len(pre_filtered) < 3:
        pre_filtered = results
    print(f"[Tool] Gate2 prefilter kept {len(pre_filtered)} candidates out of {len(results)}")
    # ─────────────────────────────────────────────────────────────────────────

    reranked = kb.rerank(query, pre_filtered, top_k=settings.RERANK_TOP_K)
    
    threshold = settings.RELEVANCE_THRESHOLD
    filtered = [doc for doc in reranked if doc.get('rerank_score', 0) >= threshold]

    _GATE5_GAP = 3.0
    if filtered and len(filtered) > 1:
        _top_score = float(filtered[0].get('rerank_score', 0) or 0)
        filtered = [
            doc for doc in filtered
            if _top_score - float(doc.get('rerank_score', 0) or 0) <= _GATE5_GAP
        ]
    # ─────────────────────────────────────────────────────────────────────────

    print(f"[Tool] reranked {len(reranked)} results; Gate4+Gate5 kept {len(filtered)} (threshold={threshold}, gap={_GATE5_GAP})")
    
    if not filtered:
        print(f"[Tool] no relevant results; top score: {reranked[0].get('rerank_score', 0):.2f}")
        return "[NO_RELEVANT_DOCS]"
    
    file_best = {}  # {file_path: doc}
    for doc in filtered:
        file_path = doc.get('file_path', '')
        if file_path not in file_best or doc.get('rerank_score', 0) > file_best[file_path].get('rerank_score', 0):
            file_best[file_path] = doc
    
    unique_docs = sorted(file_best.values(), key=lambda x: x.get('rerank_score', 0), reverse=True)
    
    print(f"[Tool] deduplicated to {len(unique_docs)} files")
    
    print("[Tool] === Retrieved evidence ===")
    for i, doc in enumerate(unique_docs[:top_k], 1):
        file_name = doc.get('file_name', '')
        score = doc.get('rerank_score', 0)
        text_preview = doc.get('text', '')[:80].replace('\n', ' ')
        print(f"  {i}. [{score:.2f}] {file_name}: {text_preview}...")
    
    response = f"Found {len(unique_docs)} relevant documents:\n\n"
    for i, doc in enumerate(unique_docs, 1):
        file_name = doc.get('file_name', '')
        summary = doc.get('doc_summary', '')
        category = doc.get('doc_category', '')
        text = doc.get('text', '')[:300]
        score = doc.get('rerank_score', 0)
        
        response += f"[{i}] {file_name}"
        if category:
            response += f" [{category}]"
        response += f" (relevance: {score:.2f})\n"
        if summary:
            response += f"Summary: {summary}\n"
        response += f"Content: {text}...\n\n"
    
    return response


@tool
def summarize_topics(category: str) -> str:
    """
    Summarize the topic distribution of documents in a category.

    Use this for synthesis questions such as "what topics do the papers mainly
    cover" or "what themes are included in the reports".

    Args:
        category: Document category, such as paper, report, or resume.

    Returns:
        Topic distribution summary for all documents under the category.
    """
    kb = get_kb_instance()
    allowed = get_active_paths()
    scope_desc = "ALL" if allowed is None else str(len(allowed))
    print(f"[Tool] summarize_topics called: category={category}, allowed_paths={scope_desc}")

    try:
        inventory = kb.indexed_file_inventory(
            allowed_paths=allowed,
            category_filter=category,
            limit=0,
            hydrate=True,
        ) if hasattr(kb, "indexed_file_inventory") else {"ready": False, "files": []}
        if not inventory.get("ready"):
            return json.dumps(
                {"ok": False, "category": category, "error": "index_not_ready", "files": []},
                ensure_ascii=False,
            )

        file_summaries = {}  # {file_path: {file_name, doc_summary}}
        for item in inventory.get("files") or []:
            meta = dict(item.get("metadata") or {})
            file_path = str(item.get("file_path") or meta.get("file_path") or "")
            if file_path and file_path not in file_summaries:
                file_summaries[file_path] = {
                    'file_name': str(item.get("file_name") or meta.get("file_name") or ""),
                    'doc_summary': str(item.get("doc_summary") or meta.get("doc_summary") or ""),
                }
        
        total = len(file_summaries)
        print(f"[Tool] found {total} documents in category={category}")
        
        files = [
            {
                "file_name": info["file_name"],
                "file_path": path,
                "doc_summary": info["doc_summary"],
            }
            for path, info in file_summaries.items()
        ]

        return json.dumps(
            {
                "ok": True,
                "category": category,
                "total": total,
                "instruction": "Use these document summaries as evidence to summarize the main topic areas.",
                "files": files,
            },
            ensure_ascii=False,
        )
        
    except Exception as e:
        print(f"[Tool] summarize_topics failed: {e}")
        return json.dumps(
            {"ok": False, "category": category, "error": str(e), "files": []},
            ensure_ascii=False,
        )



def _register_tools():
    ToolRegistry.register_langchain_tool("count_documents", count_documents)
    ToolRegistry.register_langchain_tool("count_documents_files", count_documents_files)
    ToolRegistry.register_langchain_tool("search_files", search_files)
    ToolRegistry.register_langchain_tool("get_opened_file_text", get_opened_file_text)
    ToolRegistry.register_langchain_tool("search_documents", search_documents)
    ToolRegistry.register_langchain_tool("summarize_topics", summarize_topics)
    
    ToolRegistry.register(
        "count_documents", 
        count_documents.invoke,
        description="Count indexed documents by optional category, keyword, and file extension filters.",
        category="document"
    )
    ToolRegistry.register(
        "count_documents_files",
        count_documents_files.invoke,
        description="Count indexed documents and return a structured file list for the Sources panel.",
        category="document"
    )
    ToolRegistry.register(
        "search_files",
        search_files.invoke,
        description="Find files by file name, path, or extension in the indexed database and return a structured list.",
        category="document"
    )
    ToolRegistry.register(
        "get_opened_file_text",
        get_opened_file_text.invoke,
        description="Return truncated text for a file already opened in the current session for follow-up summary or QA.",
        category="document"
    )
    ToolRegistry.register(
        "search_documents", 
        search_documents.invoke,
        description="Semantically search indexed document content.",
        category="document"
    )
    ToolRegistry.register(
        "summarize_topics", 
        summarize_topics.invoke,
        description="Summarize topic distribution for documents in a category.",
        category="document"
    )


_register_tools()

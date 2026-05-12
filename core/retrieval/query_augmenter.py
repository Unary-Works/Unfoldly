"""
Query Augmenter — retrieval query enhancement pipeline.

Extracted from langgraph_agent.py. This module provides:
  - augment_query_for_retrieval: Translate/clean user queries for vector search
  - strip_meta_for_rerank: Remove instruction words for reranker scoring
  - blend_retrieval_query_with_original_cjk: Preserve CJK entities in English queries
  - anchor_retrieval_query_with_last_search: Add context from prior results

All functions are stateless (no LLM dependency except augment_query_for_retrieval
which uses the provided llm_service for translation).
"""
from __future__ import annotations

import os
import re
import logging
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger(__name__)


def augment_query_for_retrieval(
    query: str,
    *,
    llm_service: Any,
    last_results_fn: Optional[Callable] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Augment user query for semantic retrieval.
    
    - Translates CJK queries to English keywords for vector search
    - Preserves Chinese entity words alongside the English translation
    - Anchors generic references ("this file") with prior search results
    
    Args:
        query: Raw user query
        llm_service: LLM service for translation (must have .generate() method)
        last_results_fn: Callable that returns last search results for session
        session_id: Current session ID
    """
    raw = (query or "").strip()
    if not raw:
        return raw

    # If no CJK characters, use as-is
    if not any("\u4e00" <= ch <= "\u9fff" for ch in raw):
        return raw

    try:
        def _fallback_keywords(src: str) -> str:
            parts = re.findall(r"[\u4e00-\u9fff]{2,6}|[A-Za-z0-9._-]{2,32}", src or "")
            stop = {
                "什么", "怎么", "如何", "一下", "这个", "那个", "以及", "是否",
                "what", "how", "does", "is", "are", "the", "this", "that",
            }
            kept: List[str] = []
            seen: set = set()
            for p in parts:
                t = str(p).strip()
                if not t:
                    continue
                tl = t.lower()
                if tl in stop or tl in seen:
                    continue
                seen.add(tl)
                kept.append(t)
                if len(kept) >= 10:
                    break
            return " ".join(kept).strip()

        prompt = (
            "Rewrite the following user query into concise English for semantic retrieval.\n"
            "- Preserve names, product/model terms, numbers, code tokens, and file extensions.\n"
            "- Output one English query only, no explanation.\n\n"
            f"{raw}"
        )
        raw_out = (llm_service.generate(prompt) or "").strip()
        first_line = ""
        for ln in raw_out.splitlines():
            s = ln.strip()
            if s:
                first_line = s
                break
        en_q = first_line
        low_out = (raw_out or "").lower()
        suspicious = (
            ("user wants" in low_out and "rewrite" in low_out)
            or ("output one" in low_out and "query" in low_out)
            or ("translation only" in low_out and "source" in low_out)
        )
        if suspicious or len(en_q) > 320:
            en_q = _fallback_keywords(raw)
        if not en_q:
            return raw
        if en_q.lower() == raw.lower():
            return raw

        # Preserve original CJK entity words
        try:
            cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,6}", raw)
            if cjk_terms:
                seen: set = set()
                kept: List[str] = []
                stop_terms = {
                    # Interrogative / question words
                    "是什么", "什么", "多少", "怎么", "如何", "怎么样", "怎样",
                    "哪个", "哪些", "哪里", "哪位", "什么时候", "为什么", "谁",
                    # Instruction / command verbs
                    "帮我", "帮忙", "请帮", "告诉我", "告诉", "给我", "让我",
                    "查一下", "查下", "找一下", "找下", "搜一下", "搜下",
                    "看一下", "看看", "看下", "查看", "请问",
                    # Filler / noise particles
                    "一下", "这个", "那个", "以及", "是否",
                    "的吗", "的呢", "嘛", "吗", "呢", "吧", "啊", "了", "过",
                    # Multi-word meta phrases
                    "主要内容", "主要", "内容", "协作模式",
                    "关键要点", "核心内容", "重点内容",
                }
                for term in cjk_terms:
                    t = term.strip()
                    t = re.sub(r"^[的与和在把将对请问呢吗啊呀了]+", "", t)
                    t = re.sub(r"[的与和呢吗啊呀了]+$", "", t)
                    if t.endswith("是") and len(t) > 2:
                        t = t[:-1]
                    if (not t) or (len(t) < 2) or (len(t) > 6):
                        continue
                    if t in stop_terms:
                        continue
                    if ("是什么" in t) or ("怎么样" in t) or ("怎么" in t):
                        continue
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    if t not in en_q:
                        kept.append(t)
                if kept:
                    en_q = f"{en_q} {' '.join(kept[:4])}".strip()
        except Exception:
            pass

        en_q = " ".join(en_q.split())[:320]

        # Anchor generic references with prior results
        low = en_q.lower()
        generic_ref_hits = (
            "this document", "this file", "that document",
            "that file", "the document", "the file",
        )
        if any(k in low for k in generic_ref_hits):
            try:
                last_results = last_results_fn() if last_results_fn else []
            except Exception:
                last_results = []
            if last_results:
                top = last_results[0] or {}
                anchor_name = str(top.get("file_name") or "").strip()
                anchor_summary = str(top.get("doc_summary") or "").strip()
                anchor_parts = []
                if anchor_name:
                    anchor_parts.append(anchor_name)
                if anchor_summary:
                    anchor_parts.append(anchor_summary[:160])
                if anchor_parts:
                    return f"{en_q} {' '.join(anchor_parts)}".strip()
        return en_q
    except Exception:
        return raw


def strip_meta_for_rerank(query: str) -> str:
    q = str(query or "").strip()
    if not q:
        return q

    # ── Phase 1: Chinese meta-phrase removal ──
    _cn_meta_phrases = [
        "能不能", "可不可以", "帮我找到", "帮我找", "帮我看看", "帮我看",
        "给我看看", "给我看", "给我找", "给我",
        "请问", "请帮",
        "有没有", "有哪些", "有多少",
        "查看", "查找", "搜索", "找到", "显示", "列出", "列举",
        "展示", "打开", "浏览", "获取", "下载",
        "所有的", "全部的", "所有", "全部", "一下", "一些",
        "图片", "图像", "照片", "文件", "文档",
        "我的", "关于", "有关",
        "找", "看", "请",
    ]
    result = q
    for w in _cn_meta_phrases:
        result = result.replace(w, " ")

    result = re.sub(r"的\s*$", " ", result)
    result = re.sub(r"^的\s", " ", result)

    # ── Phase 2: English — context-aware token filtering ──
    _en_instruction = frozenset({
        "find", "show", "get", "list", "give", "search", "look",
        "display", "view", "open", "locate", "retrieve", "fetch",
        "browse", "scan", "check", "see", "tell",
        "please", "can", "could", "would", "want", "need", "like",
    })
    _en_filler = frozenset({
        "me", "my", "our", "your", "i", "we", "you",
        "the", "a", "an", "this", "that", "these", "those",
        "of", "in", "for", "with", "about", "from", "to", "on", "at", "by",
        "all", "any", "some", "every", "each",
        "there", "here", "where", "which", "what",
        "is", "are", "was", "were", "be", "been",
        "do", "does", "did", "has", "have", "had",
        "related", "regarding",
    })
    _en_ambiguous = frozenset({
        "image", "images", "photo", "photos", "picture", "pictures",
        "file", "files", "document", "documents", "doc", "docs",
        "pdf", "pptx", "docx", "xlsx", "csv",
    })
    _en_always_strip = _en_instruction | _en_filler

    tokens = result.split()
    n = len(tokens)
    keep = [True] * n

    for i, tok in enumerate(tokens):
        t = tok.lower()
        if t in _en_always_strip:
            keep[i] = False
        elif t in _en_ambiguous:
            has_content_neighbor = False
            if i > 0 and tokens[i - 1].lower() not in (_en_always_strip | _en_ambiguous):
                has_content_neighbor = True
            if i < n - 1 and tokens[i + 1].lower() not in (_en_always_strip | _en_ambiguous):
                has_content_neighbor = True
            if not has_content_neighbor:
                keep[i] = False

    content_tokens = [tokens[i] for i in range(n) if keep[i]]
    result = " ".join(content_tokens)
    result = re.sub(r"\s+", " ", result).strip()

    if not result or len(result) < 2:
        return q
    return result


def blend_retrieval_query_with_original_cjk(
    retrieval_query: str,
    original_question: str,
) -> str:
    """Blend CJK entities from the original question back into the English retrieval query."""
    rq = str(retrieval_query or "").strip()
    oq = str(original_question or "").strip()
    if not oq or any("\u4e00" <= ch <= "\u9fff" for ch in rq):
        return rq

    extras: List[str] = []
    seen = {rq.casefold()}
    for seg in re.findall(r"[\u4e00-\u9fff]{2,}", oq):
        if seg in rq:
            continue
        sk = seg.casefold()
        if sk in seen:
            continue
        seen.add(sk)
        extras.append(seg)
        if len(extras) >= 6:
            break
    if not extras:
        return rq
    return f"{rq} {' '.join(extras)}".strip()[:400]


def anchor_retrieval_query_with_last_search(
    retrieval_query: str,
    original_question: str,
    last_results: List[Dict[str, Any]],
) -> str:
    """Anchor follow-up queries with directory/file names from prior search results."""
    from core.intent_analyzer import IntentAnalyzer, IntentKeywords

    rq = str(retrieval_query or "").strip()
    oq = str(original_question or "").strip()
    if not IntentAnalyzer.looks_like_content_followup_on_prior_results(oq):
        return rq
    if not last_results:
        return rq
    oql = oq.lower()

    has_prev_ref = IntentAnalyzer._is_kw_match(oql, IntentKeywords.PREV_REF_KWS, "all")
    explicit_fresh_search = bool(
        re.search(
            r"^\s*\b(find|search|look\s+for|show\s+me|list|display|retrieve|what\s+files|which\s+files|do\s+i\s+have)\b",
            oql,
            re.IGNORECASE,
        )
        or any(
            oq.startswith(prefix)
            for prefix in (
                "我有哪些", "我有什么", "有哪些", "有什么", "有多少", "多少个", "多少份",
                "列出", "显示", "查找", "搜索", "搜一下", "搜下", "查一下", "查下",
                "找一下", "找下", "帮我找", "帮我搜", "帮我查",
            )
        )
    )
    if explicit_fresh_search and not has_prev_ref:
        logger.info(
            "[query_augmenter] skip last-search anchoring for explicit fresh search: %r",
            oq,
        )
        return rq

    extras: List[str] = []
    seen = {w.casefold() for w in rq.split() if w}
    for d in last_results[:12]:
        r = str(d.get("folder_chain_root") or "").strip()
        if r:
            try:
                bn = os.path.basename(os.path.normpath(os.path.expanduser(r)))
            except Exception:
                bn = os.path.basename(r)
            if bn and bn.casefold() not in seen:
                seen.add(bn.casefold())
                extras.append(bn)
        fn = str(d.get("file_name") or "").strip()
        if fn and fn.casefold() not in seen:
            seen.add(fn.casefold())
            extras.append(fn)
        if len(extras) >= 8:
            break
    if not extras:
        return rq
    return f"{rq} {' '.join(extras)}".strip()[:420]

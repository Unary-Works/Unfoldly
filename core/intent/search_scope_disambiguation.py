from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from core.retrieval.filename_canonicalizer import classify_reference_target


_SEARCH_VERB_RE = re.compile(
    r"^\s*(?:please\s+|pls\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+)?"
    r"(?:find|search(?:\s+for)?|look\s+for|locate|get(?:\s+me)?|show(?:\s+me)?|"
    r"retrieve|fetch|list|display|browse)\b"
    r"|^\s*(?:请|麻烦你|帮我|给我)?\s*"
    r"(?:找|搜|查|检索|查找|搜索|找找|找下|搜下|查下|找一下|搜一下|查一下|列出|显示|看看|看下)",
    re.IGNORECASE,
)

_PREVIOUS_SCOPE_RE = re.compile(
    r"\b(previous|prior|last|above|earlier|same)\s+"
    r"(results?|files?|documents?|docs?|items?|set|ones?)\b"
    r"|\b(in|within|among|from)\s+(them|these|those|it|that|the\s+above|the\s+previous|the\s+last)\b"
    r"|^\s*(?:find|search(?:\s+for)?|look\s+for|locate|get(?:\s+me)?|show(?:\s+me)?)\s+"
    r"(?:them|these|those|it|that|this)\s*$"
    r"|(?:上一轮|上轮|上面|上次|刚才|前面|前一轮|之前|上述|以上).{0,8}(?:结果|文件|文档|里面|里|中|搜|找|查)"
    r"|(?:在|从|只在).{0,8}(?:上一轮|上轮|上面|上次|刚才|前面|之前|上述|以上).{0,8}(?:结果|文件|文档|里面|里|中|搜|找|查)"
    r"|这些|那些|它们|他们|她们|它|这个|那个|结果里|结果中|这批",
    re.IGNORECASE,
)

_IT_ACRONYM_QUERY_RE = re.compile(
    r"^\s*(?:find|search(?:\s+for)?|look\s+for|locate|get(?:\s+me)?|show(?:\s+me)?|"
    r"retrieve|fetch|list|display|browse)\s+IT\b"
)
_UPPERCASE_IT_TOKEN_RE = re.compile(r"\bIT\b")

_SELECTED_SCOPE_RE = re.compile(
    r"\b(current|selected|selection|chosen|checked|all\s+selected|global\s+selected)\b"
    r"|当前|全局|选中|已选|勾选|选择的|当前全局选中|当前选中|全局选中",
    re.IGNORECASE,
)

_PENDING_PREVIOUS_CHOICE_RE = re.compile(
    r"\b(previous|prior|last|above|earlier|results?)\b|上一轮|上轮|上面|上文|之前|刚才|前面|结果",
    re.IGNORECASE,
)

_PENDING_SELECTED_CHOICE_RE = re.compile(
    r"\b(current|selected|selection|chosen|checked|global|all\s+selected)\b|当前|全局|选中|选区|已选|勾选",
    re.IGNORECASE,
)

_PRIOR_ENTITY_REFERENCE_RE = re.compile(
    r"\b(he|she|him|his|hers?|their|theirs)\b|"
    r"他的?|她的?|他们的?|她们的?|这个人|这位|该候选人|候选人",
    re.IGNORECASE,
)

_SCOPED_COMPARISON_RE = re.compile(
    r"\b(?:which\s+(?:one|candidate|resume|report|file|document|item)|"
    r"which\s+of\s+(?:these|those|them|the\s+above)|who)\b"
    r".{0,80}\b(?:best|better|most|least)\b.{0,40}"
    r"\b(?:fit|fits|match(?:es|ing)?|suit(?:s|ed)?|align(?:s|ed)?|relevant)\b"
    r"|(?:这几份|这些|那几份|那些|其中|上面|上述|前面|这批|那批).{0,24}"
    r"(?:哪一份|哪份|哪个|谁).{0,24}(?:最|更).{0,8}(?:匹配|适合|符合|契合|对应)"
    r"|(?:哪一份|哪份|哪个|谁).{0,24}(?:最|更).{0,8}(?:匹配|适合|符合|契合|对应)",
    re.IGNORECASE,
)

_PERSONAL_ATTRIBUTE_RE = re.compile(
    r"'s\s*(phone|email|e-mail|address|home\s+address|residence|contact|mobile|tel|salary|birthday|number|password|id)|"
    r"\b(his|her|their|its)\s+"
    r"(phone\s*(number)?|email(\s+address)?|e-mail|mailing\s+address|street\s+address|"
    r"work\s+address|office\s+address|mobile\s*(number)?|telephone|contact\s*(info|details?)|"
    r"salary|birthday|date\s+of\s+birth|location|work\s+location|office\s+location|"
    r"home(?:\s+address)?|residence|residential\s+address|"
    r"school|university|college|alma\s+mater|education|degree|major|graduat(?:e|ed|ion)|"
    r"employer|company|job|title|role|position)\b|"
    r"\b(?:where|which\s+school|what\s+school).{0,24}\b(?:he|she|they|his|her|their).{0,24}"
    r"(?:graduate|study|school|university|college|live|lives|reside|resides|home|address|location)\b|"
    r"\b(phone\s*(number)?|email(\s+address)?|e-mail|mailing\s+address|street\s+address|"
    r"work\s+address|office\s+address|mobile\s*(number)?|telephone|contact\s*(info|details?)|"
    r"salary|birthday|date\s+of\s+birth|location|work\s+location|office\s+location|"
    r"home\s+address|residence|residential\s+address)\b|"
    r"的\s*(电话|手机|邮箱|地址|家庭住址|住址|居住地|居住地址|住所|住宅|联系方式|联系电话|邮件地址|生日|工资|薪资|薪水|密码|账号|id|工号)|"
    r"的\s*(毕业院校|毕业学校|学校|大学|高校|学历|学位|专业|公司|单位|雇主|职位|职务|岗位)|"
    r"(?<=[一-鿿])(电话|手机|邮箱|地址|家庭住址|住址|居住地|联系方式)|"
    r"(?<=[一-鿿])(毕业院校|毕业学校|学历|学位|专业)|"
    r"(电话|手机号|邮箱|地址|家庭住址|住址|居住地|联系方式|电话号码|位置|所在地)(是多少|是什么|是哪个|号码|是哪|在哪里|在哪)"
    r"|(?:他|她|这个人|这位|该候选人|候选人).{0,12}(哪个学校|什么学校|毕业于|毕业院校|毕业学校|学历|学位|专业|哪家公司|什么公司|职位|职务|岗位)"
    r"|(?:哪个学校|什么学校|毕业院校|毕业学校|学历|学位|专业|哪家公司|什么公司|职位|职务|岗位).{0,12}(他|她|这个人|这位|候选人)"
    r"|(?:他|她|这个人|这位|该候选人|候选人).{0,12}(住哪|住在哪里|住址|家庭住址|居住地|地址|家在哪|家在哪里)"
    r"|(?:住哪|住在哪里|住址|家庭住址|居住地|地址|家在哪|家在哪里).{0,12}(他|她|这个人|这位|候选人)",
    re.IGNORECASE,
)

_CONTEXTUAL_FOLLOWUP_RE = re.compile(
    r"^\s*(?:"
    r"what\s+(?:are|is|were|was)\s+(?:the\s+)?(?:key\s+points?|main\s+points?|takeaways?|summary|"
    r"common\s+points?|differences?|conclusion)|"
    r"what\s+(?:are|is|were|was)\s+(?:the\s+)?(?:main|key|top|primary|core|major|projected|expected|current|final)\b|"
    r"(?:what|which|who|where|when|why|how).{0,80}\b(?:it|them|these|those|this|that|their|the\s+(?:report|document|file|paper|resume|invoice|presentation))\b|"
    r"what\s+(?:does|do|is|are).{0,32}\b(?:say|mean|contain|cover|describe|discuss)|"
    r"why\s+(?:is|are|does|do).{0,80}|"
    r"how\s+(?:is|are|does|do).{0,80}|"
    r"(?:can|could|would)\s+you\s+(?:summari[sz]e|recap|explain|describe|analy[sz]e|compare)\b|"
    r"tell\s+me\s+(?:more|about|the\s+details)|"
    r"(?:summari[sz]e|recap|explain|describe|analy[sz]e|compare|connect|synthesi[sz]e|interpret|infer|translate)\b|"
    r"(?:if\s+(?:not|so)|otherwise).{0,80}\b(?:explain|describe|summari[sz]e|compare|list)\b|"
    r"(?:more|details?|key\s+points?|takeaways?)\s*$"
    r")"
    r"|^\s*(?:总结|概括|归纳|解释|说明|详细|展开|继续|讲讲|分析|对比|比较|关联|联系|综合|推断|"
    r"要点|重点|共同点|区别|结论|为什么|怎么|是什么|讲了什么|说了什么|如果不是)",
    re.IGNORECASE,
)

_SEARCH_PHRASED_FOLLOWUP_RE = re.compile(
    r"^\s*(?:show(?:\s+me)?|get|give(?:\s+me)?|display)\s+"
    r"(?:what\s+(?:it|this|that|they|these|those)\s+(?:says?|contains?|covers?|means?)|"
    r"(?:the\s+)?(?:content|summary|details?|key\s+points?|takeaways?)\s+"
    r"(?:of|about|for)?\s*(?:it|this|that|them|these|those)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScopeDecision:
    action: str
    params: Dict[str, Any]
    reason: str


def looks_like_search_request(question: str) -> bool:
    return bool(_SEARCH_VERB_RE.search(str(question or "").strip()))


def looks_like_personal_attribute_request(question: str) -> bool:
    q = str(question or "").strip()
    if not q:
        return False
    if has_explicit_previous_scope(q) and _SCOPED_COMPARISON_RE.search(q):
        return False
    return bool(_PERSONAL_ATTRIBUTE_RE.search(q))


def has_prior_entity_reference(question: str) -> bool:
    q = str(question or "").strip()
    if not q:
        return False
    q_for_reference = _UPPERCASE_IT_TOKEN_RE.sub("INFORMATION_TECHNOLOGY", q)
    return bool(_PRIOR_ENTITY_REFERENCE_RE.search(q_for_reference))


def looks_like_contextual_followup_request(question: str) -> bool:
    q = str(question or "").strip()
    if not q:
        return False
    if _SEARCH_PHRASED_FOLLOWUP_RE.search(q):
        return True
    if looks_like_search_request(q) or looks_like_personal_attribute_request(q):
        return False
    return bool(_CONTEXTUAL_FOLLOWUP_RE.search(q))


def has_explicit_previous_scope(question: str) -> bool:
    q = str(question or "").strip()
    if not q:
        return False
    q_for_reference = _UPPERCASE_IT_TOKEN_RE.sub("INFORMATION_TECHNOLOGY", q)
    if _PREVIOUS_SCOPE_RE.search(q_for_reference):
        return True
    if has_prior_entity_reference(q_for_reference):
        return True
    if looks_like_search_request(q):
        return False
    try:
        ref = classify_reference_target(q_for_reference)
        target = str(ref.get("target") or "").strip().lower()
        if target == "it" and _IT_ACRONYM_QUERY_RE.search(q):
            return False
        return str(ref.get("kind") or "") == "deictic"
    except Exception:
        return False


def has_explicit_selected_scope(question: str) -> bool:
    return bool(_SELECTED_SCOPE_RE.search(str(question or "").strip()))


def normalize_path_identity(raw_path: Any) -> str:
    value = str(raw_path or "").strip()
    if not value:
        return ""
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(os.path.expanduser(value))))
    except Exception:
        return os.path.normcase(os.path.normpath(value))


def result_paths(results: Sequence[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    seen = set()
    for row in list(results or []):
        path = normalize_path_identity((row or {}).get("file_path") or (row or {}).get("path"))
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def paths_match_exactly(left: Sequence[Any], right: Sequence[Any]) -> bool:
    left_set = {p for p in (normalize_path_identity(item) for item in list(left or [])) if p}
    right_set = {p for p in (normalize_path_identity(item) for item in list(right or [])) if p}
    return bool(left_set) and left_set == right_set


def selected_scope_matches_previous(
    *,
    active_paths: Sequence[str],
    last_results: Sequence[Dict[str, Any]],
    count_scope_context: Optional[Dict[str, Any]] = None,
    total_searchable: Optional[int] = None,
) -> bool:
    previous_paths = result_paths(last_results)
    if previous_paths and paths_match_exactly(active_paths, previous_paths):
        return True
    if previous_paths:
        return False

    ctx = dict(count_scope_context or {})
    if not ctx:
        return False

    try:
        total_files = int(ctx.get("total_files") or 0)
    except (TypeError, ValueError):
        total_files = 0
    if total_files <= 0:
        return False

    selected_count = len([p for p in list(active_paths or []) if str(p or "").strip()])
    if selected_count and selected_count == total_files:
        return True

    try:
        searchable_count = int(total_searchable) if total_searchable is not None else -1
    except (TypeError, ValueError):
        searchable_count = -1
    return selected_count == 0 and searchable_count > 0 and searchable_count == total_files


def clarify_message(language: str) -> str:
    if str(language or "").lower().startswith("zh"):
        return "你想处理上文的 relevant files、当前选区文件，还是其他文件？请指明范围。"
    return "Do you want me to use the previous relevant files, the current selection, or other files? Please specify the scope."


def resolve_pending_scope_choice(question: str, pending_params: Dict[str, Any]) -> Optional[ScopeDecision]:
    q = str(question or "").strip()
    if not q:
        return None

    previous = bool(_PENDING_PREVIOUS_CHOICE_RE.search(q))
    selected = bool(_PENDING_SELECTED_CHOICE_RE.search(q))
    if previous == selected:
        return None

    original_query = str((pending_params or {}).get("query") or "").strip()
    if not original_query:
        return None

    params = dict((pending_params or {}).get("search_params") or {})
    params["query"] = original_query
    params["_scope_disambiguation"] = "previous_choice" if previous else "selected_choice"
    operation = str((pending_params or {}).get("operation") or "search").strip().lower()
    if operation in {"summarize", "summary", "process_previous"}:
        if previous:
            params["_scope"] = "previous"
            return ScopeDecision(action="process_previous", params=params, reason="previous_choice")
        params.pop("_scope", None)
        params["_scope"] = "selected"
        return ScopeDecision(action="summarize_all", params=params, reason="selected_choice")
    if previous:
        params["_scope"] = "previous"
    else:
        params.pop("_scope", None)
    return ScopeDecision(action="search", params=params, reason=params["_scope_disambiguation"])

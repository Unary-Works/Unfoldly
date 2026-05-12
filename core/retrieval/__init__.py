# core/retrieval — retrieval pipeline modules

from .lookup_terms import (
    apply_media_query_constraints,
    annotate_candidates_with_topic_overlap,
    build_lookup_blob,
    build_cjk_latin_aliases,
    build_candidate_lookup_blob,
    compute_lookup_overlap_score,
    extract_lookup_terms,
    extract_strong_lookup_anchors,
    is_lookup_heavy_query,
    lookup_match_quality,
    narrow_candidates_by_topic_overlap,
    sort_candidates_by_topic_overlap,
)
from .keyword_index import (
    KeywordIndexManager,
    KeywordIndexRecord,
)
from .path_scope import (
    PathScopeMatcher,
    ensure_path_scope_matcher,
    filter_sources_to_scope,
)

__all__ = [
    "apply_media_query_constraints",
    "build_lookup_blob",
    "build_cjk_latin_aliases",
    "build_candidate_lookup_blob",
    "compute_lookup_overlap_score",
    "extract_lookup_terms",
    "extract_strong_lookup_anchors",
    "is_lookup_heavy_query",
    "lookup_match_quality",
    "annotate_candidates_with_topic_overlap",
    "sort_candidates_by_topic_overlap",
    "narrow_candidates_by_topic_overlap",
    "KeywordIndexManager",
    "KeywordIndexRecord",
    "PathScopeMatcher",
    "ensure_path_scope_matcher",
    "filter_sources_to_scope",
]

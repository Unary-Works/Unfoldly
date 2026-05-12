import os
from typing import Any, Dict, List, Optional, Tuple
from config import settings

def _env_truthy(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in {"1", "true", "yes", "on", "y"}


if _env_truthy("FILEAGENT_ENABLE_LLAMA_INDEX_FALLBACK"):
    try:
        from llama_index.core.postprocessor.types import BaseNodePostprocessor
        from llama_index.core.schema import NodeWithScore, QueryBundle
    except ImportError:
        BaseNodePostprocessor = None  # type: ignore[assignment]
        QueryBundle = None  # type: ignore[assignment]
        NodeWithScore = Any
else:
    BaseNodePostprocessor = None  # type: ignore[assignment]
    QueryBundle = None  # type: ignore[assignment]
    NodeWithScore = Any

if BaseNodePostprocessor is None:
    class BaseNodePostprocessor:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class QueryBundle:
        query_str: str = ""

    NodeWithScore = Any


def _coerce_score_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return _coerce_score_value(value.get("score"))
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return _coerce_score_value(value[0])
    raise TypeError(f"Unsupported reranker score value: {type(value).__name__}")


def _build_rank_batch_inputs(llm, pairs: List[Tuple[str, str]]) -> List[List[int]]:
    import llama_cpp

    rerank_template = None
    try:
        tpl = llama_cpp.llama_model_chat_template(llm._model.model, b"rerank")
        if tpl:
            rerank_template = tpl.decode("utf-8")
    except Exception:
        rerank_template = None

    batch_inputs: List[List[int]] = []
    query_cache: Dict[str, List[int]] = {}
    eos_id = llm.token_eos()
    sep_id = llm.token_sep() if llm.token_sep() != -1 else eos_id

    for query, document in pairs:
        if rerank_template:
            prompt = rerank_template.replace("{query}", query).replace("{document}", document)
            batch_inputs.append(
                llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
            )
            continue

        q_tokens = query_cache.get(query)
        if q_tokens is None:
            q_tokens = llm.tokenize(query.encode("utf-8"), add_bos=True, special=True)
            if q_tokens and q_tokens[-1] == eos_id:
                q_tokens = q_tokens[:-1]
            query_cache[query] = q_tokens

        d_tokens = llm.tokenize(document.encode("utf-8"), add_bos=False, special=True)
        full_seq = list(q_tokens) + [sep_id] + d_tokens
        if not full_seq or full_seq[-1] != eos_id:
            full_seq.append(eos_id)
        batch_inputs.append(full_seq)

    return batch_inputs


def _rank_with_llama(llm, query: str, documents: List[str]) -> List[float]:
    """
    Compute rerank scores using LlamaEmbedding.embed() directly.
    This bypasses the current fork's rank() single-document wrapper bug.
    """
    if not documents:
        return []

    from llama_cpp.llama_embedding import NORM_MODE_NONE

    pairs = [(str(query or ""), str(doc or "")) for doc in documents]
    batch_inputs = _build_rank_batch_inputs(llm, pairs)
    raw_scores = llm.embed(batch_inputs, normalize=NORM_MODE_NONE)

    if isinstance(raw_scores, (int, float)):
        return [float(raw_scores)] + [0.0] * (len(documents) - 1)
    if not isinstance(raw_scores, list):
        return [0.0] * len(documents)

    if raw_scores and isinstance(raw_scores[0], dict):
        scores = [0.0] * len(documents)
        for i, item in enumerate(raw_scores):
            if isinstance(item, dict) and "corpus_id" in item and "score" in item:
                corpus_id = int(item["corpus_id"])
                if 0 <= corpus_id < len(scores):
                    scores[corpus_id] = _coerce_score_value(item["score"])
            elif i < len(scores):
                scores[i] = _coerce_score_value(item)
        return scores

    scores = [_coerce_score_value(item) for item in raw_scores[: len(documents)]]
    if len(scores) < len(documents):
        scores.extend([0.0] * (len(documents) - len(scores)))
    return scores


class BGEReranker(BaseNodePostprocessor):
    """
    GGUF-based BGE Reranker using llama-cpp-python.
    Loads a GGUF quantized cross-encoder model with LLAMA_POOLING_TYPE_RANK
    for lightweight, fast reranking without FlagEmbedding / torch.
    """
    model_name: str
    top_n: int
    _model: object = None

    def __init__(self, model_name: str, top_n: int = 5):
        super().__init__(model_name=model_name, top_n=top_n)
        self.model_name = model_name
        self.top_n = top_n
        print(f"Initializing GGUF BGE Reranker: {model_name}")

        gguf_path = self._resolve_gguf_path()
        if not gguf_path:
            raise FileNotFoundError(
                f"Reranker GGUF not found. Expected at: {settings.LOCAL_RERANKER_MODEL_PATH}"
            )

        from llama_cpp import LLAMA_POOLING_TYPE_RANK
        from llama_cpp.llama_embedding import LlamaEmbedding
        import multiprocessing
        n_threads = max(1, multiprocessing.cpu_count() - 2)
        import platform
        if platform.system() == "Darwin":
            n_threads = min(n_threads, 4)

        self._model = LlamaEmbedding(
            model_path=gguf_path,
            pooling_type=LLAMA_POOLING_TYPE_RANK,
            n_ctx=512,
            n_batch=512,
            n_ubatch=512,
            n_threads=n_threads,
            verbose=False,
        )
        print(f"  GGUF reranker loaded: {gguf_path}")

    def _resolve_gguf_path(self) -> Optional[str]:
        path = settings.LOCAL_RERANKER_MODEL_PATH
        if path and os.path.isfile(path):
            return path
        return None

    @classmethod
    def class_name(cls) -> str:
        return "BGEReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")

        if not nodes:
            return []

        query_str = query_bundle.query_str
        documents = [node.node.get_content() for node in nodes]

        scores = _rank_with_llama(self._model, query_str, documents)

        for node, score in zip(nodes, scores):
            node.score = score

        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)

        if self.top_n is None or self.top_n < 0:
            return sorted_nodes
        return sorted_nodes[:self.top_n]

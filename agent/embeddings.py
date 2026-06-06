"""Embedding helpers.

Defaults to a small local sentence-transformers model so the pipeline runs
without any cloud key. If `OPENAI_API_KEY` is set and the user explicitly opts
in via `RESEARCH_EMBEDDING_PROVIDER=openai`, we use OpenAI's embedding API.

If no embedding backend is available, falls back to a deterministic hashed
bag-of-tokens vector so the rest of the pipeline (similarity comparisons) keeps
working in a degraded but functional state.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List, Optional, Sequence

import numpy as np
import requests

from . import config


_local_model = None
_local_load_failed = False


def _load_local_model():
    global _local_model, _local_load_failed
    if _local_model is not None or _local_load_failed:
        return _local_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _local_model = SentenceTransformer(config.EMBEDDING_LOCAL_MODEL)
    except Exception as e:  # noqa: BLE001
        print(f"  [embeddings] sentence-transformers unavailable ({e}); using hash fallback.")
        _local_load_failed = True
    return _local_model


def _provider() -> str:
    forced = os.environ.get(config.EMBEDDING_PROVIDER_ENV, "").strip().lower()
    if forced:
        return forced
    return "local"


def _embed_openai(texts: Sequence[str]) -> List[np.ndarray]:
    key = os.environ["OPENAI_API_KEY"].strip()
    payload = {
        "model": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "input": list(texts),
    }
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    data = r.json()
    return [np.asarray(d["embedding"], dtype=np.float32) for d in data.get("data", [])]


def _embed_local(texts: Sequence[str]) -> Optional[List[np.ndarray]]:
    model = _load_local_model()
    if model is None:
        return None
    vecs = model.encode(list(texts), normalize_embeddings=False, show_progress_bar=False)
    return [np.asarray(v, dtype=np.float32) for v in vecs]


_HASH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _embed_hash(texts: Sequence[str], dim: int = config.EMBEDDING_DIM_FALLBACK) -> List[np.ndarray]:
    """Deterministic hashed bag-of-words vectors. Cheap, unsupervised, but better than nothing."""
    out: List[np.ndarray] = []
    for t in texts:
        vec = np.zeros(dim, dtype=np.float32)
        tokens = _HASH_TOKEN_RE.findall((t or "").lower())
        if not tokens:
            out.append(vec)
            continue
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        out.append(vec)
    return out


def embed(texts: Sequence[str]) -> List[np.ndarray]:
    """Embed a batch of strings; never raises."""
    cleaned = [(t or "").strip() for t in texts]
    if not cleaned:
        return []

    provider = _provider()
    try:
        if provider == "openai" and os.environ.get("OPENAI_API_KEY", "").strip():
            return _embed_openai(cleaned)
    except Exception as e:  # noqa: BLE001
        print(f"  [embeddings] openai failed ({e}); falling back to local.")

    local = _embed_local(cleaned)
    if local is not None:
        return local

    return _embed_hash(cleaned)


def embed_one(text: str) -> np.ndarray:
    res = embed([text])
    if not res:
        return np.zeros(config.EMBEDDING_DIM_FALLBACK, dtype=np.float32)
    return res[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against each row in `mat`."""
    if mat is None or mat.size == 0 or query is None:
        return np.zeros(0, dtype=np.float32)
    qn = float(np.linalg.norm(query))
    if qn == 0.0:
        return np.zeros(mat.shape[0], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0.0] = 1e-9
    return (mat @ query) / (norms * qn)


def safe_dim(vec: np.ndarray) -> int:
    if vec is None:
        return config.EMBEDDING_DIM_FALLBACK
    if vec.ndim == 0:
        return 1
    return int(vec.shape[-1])


def softmax_norm(scores: Sequence[float]) -> List[float]:
    """Numerically stable softmax used for soft-ranking displays."""
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]

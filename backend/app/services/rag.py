import math
import re
from collections import Counter
from typing import Iterable, List

from ..schemas import RagSearchResult


def _tokenize(text: str) -> Counter:
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9_]+", text.lower())
    return Counter(tokens)


def _cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    intersection = set(a) & set(b)
    num = sum(a[t] * b[t] for t in intersection)
    denom = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / denom if denom else 0.0


def search_documents(docs: Iterable[dict], query: str, top_k: int = 3) -> List[RagSearchResult]:
    query_tokens = _tokenize(query)
    scored: list[tuple[dict, float]] = []
    for doc in docs:
        doc_tokens = _tokenize(doc["content"])
        score = _cosine_similarity(query_tokens, doc_tokens)
        snippet = doc["content"][:400]
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    results: List[RagSearchResult] = []
    for doc, score in scored[:top_k]:
        results.append(
            RagSearchResult(
                document_id=doc["id"],
                filename=doc["filename"],
                snippet=doc["content"][:500],
                score=score,
            )
        )
    return results

from math import sqrt
import re
from typing import Sequence

class SemanticValidator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device="cpu")
        except Exception:
            self.model_name = "local lexical fallback"

    def _lexical_vector(self, text: str) -> dict[str, float]:
        words = re.findall(r"[a-z0-9']+", text.lower())
        counts: dict[str, float] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        return counts

    def _fallback_similarity(self, left: str, right: str) -> float:
        a, b = self._lexical_vector(left), self._lexical_vector(right)
        if not a or not b:
            return 1.0 if left.strip() == right.strip() else 0.0
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        denom = sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
        return max(0.0, min(1.0, dot / denom if denom else 0.0))

    def similarity(self, original: str, optimized: str) -> float:
        if self.model is None:
            return self._fallback_similarity(original, optimized)
        vectors = self.model.encode([original, optimized], normalize_embeddings=True)
        score = float(vectors[0] @ vectors[1])
        return max(0.0, min(1.0, score))

    def pairwise(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) < 2:
            return [[1.0]] if texts else []
        if self.model is None:
            return [[self._fallback_similarity(a, b) for b in texts] for a in texts]
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [[float(vectors[i] @ vectors[j]) for j in range(len(texts))] for i in range(len(texts))]

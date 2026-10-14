from dataclasses import dataclass
import re
from .nlp_analyzer import Analysis

@dataclass
class ProtectedSpans:
    items: list[str]
    patterns: list[str]

class SpanProtector:
    def protect(self, analysis: Analysis) -> ProtectedSpans:
        items: list[str] = []
        items.extend(analysis.entities)
        items.extend(analysis.numbers)
        items.extend(analysis.quotes)
        items.extend(analysis.negations)
        items.extend(analysis.constraints)
        for pattern in (r"https?://\S+", r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+\b"):
            items.extend(re.findall(pattern, analysis.text))
        unique = list(dict.fromkeys(item.strip() for item in items if item.strip()))
        return ProtectedSpans(unique, ["entities", "numbers", "dates", "URLs", "email addresses", "quotes", "negations", "constraints"])

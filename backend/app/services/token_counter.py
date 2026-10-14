from typing import Protocol

class Tokenizer(Protocol):
    name: str
    def count(self, text: str) -> int: ...

class LocalTokenCounter:
    def __init__(self) -> None:
        self.name = "tiktoken cl100k_base"
        self._encoding = None
        try:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.name = "word-piece approximation"

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, round(len(text.split()) * 1.3))

"""LLMLingua-2 powered prompt compression for the Balanced/Aggressive modes.

Uses Microsoft's LLMLingua-2 encoder (``microsoft/llmlingua-2-xlm-roberta-large-meetingbank``),
a small (~560M param) token-classification model that drops low-information tokens while
keeping the ones a downstream LLM needs. Fully local: no API keys, CPU-friendly.

Protection strategy (two layers):
1. Native token constraints: every spaCy protected span is passed as ``force_tokens``
   (LLMLingua masks multi-token spans with ``[NEWi]`` placeholders internally and
   restores them afterwards; single-token spans get keep-probability 1.0).
   ``force_reserve_digit=True`` additionally pins every token containing a digit.
2. Post-compression guarantee: any protected span missing from the output is
   re-appended verbatim and reported, so the protection contract always holds.

Any failure (missing package, no disk/RAM for the checkpoint, inference error)
returns ``(None, reason)`` so the caller can gracefully fall back to the
deterministic spaCy pipeline. The model is loaded lazily on first use and cached
in ``~/.cache/huggingface`` like the Sentence Transformer model.

Env overrides:
    PROMPTLITE_LLMLINGUA_MODEL  HF model id (default: xlm-roberta-large meetingbank;
                                lighter alt: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank)
    PROMPTLITE_DISABLE_LLMLINGUA=1  disable LLMLingua entirely (deterministic only)
"""

from __future__ import annotations

import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
FALLBACK_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Keep-fraction per mode: balanced keeps ~60% of tokens (40% reduction),
# aggressive keeps ~35% (60-70% reduction, "caveman" style).
MODE_RATES: dict[str, float] = {"balanced": 0.6, "aggressive": 0.35}

MAX_FORCE_TOKENS = 100  # LLMLingua-2 default max_force_token ([NEW0]..[NEW99])
MIN_CHARS_FOR_LLM = 60  # shorter inputs compress poorly; leave them to spaCy


class LLMLinguaCompressor:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("PROMPTLITE_LLMLINGUA_MODEL", DEFAULT_MODEL)
        self._compressor = None
        self._lock = threading.Lock()
        self.available = False
        self.load_error: str | None = None
        self._disabled = os.environ.get("PROMPTLITE_DISABLE_LLMLINGUA") == "1"

    @property
    def disabled(self) -> bool:
        return self._disabled

    def _ensure_loaded(self) -> bool:
        if self._compressor is not None:
            return True
        if self._disabled:
            self.load_error = "disabled via PROMPTLITE_DISABLE_LLMLINGUA=1"
            return False
        with self._lock:
            if self._compressor is not None:
                return True
            try:
                from llmlingua import PromptCompressor as _LLMPromptCompressor
            except Exception as error:
                self.load_error = f"llmlingua package not importable: {error}"
                logger.warning("LLMLingua unavailable: %s", self.load_error)
                return False
            candidates = [self.model_name]
            if self.model_name == DEFAULT_MODEL:
                candidates.append(FALLBACK_MODEL)
            last_error: Exception | None = None
            for candidate in candidates:
                try:
                    self._compressor = _LLMPromptCompressor(
                        model_name=candidate,
                        device_map="cpu",
                        use_llmlingua2=True,
                    )
                    self.model_name = candidate
                    self.available = True
                    self.load_error = None
                    logger.info("LLMLingua-2 loaded: %s (cpu)", candidate)
                    return True
                except Exception as error:  # noqa: BLE001 - must fall back gracefully
                    last_error = error
                    logger.warning("LLMLingua-2 failed to load %s: %s", candidate, error)
            self.load_error = f"could not load LLMLingua-2 model: {last_error}"
            return False

    @staticmethod
    def _force_tokens(text: str, protected_items: list[str]) -> list[str]:
        """Dedupe protected spans to verbatim substrings, longest first, capped at 100."""
        seen: set[str] = set()
        tokens: list[str] = []
        for item in sorted((p.strip() for p in protected_items), key=len, reverse=True):
            if not item or item in seen or item not in text:
                continue
            seen.add(item)
            tokens.append(item)
            if len(tokens) >= MAX_FORCE_TOKENS:
                break
        return tokens

    @staticmethod
    def _retokenize(compressed: str, force_tokens: list[str]) -> str:
        """Fix gluing from extractive dropping: words fused onto protected spans
        ("Contactsupport@x.com") or dropped sentence spacing ("insights.Do")."""
        if force_tokens:
            alt = "|".join(sorted(map(re.escape, force_tokens), key=len, reverse=True))
            compressed = re.sub(r"(?<=[A-Za-z0-9])(" + alt + r")", r" \1", compressed)
            compressed = re.sub(r"(" + alt + r")(?=[A-Za-z])", r"\1 ", compressed)
        compressed = re.sub(r"(?<=[a-z])\.([A-Z])", r". \1", compressed)
        return re.sub(r"[ \t]+", " ", compressed).strip()

    def compress(
        self, text: str, protected_items: list[str], mode: str
    ) -> tuple[str | None, str | None]:
        """Compress with LLMLingua-2. Returns (compressed, note) or (None, reason)."""
        rate = MODE_RATES.get(mode)
        if rate is None:
            return None, f"mode '{mode}' is not an LLMLingua mode"
        if len(text.strip()) < MIN_CHARS_FOR_LLM:
            return None, "input too short for information-theoretic compression"
        if not self._ensure_loaded():
            return None, self.load_error or "LLMLingua unavailable"
        assert self._compressor is not None
        try:
            force_tokens = self._force_tokens(text, protected_items)
            with self._lock:
                result = self._compressor.compress_prompt(
                    [text],
                    rate=rate,
                    force_tokens=force_tokens,
                    force_reserve_digit=True,
                )
            compressed = result.get("compressed_prompt", "")
            if isinstance(compressed, list):
                compressed = "\n\n".join(compressed)
            compressed = re.sub(r"[ \t]+", " ", str(compressed)).strip()
            if not compressed:
                return None, "LLMLingua returned empty output"
            compressed = self._retokenize(compressed, force_tokens)
            # Layer 2: guarantee every protected span survives verbatim.
            missing = [p for p in force_tokens if p.lower() not in compressed.lower()]
            note = (
                f"LLMLingua-2 ({self.model_name}) keep-rate {rate}; "
                f"{len(force_tokens)} protected span(s) pinned"
            )
            if missing:
                compressed = compressed.rstrip() + "\nPreserved: " + "; ".join(missing)
                note += f"; re-attached {len(missing)} dropped protected span(s)"
            return compressed, note
        except Exception as error:  # noqa: BLE001 - caller falls back to spaCy
            logger.warning("LLMLingua compression failed, falling back: %s", error)
            return None, f"LLMLingua inference failed: {error}"

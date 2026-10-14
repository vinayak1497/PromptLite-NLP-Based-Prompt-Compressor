import re
from .nlp_analyzer import Analysis
from .protector import ProtectedSpans
from .redundancy_detector import RedundancyReport
from .redundancy_detector import RedundancyDetector


class PromptCompressor:
    @staticmethod
    def _short_protected(text: str, items: list[str], max_len: int = 80) -> tuple[str, dict[str, str]]:
        """Mask short atomic protected spans so aggressive word-dropping never
        touches entities/numbers/URLs/emails/quotes, while long constraint /
        negation sentences stay compressible (their numbers/entities are still
        masked individually via the short items)."""
        placeholders: dict[str, str] = {}
        masked = text
        # Longest first so nested spans resolve correctly.
        candidates = sorted(
            (p.strip() for p in items if p and p.strip()),
            key=len,
            reverse=True,
        )
        idx = 0
        for span in candidates:
            if len(span) > max_len:
                continue
            if span not in masked:
                continue
            key = f"__PX{idx}__"
            idx += 1
            placeholders[key] = span
            masked = masked.replace(span, key)
        return masked, placeholders

    @staticmethod
    def _restore(masked: str, placeholders: dict[str, str]) -> str:
        for key, span in placeholders.items():
            masked = masked.replace(key, span)
        return masked

    def compress(self, analysis: Analysis, protected: ProtectedSpans, redundancy: RedundancyReport, mode: str) -> tuple[str, list[str]]:
        kept = [s for i, s in enumerate(analysis.sentences) if i not in redundancy.duplicate_indexes]
        removed: list[str] = []
        if mode == "aggressive" and redundancy.duplicate_indexes:
            removed.append(f"Near-duplicate sentence(s) removed: {len(redundancy.duplicate_indexes)}")
        text = " ".join(kept).strip()

        passes = 1 if mode == "safe" else (1 if mode == "balanced" else 3)

        # 1) Polite/filler phrases (already detected by the redundancy pass).
        for filler in redundancy.filler_matches:
            pattern = re.compile(r"\b" + re.escape(filler) + r"\b[, ]*", re.I)
            text, count = pattern.subn("", text)
            if count:
                removed.append(f"Conversational filler: {filler}")

        # 2) Multiword verbose constructions, run a couple of times for nesting.
        # Aggressive runs its longer deletion patterns FIRST so they win over the
        # shared map (e.g. "a great deal of different" -> "many" beats
        # "a great deal of" -> "much").
        if mode == "aggressive":
            aggressive_map = sorted(
                RedundancyDetector.AGGRESSIVE_MULTIWORD.items(),
                key=lambda kv: len(kv[0]),
                reverse=True,
            )
            for _ in range(2):
                for source, target in aggressive_map:
                    if not source:
                        continue
                    text, count = re.subn(r"\b" + re.escape(source) + r"\b", target, text, flags=re.I)
                    if count:
                        removed.append(f"Aggressive simplification: {source} -> {target or '(removed)'}")
        for _ in range(passes):
            for source, target in RedundancyDetector.MULTIWORD.items():
                if not source:
                    continue
                text, count = re.subn(r"\b" + re.escape(source) + r"\b", target, text, flags=re.I)
                if count:
                    removed.append(f"Verbose construction simplified: {source} -> {target or '(removed)'}")

        # 3) Sentence lead-ins / polite prefaces that add no task meaning.
        for leadin in RedundancyDetector.LEADINS:
            text, count = re.subn(leadin, "", text, flags=re.I)
            if count:
                removed.append(f"Redundant preface removed: {leadin.rstrip(' ,?')}")

        # 4) Redundant "please ensure/make sure that" before verbs.
        text, count = re.subn(RedundancyDetector.REQUIREMENT + r"\s*", "", text, flags=re.I)
        if count:
            removed.append("Redundant requirement phrase removed")

        # 5) Hedging / filler adverb removal (balanced trims a little, aggressive trims harder).
        if mode in {"balanced", "aggressive"}:
            text, count = re.subn(
                r"\b(please|kindly|essentially|basically|actually|in fact|hopefully)\b\s*", "", text, flags=re.I
            )
            if count:
                removed.append("Hedge word removed")
        if mode == "aggressive":
            text, count = RedundancyDetector.INTENSIFIERS.subn("", text)
            if count:
                removed.append("Intensifier removed")
            text, count = re.subn(r"\b(ultimately|eventually|obviously|simply|truly|generally|particularly)\b\s*", "", text, flags=re.I)
            if count:
                removed.append("Hedge adverb removed")

        # 5b) Redundant manner adverbs / "in a ... way" constructions (aggressive only).
        if mode == "aggressive":
            text, count = RedundancyDetector.MANNER_PHRASE.subn("", text)
            if count:
                removed.append("Verbose manner phrase removed")
            text, count = re.subn(RedundancyDetector.MANNER_ADVERBS + r"\s*,?\s*", "", text, flags=re.I)
            if count:
                removed.append("Redundant manner adverb removed")

        # 5c) Second aggressive sweep catches phrases exposed by earlier
        # stripping (e.g. after intensifier/manner removal).
        if mode == "aggressive":
            aggressive_map = sorted(
                RedundancyDetector.AGGRESSIVE_MULTIWORD.items(),
                key=lambda kv: len(kv[0]),
                reverse=True,
            )
            for source, target in aggressive_map:
                if not source:
                    continue
                text, count = re.subn(r"\b" + re.escape(source) + r"\b", target, text, flags=re.I)
                if count:
                    removed.append(f"Aggressive simplification: {source} -> {target or '(removed)'}")

        # 5d) Parenthetical asides with no protected content (aggressive only).
        if mode == "aggressive":
            def _paren_replace(match: re.Match) -> str:
                inner = match.group(0)
                # Keep anything carrying digits, URLs, emails, or @-handles.
                if re.search(r"\d|https?://|@\w|\w@\w", inner):
                    return inner
                removed.append("Parenthetical aside removed")
                return " "

            text = re.sub(r"\([^()]{2,200}\)", _paren_replace, text)

        # 5e) Protected-masked word thinning (aggressive only): articles,
        # low-value adjectives, relativizer "that", and additive "also".
        # Short protected spans are masked first so entities / numbers /
        # URLs / emails can never be touched; long sentences stay open so
        # filler inside them is still compressible.
        if mode == "aggressive":
            masked, placeholders = self._short_protected(text, protected.items)
            masked, count = re.subn(r"\b(a|an|the)\b\s*", "", masked, flags=re.I)
            if count:
                removed.append(f"Articles removed (aggressive): {count}")
            masked, count = RedundancyDetector.AGGRESSIVE_ADJECTIVES.subn("", masked)
            if count:
                removed.append(f"Low-value adjectives removed (aggressive): {count}")
            masked, count = re.subn(r"\bthat\b\s*", "", masked, flags=re.I)
            if count:
                removed.append(f"Relativizer 'that' removed (aggressive): {count}")
            masked, count = re.subn(r"\balso\b\s*", "", masked, flags=re.I)
            if count:
                removed.append(f"Additive 'also' removed (aggressive): {count}")
            text = self._restore(masked, placeholders)

        # 6) Collapse repeated spaces/commas that removal leaves behind.
        text = re.sub(r"\b\s+([,.;:])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+([?!])", r"\1", text)
        text = re.sub(r"\b(?:a|an|the)\s+(?=\b(?:a|an|the)\b)", "", text)
        text = re.sub(r"([,;])\s*([,;])+", r"\1", text)
        text = re.sub(r"\b(and|or|but)\s*,?\s+\1\b", r"\1", text)
        if mode == "aggressive":
            # Dangling conjunctions left when both adjectives around "and"
            # are deleted ("detailed and comprehensive" -> "and").
            text = re.sub(r"\b(with|of|to|for|in|pay|provide|and|or)\s+(and|or)\b", r"\1", text, flags=re.I)
            text = re.sub(r"([?.!]\s+)(and|or)\b\s*", r"\1", text)
            text = re.sub(r"^(and|or)\b\s*", "", text)
            text = re.sub(r"\b(and|or)\b\s*(?=[,.;:!?])", "", text)
            text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r"([,;])\s*([,;])+", r"\1", text)
        text = re.sub(r"[,;:]\s*(?=[?])", r"?", text)
        text = re.sub(r"\s+([?!])", r"\1", text)
        text = text.strip(" ,;:.")

        # 7) Task prefix (only when a task is genuinely detected; skipped in
        # aggressive mode to save tokens — the task verb already survives).
        if mode != "aggressive" and analysis.task != "Other / general" and len(analysis.sentences) > 1:
            prefix = f"{analysis.task}: "
            if not re.match(r"^\w+:\s", text):
                text = prefix + text

        return text, removed

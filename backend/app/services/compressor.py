import re
from .nlp_analyzer import Analysis
from .protector import ProtectedSpans
from .redundancy_detector import RedundancyReport
from .redundancy_detector import RedundancyDetector


class PromptCompressor:
    def compress(self, analysis: Analysis, protected: ProtectedSpans, redundancy: RedundancyReport, mode: str) -> tuple[str, list[str]]:
        kept = [s for i, s in enumerate(analysis.sentences) if i not in redundancy.duplicate_indexes]
        removed: list[str] = []
        text = " ".join(kept).strip()

        passes = 1 if mode == "safe" else (1 if mode == "balanced" else 2)

        # 1) Polite/filler phrases (already detected by the redundancy pass).
        for filler in redundancy.filler_matches:
            pattern = re.compile(r"\b" + re.escape(filler) + r"\b[, ]*", re.I)
            text, count = pattern.subn("", text)
            if count:
                removed.append(f"Conversational filler: {filler}")

        # 2) Multiword verbose constructions, run a couple of times for nesting.
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

        # 6) Collapse repeated spaces/commas that removal leaves behind.
        text = re.sub(r"\b\s+([,.;:])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+([?!])", r"\1", text)
        text = re.sub(r"\b(?:a|an|the)\s+(?=\b(?:a|an|the)\b)", "", text)
        text = re.sub(r"([,;])\s*([,;])+", r"\1", text)
        text = re.sub(r"\b(and|or|but)\s*,?\s+\1\b", r"\1", text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r"([,;])\s*([,;])+", r"\1", text)
        text = re.sub(r"[,;:]\s*(?=[?])", r"?", text)
        text = re.sub(r"\s+([?!])", r"\1", text)
        text = text.strip(" ,;:.")

        # 7) Task prefix (only when a task is genuinely detected).
        if analysis.task != "Other / general" and len(analysis.sentences) > 1:
            prefix = f"{analysis.task}: "
            if not re.match(r"^\w+:\s", text):
                text = prefix + text

        return text, removed

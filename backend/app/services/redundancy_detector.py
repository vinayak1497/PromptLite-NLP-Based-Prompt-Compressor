from dataclasses import dataclass
import re
from .semantic_validator import SemanticValidator

@dataclass
class RedundancyReport:
    duplicate_indexes: set[int]
    filler_matches: list[str]
    consolidated: list[str]

class RedundancyDetector:
    FILLER = re.compile(
        r"\b("
        r"could you please|would you please|can you please|could you|could i get you to|"
        r"please|if you don't mind|if you dont mind|if possible|if it is possible|if its possible|"
        r"i would like you to|i would like to ask you to|i would like to tell you to|i would like|i want you to|"
        r"i 'd like you to|i 'd appreciate it if you could|i 'd appreciate it if|it would be great if you could|"
        r"it would be great if|it would be nice if|it would be helpful if|it would be appreciated if|"
        r"i was wondering if you could|i was wondering if|just wondering if|"
        r"kindly|please kindly|do me a favor and|be so kind as to|"
        r"in the event that|for the purpose of|with regard to|with regards to|in regards to|in relation to|"
        r"at this point in time|at the present time|in the near future|at the end of the day|"
        r"as a matter of fact|in fact|actually|basically|essentially|basically essentially"
        r")\b", re.I)

    INTENSIFIERS = re.compile(r"\b(very|really|quite|extremely|highly|truly|just|simply|totally|absolutely|definitely|indeed|literally)\b", re.I)
    MULTIWORD = {
        "a large number of": "many",
        "a great deal of": "much",
        "a lot of": "many",
        "lots of": "many",
        "plenty of": "many",
        "a majority of": "most",
        "the majority of": "most",
        "a number of": "some",
        "a handful of": "a few",
        "due to the fact that": "because",
        "on the grounds that": "because",
        "for the reason that": "because",
        "owing to the fact that": "because",
        "in order to": "to",
        "in order for": "for",
        "in spite of": "despite",
        "in addition to": "besides",
        "with the exception of": "except",
        "in the event that": "if",
        "in the case that": "if",
        "at this point in time": "now",
        "at the present time": "now",
        "in the near future": "soon",
        "at the end of the day": "ultimately",
        "make sure that": "ensure",
        "make sure": "ensure",
        "take into consideration": "consider",
        "give consideration to": "consider",
        "with respect to": "about",
        "in terms of": "as",
        "prior to": "before",
        "subsequent to": "after",
        "regardless of the fact that": "although",
        "in the process of": "",
    }
    LEADINS = [
        r"please be advised that ", r"please note that ", r"please remember that ",
        r"i would like to inform you that ", r"i just wanted to let you know that ",
        r"just so you know,? ", r"as you know,? ", r"as you may know,? ",
        r"it is important to note that ", r"it should be noted that ", r"it is worth mentioning that ",
        r"needless to say,? ", r"of course,? ", r"in conclusion,? ", r"to summarize,? ",
        r"in my opinion,? ", r"i think that ", r"i believe that ", r"i feel that ",
        r"i would say that ", r"i 'd say that ", r"in my view,? ",
        r"i need you to ", r"i would need you to ", r"can you ", r"could you ",
        r"okay thanks|ok thanks|thanx|thanks so much|thank you very much|thank you|no problem",
    ]
    REQUIREMENT = r"\b(please make sure that|please ensure that|please confirm that|please be certain that)\b"
    MANNER_ADVERBS = r"\b(carefully|thoroughly|clearly|concisely|properly|correctly|deeply|fully|completely|promptly|quickly|easily|simply|diligently|accurately|precisely)\b"
    MANNER_PHRASE = re.compile(r"\bin (?:a |an |the )?(?:very |really |highly )?[^,.;:!?]+? (?:way|manner|fashion)\b", re.I)

    def __init__(self, validator: SemanticValidator) -> None:
        self.validator = validator

    def detect(self, sentences: list[str], mode: str) -> RedundancyReport:
        duplicates: set[int] = set()
        matches = self.FILLER.findall(" ".join(sentences))
        if len(sentences) > 1:
            matrix = self.validator.pairwise(sentences)
            threshold = {"safe": .94, "balanced": .88, "aggressive": .80}[mode]
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    if matrix[i][j] >= threshold:
                        duplicates.add(j)
        consolidated = []
        if len(sentences) > 1 and any(re.search(r"\b(careful|detailed|thorough)\b", s, re.I) for s in sentences):
            if sum(bool(re.search(r"\b(careful|detailed|thorough)\b", s, re.I)) for s in sentences) > 1:
                consolidated.append("Repeated care/detail instructions consolidated")
        return RedundancyReport(duplicates, list(dict.fromkeys(matches)), consolidated)

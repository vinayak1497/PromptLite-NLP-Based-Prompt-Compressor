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

    INTENSIFIERS = re.compile(r"\b(very|really|quite|extremely|highly|truly|just|simply|totally|absolutely|definitely|indeed|literally|greatly|deeply|widely|fully|completely|totally|entirely|utterly|remarkably|exceptionally|incredibly)\b", re.I)
    MULTIWORD = {
        "a large number of": "many",
        "a great deal of": "much",
        "a good deal of": "much",
        "a large amount of": "much",
        "a lot of": "many",
        "lots of": "many",
        "plenty of": "many",
        "a majority of": "most",
        "the majority of": "most",
        "a number of": "some",
        "a handful of": "a few",
        "a wide variety of": "many",
        "wide variety of": "many",
        "a variety of": "many",
        "large number of": "many",
        "great deal of": "much",
        "due to the fact that": "because",
        "on the grounds that": "because",
        "for the reason that": "because",
        "owing to the fact that": "because",
        "in light of the fact that": "because",
        "given the fact that": "because",
        "on account of": "because",
        "in order to be able to": "to",
        "in order to": "to",
        "in order for": "for",
        "to be able to": "to",
        "be able to": "",
        "in spite of": "despite",
        "in addition to": "besides",
        "with the exception of": "except",
        "in the event that": "if",
        "in the case that": "if",
        "at this point in time": "now",
        "at the present time": "now",
        "at the present moment": "now",
        "at the moment": "now",
        "in the near future": "soon",
        "in the foreseeable future": "soon",
        "at the earliest convenience": "soon",
        "as soon as possible": "ASAP",
        "at the end of the day": "ultimately",
        "make sure that": "ensure",
        "make certain that": "ensure",
        "ensure that": "ensure",
        "make sure": "ensure",
        "take into consideration": "consider",
        "give consideration to": "consider",
        "with respect to": "about",
        "in terms of": "as",
        "prior to": "before",
        "subsequent to": "after",
        "regardless of the fact that": "although",
        "in spite of the fact that": "although",
        "in the process of": "",
        "during the course of": "during",
        "in the course of": "during",
        "over the course of": "during",
        "for a period of": "for",
        "each and every single": "each",
        "each and every": "each",
        "every single": "every",
        "each individual": "each",
        "first and foremost": "first",
        "as well as": "and",
        "along with": "with",
        "in conjunction with": "with",
        "by means of": "via",
        "for example": "e.g.",
        "for instance": "e.g.",
        "in connection with": "about",
        "make an effort to": "try to",
        "make an attempt to": "try to",
        "conduct an analysis of": "analyze",
        "perform an analysis of": "analyze",
        "provide a summary of": "summarize",
        "give a summary of": "summarize",
        "it is important to note that": "",
        "it is worth mentioning that": "",
        "it is worth noting that": "",
        "it goes without saying that": "",
        "in other words": "",
        "a sufficient number of": "enough",
        "have a tendency to": "tend to",
        "in all likelihood": "likely",
    }
    # Aggressive-only simplifications: applied solely in aggressive mode, after
    # the shared MULTIWORD pass. Values of "" delete the phrase outright.
    AGGRESSIVE_MULTIWORD = {
        "in great detail": "",
        "in a great deal of detail": "",
        "in detail": "",
        "with a lot of attention to": "",
        "with a lot of attention": "",
        "every single little": "",
        "single little": "",
        "at the end of the day": "",
        "ultimately": "",
        "what i am really looking for is": "",
        "what i am looking for is": "",
        "what i really want is for you to": "",
        "what i really want is": "",
        "i am really looking for": "",
        "would be great if you could": "should",
        "it would be great if you could": "",
        "it is also worth mentioning that": "",
        "much different": "many",
        "note that": "",
        "note it would be": "",
        "in the event that there are": "if there are",
        "there are a great deal of different": "there are many",
        "a great deal of different": "many",
        "wide variety of different": "many",
        "large number of different": "many",
        "great deal of different": "many",
        "different opinions from": "opinions from",
        "in a clear and concise manner": "clearly",
        "in a clear concise and easily understandable manner": "clearly",
        "for the purpose of sharing with": "for",
        "for the purpose of sharing": "for",
        "in order to ensure the highest possible level of": "for",
        "highest possible level of": "",
        "in order to be successful": "",
        "in order to understand": "to understand",
        "to be able to understand": "to understand",
        "is very important to": "must",
        "is really great if you could": "should",
        "would be really great if you could": "should",
        "it would be great if you could": "",
        "i was wondering if you could possibly": "",
        "i was wondering if you could": "",
        "as well as a comprehensive breakdown of": "and",
        "comprehensive breakdown of": "breakdown of",
        "comprehensive overview that covers": "overview of",
        "potential areas of concern": "risks",
        "areas of concern": "risks",
        "hidden issues that might need further investigation": "hidden issues",
        "each and every important aspect": "everything",
        "each and every": "each",
        "each individual piece of": "each",
        "as soon as possible": "ASAP",
        "with respect to product quality": "on quality",
        "please keep in mind that": "",
        "keep in mind that": "",
        "that i need to complete for my business": "",
        "that i have provided in the attached document": "provided",
        "that i have provided": "provided",
        "that customers have made for": "from customers for",
        "that they do about each specific feature in great detail": "about each feature",
        "very close and careful attention to": "attention to",
        "very thorough and diligent manner": "",
        "in a very thorough and diligent manner": "",
        "at this point in time": "now",
    }
    # Low-information adjectives safely droppable in aggressive mode only.
    # "that"-clause filler + articles are handled as regexes in the compressor.
    AGGRESSIVE_ADJECTIVES = re.compile(
        r"\b(different|various|diverse|specific|particular|respective|comprehensive|"
        r"detailed|thorough|careful|complete|entire|whole|additional|extra|further|"
        r"overall|general|broad|great|little|single|individual|respective|relevant|"
        r"wide|overall|possible|potential|available|close|proper|properly)\b",
        re.I,
    )
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
            threshold = {"safe": .94, "balanced": .88, "aggressive": .72}[mode]
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    if matrix[i][j] >= threshold:
                        duplicates.add(j)
        consolidated = []
        if len(sentences) > 1 and any(re.search(r"\b(careful|detailed|thorough)\b", s, re.I) for s in sentences):
            if sum(bool(re.search(r"\b(careful|detailed|thorough)\b", s, re.I)) for s in sentences) > 1:
                consolidated.append("Repeated care/detail instructions consolidated")
        return RedundancyReport(duplicates, list(dict.fromkeys(matches)), consolidated)

from dataclasses import dataclass, field
import re
import spacy

@dataclass
class Analysis:
    text: str
    sentences: list[str]
    entities: list[str] = field(default_factory=list)
    noun_phrases: list[str] = field(default_factory=list)
    negations: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    imperative_verbs: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    task: str = "Other / general"

class NLPAnalyzer:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name)
        except Exception:
            self.model_name = "spacy blank English"
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")

    def analyze(self, text: str) -> Analysis:
        doc = self.nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        entities = [ent.text for ent in doc.ents]
        try:
            noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks]
        except ValueError:
            # Blank pipelines have no dependency parse; retain useful lexical candidates.
            noun_phrases = [token.text for token in doc if token.is_alpha and len(token.text) > 2]
        negations = [token.sent.text.strip() for token in doc if token.dep_ == "neg" or token.text.lower() in {"not", "never", "without", "no"}]
        numbers = [token.text for token in doc if token.like_num]
        quotes = re.findall(r"[\"']([^\"']+)[\"']", text)
        imperative_verbs = [token.lemma_ or token.text for token in doc if token.pos_ == "VERB" and token.dep_ == "ROOT"]
        constraints = [s for s in sentences if re.search(r"\b(do not|don't|must|only|at least|no more than|return exactly|in json|include|exclude|required)\b", s, re.I)]
        return Analysis(text, sentences, entities, noun_phrases, negations, numbers, quotes, imperative_verbs, constraints, self.detect_task(text))

    def detect_task(self, text: str) -> str:
        patterns = [("Summarization", r"\b(summarize|summary|brief|key points)\b"), ("Explanation", r"\b(explain|clarify|why|how does)\b"), ("Analysis", r"\b(analy[sz]e|analysis|evaluate|insight)\b"), ("Comparison", r"\b(compare|contrast|difference|versus|vs\.)\b"), ("Classification", r"\b(classif|categorize|label|sentiment)\w*\b"), ("Extraction", r"\b(extract|identify|list|find)\b"), ("Question answering", r"\b(answer|question|what is|who is)\b"), ("Code generation", r"\b(code|function|program|implement|sql|python)\b"), ("Writing / rewrite", r"\b(write|rewrite|draft|compose|edit)\b"), ("Translation", r"\b(translat|language)\w*\b")]
        for name, pattern in patterns:
            if re.search(pattern, text, re.I):
                return name
        return "Other / general"

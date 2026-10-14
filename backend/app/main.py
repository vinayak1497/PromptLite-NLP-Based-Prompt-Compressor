from pathlib import Path
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .schemas.api import CompressRequest, CompressResponse, HealthResponse
from .services.nlp_analyzer import NLPAnalyzer
from .services.protector import SpanProtector
from .services.semantic_validator import SemanticValidator
from .services.redundancy_detector import RedundancyDetector
from .services.compressor import PromptCompressor
from .services.llmlingua_compressor import LLMLinguaCompressor, MODE_RATES
from .services.token_counter import LocalTokenCounter
from .services.document_processor import DocumentProcessor

app = FastAPI(title="PromptLite API", version="1.0.0")
# Comma-separated extra origins via ALLOWED_ORIGINS env var (e.g. the Vercel
# frontend URL in production). Localhost defaults keep local dev working.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])
analyzer = NLPAnalyzer()
validator = SemanticValidator()
protector = SpanProtector()
detector = RedundancyDetector(validator)
compressor = PromptCompressor()
llm_compressor = LLMLinguaCompressor()
tokens = LocalTokenCounter()
documents = DocumentProcessor()

# Minimum MiniLM cosine similarity before softening to a safer mode.
# Balanced/Aggressive now use information-theoretic token dropping, whose
# intentionally ungrammatical ("caveman") output scores lower than a
# grammatical rewrite, so their floors sit below Safe's.
SIMILARITY_FLOOR = {"safe": .90, "balanced": .80, "aggressive": .68}

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        spacy_model=analyzer.model_name,
        embedding_model=validator.model_name,
        embedding_available=validator.model is not None,
        tokenizer=tokens.name,
        llmlingua_available=llm_compressor.available,
        llmlingua_model=llm_compressor.model_name,
        llmlingua_error=llm_compressor.load_error,
    )

def run_compression(text: str, mode: str) -> CompressResponse:
    if not text.strip():
        raise HTTPException(status_code=422, detail="Text must contain at least one non-whitespace character.")
    analysis = analyzer.analyze(text)
    protected = protector.protect(analysis)
    redundancy = detector.detect(analysis.sentences, mode)
    warnings: list[str] = []
    if mode == "safe":
        # Grammatical, conservative: deterministic spaCy pipeline only.
        optimized, removed = compressor.compress(analysis, protected, redundancy, mode)
        engine = "spacy-deterministic"
    else:
        # Balanced/Aggressive: local LLMLingua-2 token dropping with protected
        # spans pinned via force_tokens (+ digit pinning). Falls back to the
        # deterministic pipeline if the model cannot load (e.g. RAM/disk limits).
        llm_text, llm_note = llm_compressor.compress(text, protected.items, mode)
        if llm_text is not None:
            optimized = llm_text
            engine = f"llmlingua-2 (keep-rate {MODE_RATES[mode]}, {llm_compressor.model_name})"
            removed = [llm_note or engine]
        else:
            optimized, removed = compressor.compress(analysis, protected, redundancy, mode)
            engine = "spacy-deterministic"
            warnings.append(f"LLMLingua unavailable ({llm_note}); fell back to deterministic spaCy compression.")
    score = validator.similarity(text, optimized)
    used_mode = mode
    if score < SIMILARITY_FLOOR[mode] and mode != "safe":
        fallback_mode = "safe" if mode == "balanced" else "balanced"
        fallback_report = detector.detect(analysis.sentences, fallback_mode)
        fallback, fallback_removed = compressor.compress(analysis, protected, fallback_report, fallback_mode)
        fallback_score = validator.similarity(text, fallback)
        if fallback_score >= score:
            optimized, removed, score, used_mode = fallback, fallback_removed, fallback_score, fallback_mode
            engine = "spacy-deterministic (semantic safety fallback)"
        warnings.append(f"Compression was softened to {used_mode} to protect semantic preservation.")
    original_count, optimized_count = tokens.count(text), tokens.count(optimized)
    reduction = max(0.0, round((1 - optimized_count / original_count) * 100, 1)) if original_count else 0.0
    return CompressResponse(
        optimized_text=optimized,
        original_tokens=original_count,
        optimized_tokens=optimized_count,
        token_reduction_percentage=reduction,
        compression_ratio=round(optimized_count / original_count, 3) if original_count else 1.0,
        semantic_similarity=round(score * 100, 1),
        detected_task=analysis.task,
        protected_items=protected.items,
        removed_items_summary=[f"Engine: {engine}"] + removed + redundancy.consolidated,
        warnings=warnings,
        mode_used=used_mode,
    )

@app.post("/api/compress", response_model=CompressResponse)
def compress(request: CompressRequest) -> CompressResponse:
    text = f"{request.task.strip()}\n\n{request.text}" if request.task and request.task.strip() else request.text
    return run_compression(text, request.mode)

@app.post("/api/upload", response_model=CompressResponse)
async def upload(file: UploadFile = File(...), mode: str = "balanced", task: str | None = None) -> CompressResponse:
    if mode not in {"safe", "balanced", "aggressive"}:
        raise HTTPException(status_code=422, detail="Mode must be safe, balanced, or aggressive.")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")
    try:
        text = documents.clean(documents.extract(file.filename or "upload.txt", content))
    except ValueError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    if task and task.strip():
        text = f"Task: {task.strip()}\n\n{text}"
    return run_compression(text, mode)

from typing import Literal
from pydantic import BaseModel, Field

CompressionMode = Literal["safe", "balanced", "aggressive"]

class CompressRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    mode: CompressionMode = "balanced"
    task: str | None = Field(default=None, max_length=2_000)

class CompressResponse(BaseModel):
    optimized_text: str
    original_tokens: int
    optimized_tokens: int
    token_reduction_percentage: float
    compression_ratio: float
    semantic_similarity: float
    detected_task: str
    protected_items: list[str]
    removed_items_summary: list[str]
    warnings: list[str]
    mode_used: CompressionMode

class HealthResponse(BaseModel):
    status: str
    spacy_model: str
    embedding_model: str
    embedding_available: bool
    tokenizer: str
    llmlingua_available: bool = False
    llmlingua_model: str = ""
    llmlingua_error: str | None = None

from app.main import run_compression
from fastapi.testclient import TestClient
from app.main import app
import pytest


def test_compress_with_task():
    client = TestClient(app)
    response = client.post("/api/compress", json={"text": "Please summarize the notes.", "mode": "aggressive", "task": "Extract key points"})
    assert response.status_code == 200
    body = response.json()
    assert body["optimized_tokens"] < body["original_tokens"]

def test_filler_removed():
    result = run_compression("Could you please analyze the report?", "balanced")
    assert "Could you please" not in result.optimized_text

def test_protected_items():
    result = run_compression("Do not remove 42 records from Microsoft Azure.", "aggressive")
    assert "42" in result.optimized_text
    assert any("Microsoft" in item or "Azure" in item for item in result.protected_items)

def test_duplicate_sentence_removed():
    result = run_compression("Analyze the data carefully. Analyze the data carefully.", "safe")
    assert result.optimized_text.count("Analyze") <= 2

def test_empty_rejected():
    with pytest.raises(Exception):
        run_compression(" ", "balanced")

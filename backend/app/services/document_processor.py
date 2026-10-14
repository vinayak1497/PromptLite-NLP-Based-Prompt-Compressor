from io import BytesIO
from pathlib import Path
import re

class DocumentProcessor:
    def extract(self, filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md"}:
            return content.decode("utf-8", errors="replace")
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        raise ValueError("Unsupported file type. Upload a TXT or PDF document.")

    def clean(self, text: str) -> str:
        return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()

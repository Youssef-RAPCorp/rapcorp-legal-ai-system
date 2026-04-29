"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    CASE DIRECTORY SCANNER                                     ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Reads all legal documents from a directory and extracts their text          ║
║  content for AI-based case analysis.                                          ║
║                                                                               ║
║  Supported formats:                                                           ║
║  • .txt / .md / .rtf / .csv / .json / .html  — read directly                ║
║  • .pdf                                       — extracted via pypdf           ║
║  • .docx / .doc                               — extracted via python-docx    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import io
import zipfile
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".rtf", ".csv", ".json", ".html", ".htm", ".log",
    ".pdf",
    ".docx", ".doc",
}

MAX_CHARS_PER_DOC = 60_000  # Truncate very large docs to keep prompt manageable
MAX_TOTAL_CHARS   = 150_000 # Safety cap on total context fed to AI

# Filename keywords → document type label
_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "petition":       ["petition", "complaint", "original_petition"],
    "response":       ["response", "answer", "reply", "opposition", "objection"],
    "court_order":    ["order", "ruling", "judgment", "decree", "decision", "mandate"],
    "motion":         ["motion"],
    "affidavit":      ["affidavit", "declaration", "sworn", "verification"],
    "summons":        ["summons", "notice_of"],
    "exhibit":        ["exhibit"],
    "checklist":      ["checklist"],
    "cover_sheet":    ["cover_sheet", "cover sheet"],
    "proposed_order": ["proposed_order"],
    "transcript":     ["transcript"],
    "brief":          ["brief", "memorandum", "memo"],
    "evidence_plan":  ["document_plan"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseDocument:
    """A single document read from the case directory."""
    filename: str
    file_path: str
    doc_type: str           # Inferred: petition, response, court_order, motion, etc.
    text_content: str       # Extracted text (may be truncated)
    char_count: int
    truncated: bool = False
    read_error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL READERS
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_doc_type(filename: str) -> str:
    name = filename.lower().replace(" ", "_").replace("-", "_")
    for doc_type, keywords in _TYPE_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return doc_type
    return "document"


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
    # OSError / PermissionError propagate to _read_file, which sets read_error


_SCANNED_PDF_THRESHOLD = 200  # chars; below this, pypdf result is considered insufficient


def _read_pdf(path: str, gemini_api_key: str = "") -> str:
    """
    Extract text from a PDF.

    Strategy:
    1. Try pypdf (fast, free, works on text-layer PDFs).
    2. If pypdf yields fewer than _SCANNED_PDF_THRESHOLD chars (scanned/image PDF),
       fall back to Gemini Flash via the File API, which can OCR scanned documents.
    3. If no API key is available, return whatever pypdf managed to extract.
    """
    pypdf_text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        pypdf_text = "\n\n".join(pages)
    except ImportError:
        pypdf_text = ""
    except Exception:
        pypdf_text = ""

    if len(pypdf_text) >= _SCANNED_PDF_THRESHOLD:
        return pypdf_text  # Good enough — no need for Gemini

    # pypdf got little/nothing — try Gemini File API for scanned PDFs
    if not gemini_api_key:
        return pypdf_text or "[PDF appears to be a scanned image; set GOOGLE_API_KEY to enable OCR extraction]"

    try:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        genai.configure(api_key=gemini_api_key)

        uploaded = genai.upload_file(path=path, mime_type="application/pdf")

        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(
            [
                uploaded,
                "Extract ALL text from this document exactly as written. "
                "Preserve paragraph structure. Do not summarize, omit, or reformat — "
                "output the complete verbatim text content only.",
            ],
            generation_config=GenerationConfig(temperature=0.0, max_output_tokens=8192),
        )

        extracted = response.text.strip() if hasattr(response, "text") else ""

        # Clean up the uploaded file from Gemini's servers
        try:
            genai.delete_file(uploaded.name)
        except Exception:
            pass

        return extracted if extracted else (pypdf_text or "[Gemini extraction returned empty result]")

    except Exception as e:
        return pypdf_text or f"[PDF extraction failed: {e}]"


def _read_docx(path: str) -> str:
    # Try python-docx first
    try:
        import docx
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except ImportError:
        pass
    except Exception as e:
        return f"[DOCX read error: {e}]"

    # Fallback: unzip and strip XML tags from word/document.xml
    try:
        with open(path, "rb") as f:
            raw = f.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if "word/document.xml" in z.namelist():
                xml = z.read("word/document.xml").decode("utf-8", errors="replace")
                text = re.sub(r"<[^>]+>", " ", xml)
                text = re.sub(r"\s+", " ", text).strip()
                return text
    except Exception as e:
        return f"[DOCX fallback read error: {e}]"

    return "[DOCX extraction requires python-docx: pip install python-docx]"


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

class CaseDirectoryScanner:
    """
    Scans a directory for legal documents and extracts their text content.

    Usage:
        scanner = CaseDirectoryScanner(gemini_api_key="...")
        documents = scanner.scan("/path/to/case/folder")
        stats = scanner.get_stats(documents)
    """

    def __init__(self, gemini_api_key: str = ""):
        self._gemini_api_key = gemini_api_key

    def scan(self, directory: str) -> List[CaseDocument]:
        """
        Scan the directory for all supported documents and extract text.

        Files are sorted by name (which usually reflects filing order when
        numbered, e.g. 01_petition.txt, 02_response.txt).

        Returns a list of CaseDocument objects.
        Raises FileNotFoundError / NotADirectoryError if the path is invalid.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        files = sorted(
            [f for f in dir_path.iterdir()
             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS],
            key=lambda f: f.name.lower(),
        )

        documents: List[CaseDocument] = []
        total_chars = 0

        for file_path in files:
            if total_chars >= MAX_TOTAL_CHARS:
                # Still register the document but note it was skipped
                documents.append(CaseDocument(
                    filename=file_path.name,
                    file_path=str(file_path),
                    doc_type=_infer_doc_type(file_path.name),
                    text_content="",
                    char_count=0,
                    read_error="Skipped — total context limit reached",
                ))
                continue

            doc = self._read_file(file_path)
            documents.append(doc)
            total_chars += doc.char_count

        return documents

    def _read_file(self, file_path: Path) -> CaseDocument:
        ext = file_path.suffix.lower()
        doc_type = _infer_doc_type(file_path.name)

        try:
            if ext == ".pdf":
                text = _read_pdf(str(file_path), gemini_api_key=self._gemini_api_key)
            elif ext in (".docx", ".doc"):
                text = _read_docx(str(file_path))
            else:
                text = _read_text_file(str(file_path))

            # Treat bracketed error strings from readers as failures, not content
            if text.startswith("[") and ("error" in text.lower() or "failed" in text.lower()):
                return CaseDocument(
                    filename=file_path.name,
                    file_path=str(file_path),
                    doc_type=doc_type,
                    text_content="",
                    char_count=0,
                    read_error=text.strip("[]"),
                )

            truncated = len(text) > MAX_CHARS_PER_DOC
            if truncated:
                text = text[:MAX_CHARS_PER_DOC] + "\n[... content truncated ...]"

            return CaseDocument(
                filename=file_path.name,
                file_path=str(file_path),
                doc_type=doc_type,
                text_content=text,
                char_count=len(text),
                truncated=truncated,
            )

        except Exception as e:
            return CaseDocument(
                filename=file_path.name,
                file_path=str(file_path),
                doc_type=doc_type,
                text_content="",
                char_count=0,
                read_error=str(e),
            )

    def get_stats(self, documents: List[CaseDocument]) -> Dict:
        """Return summary statistics about the scanned documents."""
        readable  = [d for d in documents if not d.read_error and d.char_count > 0]
        errored   = [d for d in documents if d.read_error]
        by_type: Dict[str, int] = {}
        for d in readable:
            by_type[d.doc_type] = by_type.get(d.doc_type, 0) + 1
        return {
            "total":        len(documents),
            "readable":     len(readable),
            "errors":       len(errored),
            "by_type":      by_type,
            "total_chars":  sum(d.char_count for d in readable),
            "error_files":  [d.filename for d in errored],
        }

    def build_context_block(self, documents: List[CaseDocument]) -> str:
        """
        Format all readable documents into a single context block for
        inclusion in an AI prompt.
        """
        parts = []
        for doc in documents:
            if doc.read_error or not doc.text_content:
                parts.append(
                    f"\n[DOCUMENT: {doc.filename} | Type: {doc.doc_type}]\n"
                    f"[Could not read: {doc.read_error or 'empty file'}]\n"
                )
            else:
                truncation_note = " [TRUNCATED]" if doc.truncated else ""
                parts.append(
                    f"\n{'='*60}\n"
                    f"[DOCUMENT: {doc.filename} | Type: {doc.doc_type}{truncation_note}]\n"
                    f"{'='*60}\n"
                    f"{doc.text_content}\n"
                )
        return "\n".join(parts)

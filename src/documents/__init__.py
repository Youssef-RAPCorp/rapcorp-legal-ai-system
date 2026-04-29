"""Document generation module for RAPCorp Legal AI System."""

from src.documents.document_generator import DocumentGenerator, GeneratedDocument
from src.documents.docx_writer import txt_to_docx, find_replace_docx, count_occurrences, ai_fix_docx

__all__ = [
    "DocumentGenerator", "GeneratedDocument",
    "txt_to_docx", "find_replace_docx", "count_occurrences", "ai_fix_docx",
]

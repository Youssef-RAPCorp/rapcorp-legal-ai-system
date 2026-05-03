"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    DOCX WRITER & EDITOR UTILITIES                             ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Provides:                                                                    ║
║  • txt_to_docx     — converts plain-text legal doc → formatted .docx         ║
║  • find_replace_docx — find/replace text in a .docx file                     ║
║  • count_occurrences — count matches without modifying the file               ║
║  • ai_fix_docx     — apply an AI-generated targeted edit to a .docx          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import re
from pathlib import Path
from typing import Optional

try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# TXT → DOCX CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

# Tab stop for caption bracket column: 3.5 inches from the left text margin.
# Twips = 1/1440 inch.  3.5 × 1440 = 5040.
_CAPTION_TAB_TWIPS = 5040


def _add_caption_tab_stop(para) -> None:
    """Set a left-aligned tab stop at _CAPTION_TAB_TWIPS on *para*.

    This makes every \t before ) in a caption line jump to exactly the same
    horizontal position regardless of how wide (or bold) the preceding text is.
    """
    pPr = para._p.get_or_add_pPr()
    tabs_el = OxmlElement('w:tabs')
    tab = OxmlElement('w:tab')
    tab.set(qn('w:val'), 'left')
    tab.set(qn('w:pos'), str(_CAPTION_TAB_TWIPS))
    tabs_el.append(tab)
    pPr.append(tabs_el)


def txt_to_docx(txt_path: str, docx_path: str, title: str = "") -> None:
    """
    Convert a plain-text legal document to a properly formatted .docx file.

    Formatting heuristics:
      • ALL CAPS lines              → bold centered heading
      • Lines matching /^\\d+\\.\\s/ → numbered paragraph (indented)
      • Lines starting with •/–/-   → bullet item (indented)
      • ═══/───/===/ separator lines → thin horizontal rule paragraph
      • Everything else             → normal body paragraph

    Args:
        txt_path:  Path to the source .txt file.
        docx_path: Path to write the output .docx file.
        title:     Optional document title (added as first heading if provided).
    """
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not installed. Run: pip install python-docx")

    try:
        content = Path(txt_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not read source file '{txt_path}': {exc}") from exc
    doc = Document()

    # ── Page layout ───────────────────────────────────────────────────────
    section = doc.sections[0]
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.25)
    section.right_margin  = Inches(1.25)

    # ── Default style ─────────────────────────────────────────────────────
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)

    if title:
        p = doc.add_paragraph()
        r = p.add_run(title.upper())
        r.bold = True
        r.font.size = Pt(14)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(12)

    _SEPARATOR_RE = re.compile(r'^[═─━=\-─]{4,}$')
    _NUMBERED_RE  = re.compile(r'^\d+[.)]\s')

    in_caption = True   # True until the first === separator line

    for line in content.split("\n"):
        stripped = line.strip()

        # Track end of caption region
        if in_caption and _SEPARATOR_RE.match(stripped):
            in_caption = False

        # Caption bracket line — left text + TAB + ) + optional case info
        # _fix_caption_parens emits "LEFT\t)" or "LEFT\t)    CASE INFO"
        if in_caption and '\t)' in line:
            parts = line.split('\t)', 1)
            left_text = parts[0]
            right_text = parts[1].strip() if len(parts) > 1 else ""
            p = doc.add_paragraph()
            _add_caption_tab_stop(p)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after  = Pt(2)
            # Left party text (may be bold if ALL CAPS — honour that)
            rl = p.add_run(left_text)
            rl.font.size = Pt(12)
            if left_text == left_text.upper() and left_text.strip():
                rl.bold = True
            # Tab then bracket
            p.add_run('\t')
            rb = p.add_run(')')
            rb.font.size = Pt(12)
            # Right-column case info (Case No., Judge, etc.)
            if right_text:
                rc = p.add_run('    ' + right_text)
                rc.font.size = Pt(12)
            continue

        # Separator → thin horizontal rule (em-dashes)
        if _SEPARATOR_RE.match(stripped):
            p = doc.add_paragraph()
            r = p.add_run("─" * 60)
            r.font.size = Pt(8)
            r.font.color.rgb = None   # let Word theme handle it
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after  = Pt(2)
            continue

        # Blank line → paragraph break
        if not stripped:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after  = Pt(0)
            continue

        # ALL CAPS heading (court name, section title, etc.)
        if (stripped == stripped.upper()
                and len(stripped) > 4
                and not stripped.startswith("[")
                and not _NUMBERED_RE.match(stripped)):
            p = doc.add_paragraph()
            r = p.add_run(stripped)
            r.bold = True
            r.font.size = Pt(13)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after  = Pt(6)
            continue

        # Numbered paragraph
        if _NUMBERED_RE.match(stripped):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent   = Inches(0.5)
            p.paragraph_format.space_after   = Pt(4)
            r = p.add_run(stripped)
            r.font.size = Pt(12)
            continue

        # Bullet / checklist item
        if stripped.startswith(("•", "–", "[ ]", "[x]", "[X]")) or (
            stripped.startswith("- ") and len(stripped) > 2
        ):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(stripped)
            r.font.size = Pt(12)
            continue

        # Signature / underline placeholders
        if stripped.startswith("_" * 5) or stripped.startswith("Date:"):
            p = doc.add_paragraph()
            r = p.add_run(stripped)
            r.font.size = Pt(12)
            p.paragraph_format.space_before = Pt(6)
            continue

        # Normal body paragraph
        p = doc.add_paragraph()
        r = p.add_run(stripped)
        r.font.size = Pt(12)
        p.paragraph_format.space_after = Pt(4)

    try:
        doc.save(docx_path)
    except OSError as exc:
        raise OSError(f"Could not save '{docx_path}': {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════════
# FIND / REPLACE
# ═══════════════════════════════════════════════════════════════════════════════

def _iter_paragraphs(doc: "Document"):
    """Yield every paragraph in the doc body and all table cells."""
    yield from doc.paragraphs
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def find_replace_docx(file_path: str, find_text: str, replace_text: str) -> int:
    """
    Find and replace all occurrences of find_text in a .docx file.

    Handles run-split text (Word often splits a word across multiple runs due
    to spell-check or formatting marks) by operating on the full paragraph text
    and rewriting the first run.

    Returns the number of substitutions made.
    """
    if not find_text:
        return 0

    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not installed. Run: pip install python-docx")

    if not file_path.endswith(".docx"):
        # Plain-text file — safe to do a text replacement
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            count = content.count(find_text)
            Path(file_path).write_text(
                content.replace(find_text, replace_text), encoding="utf-8"
            )
            return count
        except OSError as exc:
            raise OSError(f"Could not edit '{file_path}': {exc}") from exc

    try:
        doc = Document(file_path)
    except Exception as exc:
        raise ValueError(f"Could not open '{file_path}' as a Word document: {exc}") from exc

    total = 0
    for para in _iter_paragraphs(doc):
        if find_text not in para.text:
            continue
        total += para.text.count(find_text)
        new_text = para.text.replace(find_text, replace_text)
        for i, run in enumerate(para.runs):
            run.text = new_text if i == 0 else ""

    try:
        doc.save(file_path)
    except OSError as exc:
        raise OSError(f"Could not save '{file_path}': {exc}") from exc
    return total


def count_occurrences(file_path: str, find_text: str) -> int:
    """
    Count occurrences of find_text in a .docx or plain-text file.
    Does not modify the file.
    """
    if not find_text:
        return 0

    if DOCX_AVAILABLE and file_path.endswith(".docx"):
        doc = Document(file_path)
        return sum(para.text.count(find_text) for para in _iter_paragraphs(doc))
    else:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return content.count(find_text)


# ═══════════════════════════════════════════════════════════════════════════════
# AI-ASSISTED FIX
# ═══════════════════════════════════════════════════════════════════════════════

async def ai_fix_docx(config, file_path: str, instruction: str) -> None:
    """
    Apply an AI-generated targeted edit to a .docx file, in-place.

    Extracts the full text, sends it to Gemini Pro with the instruction,
    then rebuilds the .docx from the revised text.

    Args:
        config:      LegalAIConfig (for API key and model name).
        file_path:   Path to the .docx file to edit.
        instruction: Natural-language description of the change to make.
    """
    try:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig
    except ImportError:
        print("  google-generativeai not available — AI fix skipped.")
        return

    # Extract full text from the docx
    if DOCX_AVAILABLE and file_path.endswith(".docx"):
        doc = Document(file_path)
        content = "\n".join(p.text for p in _iter_paragraphs(doc))
    else:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")

    genai.configure(api_key=config.google_api_key)
    model = genai.GenerativeModel(config.model_pro)

    prompt = f"""You are editing a legal document. Apply ONLY the change described below.
Return the complete revised document text — no preamble, no commentary, no markdown.
Preserve all formatting structure, paragraph numbering, headings, and content
not mentioned in the change.

CHANGE REQUESTED:
{instruction}

DOCUMENT:
{content}

Write the complete revised document now:"""

    gen_config = GenerationConfig(temperature=0.1, max_output_tokens=16384)
    try:
        response = await asyncio.to_thread(
            model.generate_content, prompt, generation_config=gen_config
        )
        revised = (response.text or "").strip()
        if not revised:
            print(f"  AI returned empty output — {Path(file_path).name} unchanged.")
            return
    except Exception as exc:
        print(f"  AI fix failed for {Path(file_path).name}: {exc}")
        return

    # Rebuild the docx from the revised text
    if DOCX_AVAILABLE and file_path.endswith(".docx"):
        # Write to a temp txt, convert back to docx, clean up
        tmp = file_path.replace(".docx", "_aitmp.txt")
        Path(tmp).write_text(revised, encoding="utf-8")
        try:
            txt_to_docx(tmp, file_path)
        finally:
            Path(tmp).unlink(missing_ok=True)
    else:
        Path(file_path).write_text(revised, encoding="utf-8")

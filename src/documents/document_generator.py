"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PETITION DOCUMENT GENERATOR                                ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Generates all documents required to file a petition in a given state:       ║
║  • Petition / Complaint (main document)                                       ║
║  • Affidavit of Facts in Support                                              ║
║  • Exhibit Index                                                              ║
║  • Certificate of Service                                                     ║
║  • Filing Cover Sheet                                                         ║
║  • Filing Checklist (what to bring / submit)                                  ║
║                                                                               ║
║  All documents are drafted by gemini-pro-latest and saved as .txt files      ║
║  in an output/ directory with a timestamp.                                    ║
║                                                                               ║
║  Documents are generated for direct pro se filing.                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from configs.config import LegalAIConfig, USState
from src.evidence.evidence_analyzer import EvidenceAnalysisResult
from src.documents.document_reviewer import DocumentReviewer


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratedDocument:
    """A single generated legal document."""
    title: str           # Human-readable title
    filename: str        # File name (e.g. "01_petition.txt")
    file_path: str       # Full path to the saved file
    doc_type: str        # petition|affidavit|exhibit_index|certificate_of_service|cover_sheet|checklist
    description: str     # Brief description of what the document is
    requires_signature: bool = False
    requires_notarization: bool = False
    filing_required: bool = True  # Must be filed with the court


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentGenerator:
    """
    Generates all documents needed to file a petition in a given state,
    based on the evidence analysis and case situation.

    Output is saved to:  output/YYYY-MM-DD_HH-MM-SS/

    All documents are plaintext drafts. They must be reviewed by a licensed
    attorney before filing.
    """

    _DISCLAIMER = ""

    def __init__(self, config: LegalAIConfig):
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed.")
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        self.config = config
        genai.configure(api_key=config.google_api_key)
        self._model = genai.GenerativeModel(config.model_pro)
        self._reviewer = DocumentReviewer(config)

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _make_output_dir(self) -> str:
        """Create a timestamped output directory and return its path."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path("output") / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    def _save(self, out_dir: str, filename: str, content: str) -> str:
        """Save content to a file and return the full path."""
        path = Path(out_dir) / filename
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _evidence_summary_for_prompt(self, analysis: EvidenceAnalysisResult) -> str:
        """Format evidence items for inclusion in generation prompts."""
        lines = [f"SUMMARY: {analysis.summary}", "", "EVIDENCE ITEMS:"]
        for i, item in enumerate(
            sorted(analysis.evidence_items, key=lambda x: x.relevance_score, reverse=True), 1
        ):
            lines.append(f"\n[{i}] {item.description}")
            lines.append(f"    Type: {item.evidence_type} | Relevance: {item.relevance_score:.1f}")
            if item.location:
                lines.append(f"    Location: {item.location}")
            if item.speaker_or_source:
                lines.append(f"    Source: {item.speaker_or_source}")
            lines.append(f"    Legal significance: {item.legal_significance}")
            if item.verbatim_excerpt:
                lines.append(f"    Verbatim: \"{item.verbatim_excerpt}\"")
        if analysis.key_findings:
            lines += ["", "KEY FINDINGS:"] + [f"  • {f}" for f in analysis.key_findings]
        return "\n".join(lines)

    def _caption_block(self, plan: Dict, state: USState, doc_title: str = "") -> str:
        """
        Return a required caption instruction injected into every document
        prompt, enforcing the standard two-column bracket-style court heading.

        Layout:
          Left column  (~44 chars) — party / matter lines
          Centre column            — vertical ) characters
          Right column             — Case No., Division, Judge
        """
        court = plan.get("court_name", "[COURT NAME]").upper()
        title = (doc_title or plan.get("petition_type", "[DOCUMENT TITLE]")).upper()

        return f"""REQUIRED CAPTION FORMAT
=======================
Your document MUST open with the official two-column bracket-style court caption.
Left column: party / matter lines (pad each line to ~44 characters).
Centre:      a ) on every line.
Right column: Case No. on the third or fourth line; Division and Judge if known.
After the caption box, place the document title centred on its own line,
followed by a row of = signs, then begin the body.

Fill in all placeholders from the case facts. Leave [BRACKETS] only where
the information is genuinely unknown.

---- GUARDIANSHIP / PROBATE EXAMPLE ----

IN THE COUNTY COURT OF SARPY COUNTY, NEBRASKA

IN THE MATTER OF THE GUARDIANSHIP         )
AND CONSERVATORSHIP OF:                   )
YOUSSEF EWEIS,                            )    Case No. PR260000125
An Alleged Protected Person.              )
                                          )

        OBJECTION TO PETITION FOR APPOINTMENT OF GUARDIAN AND CONSERVATOR

============================================================

---- CIVIL (PLAINTIFF v. DEFENDANT) EXAMPLE ----

IN THE DISTRICT COURT OF DOUGLAS COUNTY, NEBRASKA

JOHN DOE,                                 )
                                          )
         Plaintiff,                       )    Case No. CI 26-1234
                                          )
    vs.                                   )    Division [  ]
                                          )
JANE SMITH,                               )    Judge: [JUDGE NAME]
                                          )
         Defendant.                       )

                        COMPLAINT FOR DAMAGES

============================================================

Now produce the actual document for this case.
Court: {court}
Document title: {title}
Use the case facts below to fill in the party / matter lines and case number.
"""

    def _chain_block_for_prompt(self, analysis: EvidenceAnalysisResult) -> str:
        """
        Return the chain-of-events narrative for inclusion in document prompts.
        If no chain was built, returns an empty string.
        """
        if analysis.chain_of_events is None:
            return ""
        return (
            "\n\nCHRONOLOGICAL CHAIN OF EVENTS\n"
            "(Use this to ensure every event appears in the document in correct order)\n"
            + analysis.chain_of_events.to_court_narrative()[:8000]
        )

    def _get_review_context(self, analysis: EvidenceAnalysisResult) -> tuple:
        """Return (chain_narrative, statements_index) for use in document review."""
        if analysis.chain_of_events is None:
            return "", "No chain of events available."
        return (
            analysis.chain_of_events.to_court_narrative(),
            analysis.chain_of_events.to_statements_index(),
        )

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """
        Remove markdown syntax from AI-generated legal document text.

        Legal documents must be plain-text; markdown characters (**, ##, *, _,
        backticks) are not valid court-filing format and appear as literal
        garbage in Word documents and PDF prints.
        """
        import re
        # ### / ## / # headings → plain text (preserve the heading text)
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        # **bold** and __bold__
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        # *italic* and _italic_
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        # `inline code`
        text = re.sub(r'`(.+?)`', r'\1', text)
        # Markdown horizontal rules (--- / *** / ___) → court separator line
        text = re.sub(r'^[-*_]{3,}\s*$', '=' * 60, text, flags=re.MULTILINE)
        # Markdown blockquotes
        text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
        # Markdown links [text](url) → text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        return text

    async def _generate_text(self, prompt: str, max_tokens: int = 8192) -> str:
        """Call Gemini Pro and return the text response. Retries up to 3 times."""
        gen_config = GenerationConfig(temperature=0.2, max_output_tokens=max_tokens)
        last_exc: Exception = RuntimeError("Unknown error")
        for attempt in range(1, 4):
            try:
                response = await asyncio.to_thread(
                    self._model.generate_content, prompt, generation_config=gen_config
                )
                text = response.text if hasattr(response, "text") else ""
                if text:
                    return self._strip_markdown(text)
                last_exc = RuntimeError("Gemini returned an empty response")
            except Exception as exc:
                last_exc = exc
                print(f"    Warning: Generation attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                await asyncio.sleep(2 ** attempt)  # 2 s, 4 s backoff
        raise RuntimeError(f"Document generation failed after 3 attempts: {last_exc}")

    # ─── Document planners ───────────────────────────────────────────────────

    async def _plan_documents(
        self,
        situation: str,
        analysis: EvidenceAnalysisResult,
        clarifications: Dict[str, str],
        state: USState,
        doc_mode: str = "petition",
    ) -> Dict[str, Any]:
        """
        Ask Gemini to determine the petition type and what documents are
        required for that state.  doc_mode is either "petition" (new action)
        or "reply" (responding to an existing proceeding).
        """
        clarification_block = (
            "\n".join(f"Q: {q}\nA: {a}" for q, a in clarifications.items())
            if clarifications else "None provided."
        )

        if doc_mode == "reply":
            mode_instruction = """DOCUMENT MODE: REPLY / RESPONSE TO EXISTING PROCEEDINGS
You are planning RESPONSE documents for an existing case — NOT initiating a new lawsuit.
Do NOT plan a new Petition or Complaint as the primary document.
Instead, choose the appropriate response/reply documents from this list:
  - "objection"            — Objection to Petition or Motion
  - "motion_to_terminate"  — Motion to Terminate / Dismiss / Vacate
  - "proposed_order"       — Proposed Order requesting specific relief
  - "reply_brief"          — Reply Brief / Supplemental Pleading / Memorandum
  - "response"             — Response or Answer to a Petition or Motion
  - "affidavit"            — Supporting Affidavit of Facts
  - "exhibit_index"        — Exhibit Index
  - "certificate_of_service" — Certificate of Service
  - "cover_sheet"          — Filing Cover Sheet
Plan only the documents that are genuinely required for this response.
Use "petition_type" to name the response action (e.g. "Objection and Motion to Terminate Guardianship")."""
            doc_type_hint = "objection|motion_to_terminate|proposed_order|reply_brief|response|affidavit|exhibit_index|certificate_of_service|cover_sheet"
        else:
            mode_instruction = """DOCUMENT MODE: NEW PETITION / LEGAL ACTION
You are planning the complete filing package to initiate a new legal action."""
            doc_type_hint = "petition|affidavit|exhibit_index|certificate_of_service|cover_sheet|proposed_order|summons|other"

        prompt = f"""You are a legal document specialist. Based on the case situation and
evidence analysis below, plan the required court documents.

{mode_instruction}

CASE SITUATION:
{situation}

PETITIONER CLARIFICATIONS:
{clarification_block}

EVIDENCE ANALYSIS SUMMARY:
{analysis.summary}

KEY FINDINGS:
{json.dumps(analysis.key_findings, indent=2)}

Before planning documents, identify:
- Any admissions by the opposing party (e.g. a motion they filed that concedes a key point)
- Any court findings or orders that already favour the filer
- The single strongest argument that should LEAD every document

These must be prominently featured — not buried in footnotes.

Return JSON in this exact format:
{{
    "petition_type": "Exact name of the action or response being filed",
    "court_name": "Full name of the court to file in",
    "court_address": "Court address if known, else null",
    "filing_fee_estimate": "Estimated filing fee range or 'Unknown'",
    "strongest_argument": "One sentence: the single most powerful argument available",
    "opposing_admissions": ["Any statement, motion, or filing by the opposing party that concedes or undermines their own position"],
    "documents": [
        {{
            "doc_type": "{doc_type_hint}",
            "title": "Full title of the document",
            "description": "What this document does and why it is required — include how it uses the strongest argument",
            "requires_signature": true,
            "requires_notarization": false,
            "filing_required": true,
            "priority": 1
        }}
    ],
    "key_statutes": [
        "Full citation for each statute that applies. Include specific section numbers — never generic references."
    ],
    "special_notes": ["Any important procedural notes for filing in {state.value}"]
}}

Order documents by priority (1 = must file first)."""

        response = await self._generate_text(prompt, max_tokens=4096)
        try:
            # Strip markdown code fences if present
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            # Fallback plan — varies by mode
            if doc_mode == "reply":
                return {
                    "petition_type": "Response / Reply",
                    "court_name": f"{state.value} District Court",
                    "court_address": None,
                    "filing_fee_estimate": "Unknown — check with clerk",
                    "documents": [
                        {"doc_type": "response", "title": "Response to Petition", "description": "Response to the opposing party's petition",
                         "requires_signature": True, "requires_notarization": False, "filing_required": True, "priority": 1},
                        {"doc_type": "affidavit", "title": "Affidavit of Facts", "description": "Sworn statement of facts",
                         "requires_signature": True, "requires_notarization": True, "filing_required": True, "priority": 2},
                        {"doc_type": "exhibit_index", "title": "Exhibit Index", "description": "List of exhibits",
                         "requires_signature": False, "requires_notarization": False, "filing_required": True, "priority": 3},
                        {"doc_type": "certificate_of_service", "title": "Certificate of Service",
                         "description": "Proof of service on respondent",
                         "requires_signature": True, "requires_notarization": False, "filing_required": True, "priority": 4},
                    ],
                    "special_notes": [],
                }
            return {
                "petition_type": "Civil Rights Petition",
                "court_name": f"{state.value} District Court",
                "court_address": None,
                "filing_fee_estimate": "Unknown — check with clerk",
                "documents": [
                    {"doc_type": "petition", "title": "Petition", "description": "Main petition document",
                     "requires_signature": True, "requires_notarization": False, "filing_required": True, "priority": 1},
                    {"doc_type": "affidavit", "title": "Affidavit of Facts", "description": "Sworn statement of facts",
                     "requires_signature": True, "requires_notarization": True, "filing_required": True, "priority": 2},
                    {"doc_type": "exhibit_index", "title": "Exhibit Index", "description": "List of exhibits",
                     "requires_signature": False, "requires_notarization": False, "filing_required": True, "priority": 3},
                    {"doc_type": "certificate_of_service", "title": "Certificate of Service",
                     "description": "Proof of service on respondent",
                     "requires_signature": True, "requires_notarization": False, "filing_required": True, "priority": 4},
                ],
                "special_notes": [],
            }

    # ─── Individual document generators ─────────────────────────────────────

    async def _gen_petition(
        self, doc_info: Dict, situation: str, analysis: EvidenceAnalysisResult,
        clarifications: Dict[str, str], plan: Dict, state: USState,
        case_law_context: str = "",
    ) -> str:
        clarification_block = (
            "\n".join(f"Q: {q}\nA: {a}" for q, a in clarifications.items())
            if clarifications else "None."
        )
        case_law_block = (
            f"\n\nRELEVANT CASE LAW (cite these where applicable in your legal claims):\n"
            f"{case_law_context[:4000]}"
        ) if case_law_context else ""

        strongest = plan.get("strongest_argument", "")
        admissions = plan.get("opposing_admissions", [])
        strength_block = ""
        if strongest or admissions:
            strength_block = "\n\nSTRONGEST ARGUMENT — LEAD WITH THIS:\n"
            if strongest:
                strength_block += f"{strongest}\n"
            if admissions:
                strength_block += "\nOPPOSING PARTY ADMISSIONS (use these prominently, not in footnotes):\n"
                strength_block += "\n".join(f"  • {a}" for a in admissions)

        key_statutes = plan.get("key_statutes", [])
        statutes_block = ""
        if key_statutes:
            statutes_block = "\n\nKEY STATUTES TO CITE (always include these with their full section numbers in the Legal Claims section — never replace with a generic reference to 'state law'):\n"
            statutes_block += "\n".join(f"  - {s}" for s in key_statutes)

        prompt = f"""{self._caption_block(plan, state, doc_info['title'])}
Draft a formal {plan['petition_type']} to be filed in the
{plan['court_name']} in {state.value}.

CASE SITUATION:
{situation}

PETITIONER CLARIFICATIONS:
{clarification_block}

EVIDENCE SUMMARY:
{self._evidence_summary_for_prompt(analysis)}
{self._chain_block_for_prompt(analysis)}{case_law_block}{strength_block}{statutes_block}

DRAFTING INSTRUCTIONS:
- Use the standard format for a {state.value} court petition
- Include: case caption, introduction, parties, jurisdiction/venue,
  statement of facts (organized chronologically), legal claims with
  applicable statutes and case law, prayer for relief
- LEAD with the strongest argument identified above — it must appear in the
  introduction and in the first substantive section, not buried at the end
- Any admissions by the opposing party must be cited as a standalone argument
  section with their own header, not folded into other paragraphs
- Number every paragraph
- Reference exhibits as Exhibit A, B, C etc. for each piece of evidence
- Be precise, factual, and professional — no hyperbole
- Leave blanks like [DATE] or [CASE NUMBER] where information is unknown
- Cite every statute from the KEY STATUTES list above by its exact section number
  in the body of the legal claims — do not substitute generic references to "state law"
- Cite relevant case law from the list above where it strengthens each claim
- Plain text only — NO markdown, NO asterisks for bold, NO pound signs for headings,
  NO hyphens as bullet markers. Use Roman numerals (I., II.) or letters (A., B.)
  for sections, and plain numbered paragraphs.

Write the complete petition document now:"""

        return await self._generate_text(prompt, max_tokens=16384)

    async def _gen_affidavit(
        self, doc_info: Dict, situation: str, analysis: EvidenceAnalysisResult,
        clarifications: Dict[str, str], plan: Dict, state: USState,
        case_law_context: str = "",
    ) -> str:
        clarification_block = (
            "\n".join(f"Q: {q}\nA: {a}" for q, a in clarifications.items())
            if clarifications else "None."
        )
        case_law_block = (
            f"\n\nRELEVANT CASE LAW (reference where it supports specific facts or legal significance):\n"
            f"{case_law_context[:3000]}"
        ) if case_law_context else ""

        prompt = f"""{self._caption_block(plan, state, doc_info['title'])}
Draft a sworn Affidavit of Facts in Support of the {plan['petition_type']}
to be filed in {plan['court_name']}, {state.value}.

CASE SITUATION:
{situation}

PETITIONER CLARIFICATIONS:
{clarification_block}

EVIDENCE ITEMS TO INCORPORATE:
{self._evidence_summary_for_prompt(analysis)}
{self._chain_block_for_prompt(analysis)}{case_law_block}

DRAFTING INSTRUCTIONS:
- Written in FIRST PERSON from the petitioner's perspective ("I, [Petitioner Name], ...")
- Include: jurat (I swear/affirm under penalty of perjury...), statement of
  personal knowledge, numbered factual paragraphs, signature block, notary block
- Each paragraph should state ONE specific fact with a specific date/time/location
- Reference supporting evidence inline (e.g. "as shown in Exhibit A at timestamp 02:14")
- Include ALL relevant facts from the evidence analysis
- Leave [BRACKETS] for unknown information (dates, case numbers, etc.)
- End with signature line, date line, and notary acknowledgment block
- Plain text only — NO markdown, NO asterisks, NO pound signs for headings.
  Use Roman numerals or letters for sections, numbered paragraphs throughout.

Write the complete affidavit now:"""

        return await self._generate_text(prompt, max_tokens=16384)

    async def _gen_exhibit_index(
        self, doc_info: Dict, situation: str, analysis: EvidenceAnalysisResult,
        plan: Dict, state: USState
    ) -> str:
        exhibits = []
        label = ord('A')
        for item in sorted(analysis.evidence_items, key=lambda x: x.relevance_score, reverse=True):
            exhibits.append({
                "label": chr(label),
                "description": item.description,
                "type": item.evidence_type,
                "location": item.location or "N/A",
                "source_file": item.speaker_or_source or "N/A",
                "relevance": f"{item.relevance_score:.1f}",
            })
            label += 1
            if label > ord('Z'):
                break

        # Also list the source files
        source_files = analysis.files_analyzed

        content = [
            f"IN THE {plan['court_name'].upper()}",
            f"STATE OF {state.value}",
            "",
            f"Case No.: [TO BE ASSIGNED]",
            "",
            f"EXHIBIT INDEX — {plan['petition_type'].upper()}",
            "",
            f"Petitioner: [PETITIONER NAME]",
            f"Respondent: [RESPONDENT NAME]",
            "",
            "=" * 60,
            "",
            "SOURCE FILES SUBMITTED:",
        ]
        for i, sf in enumerate(source_files, 1):
            content.append(f"  {i}. {sf}")

        content += ["", "=" * 60, "", "EXHIBIT LIST:", ""]
        for ex in exhibits:
            content += [
                f"Exhibit {ex['label']}:",
                f"  Description:  {ex['description']}",
                f"  Type:         {ex['type']}",
                f"  Location:     {ex['location']}",
                f"  Source:       {ex['source_file']}",
                f"  Relevance:    {ex['relevance']}",
                "",
            ]

        content += [
            "=" * 60,
            "",
            "Respectfully submitted,",
            "",
            "_______________________________",
            "[Petitioner Name]",
            "Date: _______________",
        ]

        return "\n".join(content)

    async def _gen_certificate_of_service(
        self, doc_info: Dict, plan: Dict, state: USState
    ) -> str:
        return f"""IN THE {plan['court_name'].upper()}
STATE OF {state.value}

Case No.: [TO BE ASSIGNED]

CERTIFICATE OF SERVICE

I, [PETITIONER NAME], hereby certify that on [DATE], I served a true and
correct copy of the {plan['petition_type']}, Affidavit of Facts, and all
exhibits upon the following parties by the method indicated:

Respondent:
  Name:    [RESPONDENT FULL NAME]
  Address: [RESPONDENT ADDRESS]
  Method:  [ ] Personal Service  [ ] Certified Mail  [ ] Electronic Service

If served by mail, the document was deposited in the United States Mail,
postage prepaid, addressed as stated above.

If served electronically, the document was transmitted to the email address
on file with the court.

I declare under penalty of perjury under the laws of the State of {state.value}
that the foregoing is true and correct.

Executed on: _______________

_______________________________
[Petitioner Name]
[Address]
[City, State, ZIP]
[Phone]
[Email]
{self._DISCLAIMER}"""

    async def _gen_proposed_order(
        self, doc_info: Dict, situation: str, plan: Dict, state: USState,
        case_law_context: str = "",
    ) -> str:
        case_law_block = (
            f"\n\nRELEVANT STATUTES AND CASE LAW (cite in the order where applicable):\n"
            f"{case_law_context[:2000]}"
        ) if case_law_context else ""

        key_statutes = plan.get("key_statutes", [])
        statutes_block = ""
        if key_statutes:
            statutes_block = "\n\nKEY STATUTES (cite these by full section number in the ORDER items where applicable):\n"
            statutes_block += "\n".join(f"  - {s}" for s in key_statutes)

        prompt = f"""{self._caption_block(plan, state, doc_info['title'])}
Draft a PROPOSED ORDER for the petitioner/respondent to FILE WITH THE COURT
in the matter of {plan['petition_type']} in {plan['court_name']}, {state.value}.

IMPORTANT — WHAT A PROPOSED ORDER IS:
A proposed order is a document drafted by a PARTY (not the judge) and submitted
to the court asking the judge to sign it. It states what RELIEF the party is
requesting. It is NOT a decision — the judge has not ruled yet.

CASE SITUATION:
{situation}{case_law_block}{statutes_block}

REQUIRED FORMAT — follow this structure exactly:

1. CASE CAPTION (court name, case number, parties)
2. TITLE: "PROPOSED ORDER [describing relief sought]"
3. ONE short recitation paragraph:
   "THIS MATTER having come before the Court upon [document name] filed by
   [party], and the Court having considered the pleadings and record, and
   being duly advised in the premises:"
4. "IT IS HEREBY ORDERED:" followed by numbered relief items — each item is
   a specific thing you are asking the judge to order (dismiss the petition,
   terminate the conservatorship, restore legal capacity, etc.)
5. Signature block for the JUDGE:
   "SO ORDERED this ___ day of __________, 20__."
   "_________________________________"
   "Judge, [Court Name]"
   "[JUDGE NAME]"

RULES:
- Do NOT write findings of fact — that is the judge's job, not the filer's
- Do NOT write conclusions of law — those belong in the brief/objection
- Do NOT write in the judge's voice making legal determinations
- Keep it short: caption + one-paragraph recitation + numbered ORDER items + signature
- Leave [BRACKETS] for case number, judge name, date, and any unknown information
- Plain text only — no markdown, no asterisks, no pound signs

Write the complete proposed order now:"""

        return await self._generate_text(prompt, max_tokens=2048)

    async def _gen_reply_document(
        self, doc_info: Dict, situation: str, analysis: EvidenceAnalysisResult,
        clarifications: Dict[str, str], plan: Dict, state: USState,
        case_law_context: str = "",
    ) -> str:
        clarification_block = (
            "\n".join(f"Q: {q}\nA: {a}" for q, a in clarifications.items())
            if clarifications else "None."
        )
        case_law_block = (
            f"\n\nRELEVANT CASE LAW (cite where applicable):\n{case_law_context[:3000]}"
        ) if case_law_context else ""

        key_statutes = plan.get("key_statutes", [])
        statutes_block = ""
        if key_statutes:
            statutes_block = "\n\nKEY STATUTES TO CITE (with full section numbers):\n"
            statutes_block += "\n".join(f"  - {s}" for s in key_statutes)

        strongest = plan.get("strongest_argument", "")
        admissions = plan.get("opposing_admissions", [])
        strength_block = ""
        if strongest or admissions:
            strength_block = "\n\nSTRONGEST ARGUMENT — LEAD WITH THIS:\n"
            if strongest:
                strength_block += f"{strongest}\n"
            if admissions:
                strength_block += "\nOPPOSING PARTY ADMISSIONS (cite prominently):\n"
                strength_block += "\n".join(f"  • {a}" for a in admissions)

        prompt = f"""{self._caption_block(plan, state, doc_info['title'])}
You are a senior litigation attorney drafting a reply/response document
for pro se filing in the {plan['court_name']}, {state.value}.

DOCUMENT TO DRAFT: {doc_info['title']}
PURPOSE: {doc_info.get('description', doc_info['title'])}
ACTION: {plan['petition_type']}

CASE SITUATION:
{situation}

CLARIFICATIONS:
{clarification_block}

EVIDENCE SUMMARY:
{self._evidence_summary_for_prompt(analysis)}
{self._chain_block_for_prompt(analysis)}{case_law_block}{statutes_block}{strength_block}

DRAFTING INSTRUCTIONS:
- Written in the voice of the RESPONDING PARTY (not initiating a new lawsuit)
- Include: case caption, introduction referencing the document being responded to,
  numbered argument sections, prayer for relief, signature block
- LEAD with the strongest argument and any admissions by the opposing party
- Cite every applicable statute by its full section number
- Reference exhibits by letter (Exhibit A, B, C…)
- Number every paragraph
- Plain text only — NO markdown, NO asterisks, NO pound signs for headings.
  Use Roman numerals (I., II.) or letters (A., B.) for sections.
- Leave [BRACKETS] for unknown information (dates, case numbers, etc.)

Write the complete document now:"""

        return await self._generate_text(prompt, max_tokens=16384)

    def _gen_cover_sheet(self, plan: Dict, situation: str, state: USState) -> str:
        return f"""CIVIL CASE COVER SHEET — {state.value}
{plan['court_name'].upper()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CASE INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Case Type:        {plan['petition_type']}
Court:            {plan['court_name']}
Filing Date:      [DATE OF FILING]
Case Number:      [ASSIGNED BY CLERK]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PETITIONER
  Full Name:      [PETITIONER FULL LEGAL NAME]
  Address:        [ADDRESS]
  City/State/ZIP: [CITY, STATE ZIP]
  Phone:          [PHONE NUMBER]
  Email:          [EMAIL ADDRESS]
  Attorney:       [ ] Self-Represented (Pro Se)  [ ] Represented by: ___________

RESPONDENT
  Full Name:      [RESPONDENT FULL LEGAL NAME]
  Address:        [ADDRESS]
  City/State/ZIP: [CITY, STATE ZIP]
  Employer/Title: [RESPONDENT'S EMPLOYER AND JOB TITLE]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENTS FILED WITH THIS COVER SHEET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] Petition / Complaint
[ ] Affidavit of Facts in Support
[ ] Exhibit Index
[ ] Exhibits (A through __)
[ ] Proposed Order
[ ] Certificate of Service
[ ] Filing Fee (Est. {plan.get('filing_fee_estimate', 'See clerk')})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Signature: _______________________________ Date: _______________
           [Petitioner Name or Attorney]
{self._DISCLAIMER}"""

    def _gen_filing_checklist(
        self, plan: Dict, generated_docs: List[GeneratedDocument], state: USState
    ) -> str:
        lines = [
            f"FILING CHECKLIST — {plan['petition_type'].upper()}",
            f"Court: {plan['court_name']} | State: {state.value}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "=" * 60,
            "STEP 1 — COMPLETE THESE ITEMS BEFORE GOING TO THE COURTHOUSE",
            "=" * 60,
            "",
            "[ ] Fill in ALL [BRACKET] placeholders in every document",
            "[ ] Have an attorney review ALL documents before filing",
            "[ ] Sign all documents that require a signature",
            "[ ] Have affidavit(s) notarized by a licensed notary public",
            "[ ] Make physical copies of ALL exhibits (label each one clearly)",
            "[ ] Print at least 3 copies of every document",
            "    (1 for court, 1 for respondent, 1 for your records)",
            "",
            "=" * 60,
            "STEP 2 — DOCUMENTS TO BRING TO THE COURTHOUSE",
            "=" * 60,
            "",
        ]

        for i, doc in enumerate(generated_docs, 1):
            sig_note = " ← MUST BE SIGNED" if doc.requires_signature else ""
            notary_note = " ← MUST BE NOTARIZED" if doc.requires_notarization else ""
            filing_note = "" if doc.filing_required else " (keep for your records)"
            lines.append(f"[ ] {i:02d}. {doc.title}{sig_note}{notary_note}{filing_note}")
            lines.append(f"      File: {Path(doc.file_path).name}")
            lines.append(f"      {doc.description}")
            lines.append("")

        lines += [
            "=" * 60,
            "STEP 3 — AT THE COURTHOUSE",
            "=" * 60,
            "",
            f"[ ] Go to the Clerk of Courts office at {plan['court_name']}",
            f"    Address: {plan.get('court_address') or 'Look up on the court website'}",
            "[ ] Bring a valid government-issued photo ID",
            f"[ ] Bring filing fee: {plan.get('filing_fee_estimate', 'Ask the clerk')}",
            "    (Clerks typically accept cash, money order, or credit card)",
            "[ ] Ask the clerk to file-stamp all copies",
            "[ ] Keep your file-stamped copies — they are your proof of filing",
            "[ ] Ask the clerk about the service requirements for the respondent",
            "",
            "=" * 60,
            "STEP 4 — AFTER FILING",
            "=" * 60,
            "",
            "[ ] Serve the respondent per the court's service requirements",
            "[ ] File the Certificate of Service with the court after serving",
            "[ ] Note your hearing date (the clerk will advise or mail it to you)",
            "[ ] Keep ALL documents, receipts, and correspondence in a safe place",
            "",
        ]

        if plan.get("special_notes"):
            lines += ["=" * 60, "SPECIAL NOTES FOR " + state.value, "=" * 60, ""]
            for note in plan["special_notes"]:
                lines.append(f"  • {note}")
            lines.append("")

        lines.append(self._DISCLAIMER)
        return "\n".join(lines)

    # ─── Public API ──────────────────────────────────────────────────────────

    async def generate_petition_package(
        self,
        situation: str,
        analysis: EvidenceAnalysisResult,
        clarifications: Dict[str, str],
        state: USState,
        output_dir: Optional[str] = None,
        case_law_context: str = "",
        doc_mode: str = "petition",
    ) -> List[GeneratedDocument]:
        """
        Generate the complete package of documents needed to file the petition.

        Args:
            situation: Full case situation description.
            analysis: The final (clarified) evidence analysis result.
            clarifications: Dict of {question: answer} from the interactive session.
            state: The jurisdiction.
            output_dir: Override output directory (default: output/TIMESTAMP/).

        Returns:
            List of GeneratedDocument objects, one per file produced.
        """
        out_dir = output_dir or self._make_output_dir()
        print(f"\n  Generating petition documents in: {out_dir}/")

        # Determine what documents are needed
        print("    Planning required documents...")
        plan = await self._plan_documents(situation, analysis, clarifications, state, doc_mode)
        print(f"    Petition type: {plan['petition_type']}")
        print(f"    Court: {plan['court_name']}")
        print(f"    Documents to generate: {len(plan['documents'])}")

        generated: List[GeneratedDocument] = []
        idx = 1

        # Pre-compute review context from chain of events (used for all docs)
        chain_narrative, statements_index = self._get_review_context(analysis)
        evidence_summary = self._evidence_summary_for_prompt(analysis)

        # Documents that are AI-generated and benefit from iterative review
        _AI_REVIEWED_TYPES = {
            "petition", "affidavit", "proposed_order",
            "objection", "motion_to_terminate", "reply_brief", "response", "motion",
        }

        for doc_info in sorted(plan["documents"], key=lambda d: d.get("priority", 99)):
            doc_type = doc_info["doc_type"]
            title = doc_info["title"]
            filename = f"{idx:02d}_{doc_type}.txt"
            print(f"    Drafting: {title}...")

            content = None

            if doc_type == "petition" and doc_mode != "reply":
                content = await self._gen_petition(doc_info, situation, analysis, clarifications, plan, state, case_law_context)
            elif doc_type == "petition" and doc_mode == "reply":
                content = await self._gen_reply_document(doc_info, situation, analysis, clarifications, plan, state, case_law_context)
            elif doc_type == "affidavit":
                content = await self._gen_affidavit(doc_info, situation, analysis, clarifications, plan, state, case_law_context)
            elif doc_type == "exhibit_index":
                content = await self._gen_exhibit_index(doc_info, situation, analysis, plan, state)
            elif doc_type == "certificate_of_service":
                content = await self._gen_certificate_of_service(doc_info, plan, state)
            elif doc_type == "proposed_order":
                content = await self._gen_proposed_order(doc_info, situation, plan, state, case_law_context)
            elif doc_type == "cover_sheet":
                content = self._gen_cover_sheet(plan, situation, state)
            elif doc_type in ("objection", "motion_to_terminate", "reply_brief",
                              "response", "motion", "supplemental_pleading", "notice"):
                content = await self._gen_reply_document(doc_info, situation, analysis, clarifications, plan, state, case_law_context)
            else:
                content = await self._gen_reply_document(doc_info, situation, analysis, clarifications, plan, state, case_law_context)

            if content:
                # Iterative review for AI-generated substantive documents
                if doc_type in _AI_REVIEWED_TYPES and chain_narrative:
                    print(f"    Reviewing: {title}...")
                    content, review_history = await self._reviewer.review_and_refine(
                        doc_title=title,
                        doc_content=content,
                        chain_narrative=chain_narrative,
                        statements_index=statements_index,
                        evidence_summary=evidence_summary,
                        situation=situation,
                        max_iterations=10,
                        verbose=True,
                    )
                    # Save review log alongside the document
                    review_log = "\n\n".join(r.format_for_log() for r in review_history)
                    review_log_filename = f"{idx:02d}_{doc_type}_REVIEW_LOG.txt"
                    self._save(out_dir, review_log_filename, review_log)

                content += self._DISCLAIMER
                file_path = self._save(out_dir, filename, content)
                generated.append(GeneratedDocument(
                    title=title,
                    filename=filename,
                    file_path=file_path,
                    doc_type=doc_type,
                    description=doc_info.get("description", ""),
                    requires_signature=doc_info.get("requires_signature", False),
                    requires_notarization=doc_info.get("requires_notarization", False),
                    filing_required=doc_info.get("filing_required", True),
                ))
                idx += 1

        # Always generate cover sheet if not already included
        if not any(d.doc_type == "cover_sheet" for d in generated):
            content = self._gen_cover_sheet(plan, situation, state) + self._DISCLAIMER
            filename = f"{idx:02d}_cover_sheet.txt"
            file_path = self._save(out_dir, filename, content)
            generated.append(GeneratedDocument(
                title="Filing Cover Sheet",
                filename=filename,
                file_path=file_path,
                doc_type="cover_sheet",
                description="Cover sheet to attach to the filing package",
                requires_signature=True,
                filing_required=True,
            ))
            idx += 1

        # Always generate filing checklist last
        checklist_content = self._gen_filing_checklist(plan, generated, state)
        checklist_filename = f"{idx:02d}_FILING_CHECKLIST.txt"
        checklist_path = self._save(out_dir, checklist_filename, checklist_content)
        generated.append(GeneratedDocument(
            title="Filing Checklist",
            filename=checklist_filename,
            file_path=checklist_path,
            doc_type="checklist",
            description="Step-by-step instructions for filing the petition package",
            requires_signature=False,
            filing_required=False,
        ))

        # Save the plan summary
        plan_path = self._save(out_dir, "00_document_plan.json", json.dumps(plan, indent=2))

        print(f"    Done — {len(generated)} documents saved to {out_dir}/")
        return generated

    async def generate_continuation_document(
        self,
        document_type: str,
        case_summary: str,
        case_documents_context: str,
        recommendations: List[str],
        state: USState,
        additional_instructions: str = "",
        output_dir: Optional[str] = None,
    ) -> GeneratedDocument:
        """
        Draft a single follow-up/continuation document based on the existing
        case file analysis.

        Args:
            document_type:   What to draft — "reply", "response", "motion",
                             "opposition", "appeal", "supplement", "amended_petition"
            case_summary:    AI-generated overview of the case.
            case_documents_context: Full text of existing case documents.
            recommendations: AI recommendations from case analysis.
            state:           Jurisdiction.
            additional_instructions: Any specific user instructions for this doc.
            output_dir:      Override output dir (default: output/TIMESTAMP/).

        Returns:
            GeneratedDocument pointing to the saved file.
        """
        out_dir = output_dir or self._make_output_dir()

        type_guidance = {
            "reply":             "a Reply to the opposing party's Response/Answer",
            "response":          "a Response/Answer to the opposing party's Complaint or Petition",
            "motion":            "a Motion (specify the relief sought based on the case context)",
            "opposition":        "an Opposition to the opposing party's Motion",
            "appeal":            "a Notice of Appeal and supporting brief",
            "supplement":        "a Supplemental Filing providing additional facts or evidence",
            "amended_petition":  "an Amended Petition incorporating the new developments",
            "motion_to_compel":  "a Motion to Compel compliance with discovery or court orders",
            "motion_to_dismiss": "a Motion to Dismiss based on the case context",
        }.get(document_type, f"a {document_type.replace('_', ' ').title()}")

        recs_block = "\n".join(f"  - {r}" for r in recommendations) if recommendations else "  None."
        user_instructions = f"\n\nADDITIONAL INSTRUCTIONS FROM USER:\n{additional_instructions}" if additional_instructions else ""

        prompt = f"""You are a senior litigation attorney drafting a legal document for pro se filing.

JURISDICTION: {state.value}
DOCUMENT TO DRAFT: {type_guidance}

CASE SUMMARY:
{case_summary}

AI RECOMMENDATIONS LEADING TO THIS DOCUMENT:
{recs_block}
{user_instructions}

EXISTING CASE DOCUMENTS (for reference and continuity):
{case_documents_context[:60000]}

DRAFTING INSTRUCTIONS:
- Draft {type_guidance} that directly addresses the current case posture
- Use the standard format for {state.value} court filings
- Include: case caption with [CASE NUMBER] placeholder, date, parties
- Reference specific documents, dates, and facts from the existing case file
- Number all paragraphs
- Use precise, professional legal language appropriate for pro se filing
- Leave [BRACKETS] for any unknown information (names, dates, case numbers)
- End with signature block, date line, and certificate of service if applicable
- Do NOT include preamble or meta-commentary — write the document directly

Write the complete {document_type.replace('_', ' ')} now:"""

        print(f"    Drafting {document_type.replace('_', ' ')}...")
        content = await self._generate_text(prompt, max_tokens=16384)

        # Iterative review — cross-reference against the full case file
        print(f"    Reviewing {document_type.replace('_', ' ')} against case documents...")
        content, review_history = await self._reviewer.review_and_refine(
            doc_title=type_guidance.replace("a ", "").replace("an ", "").title(),
            doc_content=content,
            chain_narrative=case_summary,
            statements_index="",
            evidence_summary="",
            situation=case_summary[:1000],
            max_iterations=10,
            verbose=True,
            case_documents_context=case_documents_context,
        )
        review_log = "\n\n".join(r.format_for_log() for r in review_history)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        review_log_filename = f"{document_type}_{timestamp}_REVIEW_LOG.txt"
        self._save(out_dir, review_log_filename, review_log)

        filename = f"{document_type}_{timestamp}.txt"
        file_path = self._save(out_dir, filename, content + self._DISCLAIMER)

        return GeneratedDocument(
            title=type_guidance.replace("a ", "").replace("an ", "").title(),
            filename=filename,
            file_path=file_path,
            doc_type=document_type,
            description=f"Continuation document: {type_guidance}",
            requires_signature=True,
            requires_notarization=document_type in ("affidavit",),
            filing_required=True,
        )

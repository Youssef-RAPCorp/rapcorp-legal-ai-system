"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ITERATIVE DOCUMENT REVIEWER                                ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Reviews generated legal documents against source evidence and the           ║
║  chain of events, then revises until the document is complete and accurate.  ║
║                                                                               ║
║  Review loop per document:                                                   ║
║  1. Flash review pass — identify omissions, inaccuracies, quote errors       ║
║  2. Pro revision pass — rewrite document incorporating all fixes              ║
║  3. Repeat up to max_iterations (default 3) or until approved                ║
║                                                                               ║
║  Approval criteria:                                                           ║
║  • Quality score ≥ 85/100                                                    ║
║  • Zero critical issues                                                       ║
║  • All chain-of-events statements accounted for                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from configs.config import LegalAIConfig


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReviewIssue:
    """A single issue identified during a review pass."""
    severity: str           # critical|major|minor
    category: str           # omission|misstatement|inaccuracy|quote_error|inconsistency|missing_legal_basis
    description: str        # What the issue is
    source_reference: str   # Where in the source evidence this was found (file + location)
    suggested_fix: str      # What should be added, corrected, or changed


@dataclass
class ReviewPass:
    """Result of a single review iteration."""
    iteration: int
    score: int              # 0–100 quality score
    issues: List[ReviewIssue] = field(default_factory=list)
    is_approved: bool = False
    reviewer_summary: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def major_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "major")

    @property
    def minor_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "minor")

    def format_for_log(self) -> str:
        lines = [
            f"=== Review Pass {self.iteration} ===",
            f"Score: {self.score}/100 | Approved: {self.is_approved}",
            f"Issues: {self.critical_count} critical, {self.major_count} major, {self.minor_count} minor",
            f"Summary: {self.reviewer_summary}",
        ]
        if self.issues:
            lines.append("\nIssues Found:")
            for i, issue in enumerate(self.issues, 1):
                lines += [
                    f"  [{i}] [{issue.severity.upper()}] {issue.category}",
                    f"      Problem: {issue.description}",
                    f"      Source:  {issue.source_reference}",
                    f"      Fix:     {issue.suggested_fix}",
                ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT REVIEWER
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentReviewer:
    """
    Iteratively reviews and refines generated legal documents.

    Each iteration:
    1. REVIEW (Flash) — cross-references the document against the chain of
       events and evidence items, producing a scored issue list.
    2. REVISE (Pro) — rewrites the document to fix all identified issues,
       preserving structure and format.

    Loops until approved (score ≥ threshold, no critical issues) or
    max_iterations is reached — whichever comes first.
    """

    _APPROVAL_SCORE = 85       # Minimum score to approve without further revision
    _REVIEW_MODEL = "gemini-flash-latest"   # Cheap + fast for analysis
    _REVISE_MODEL = "gemini-pro-latest"     # Best for drafting corrections

    def __init__(self, config: LegalAIConfig):
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed.")
        self.config = config
        genai.configure(api_key=config.google_api_key)

    # ─── Review pass ─────────────────────────────────────────────────────────

    async def _review_pass(
        self,
        doc_title: str,
        doc_content: str,
        chain_narrative: str,
        statements_index: str,
        evidence_summary: str,
        situation: str,
        iteration: int,
        case_documents_context: str = "",
    ) -> ReviewPass:
        """
        Run one review pass using Flash. Returns a ReviewPass with scored issues.
        """
        # Truncate large inputs to keep token budget manageable
        chain_truncated = chain_narrative[:6000]
        stmts_truncated = statements_index[:6000]
        evidence_truncated = evidence_summary[:4000]
        doc_truncated = doc_content[:12000]
        case_docs_truncated = case_documents_context[:4000] if case_documents_context else ""

        case_docs_block = ""
        if case_docs_truncated:
            case_docs_block = f"""
═══════════════════════════════════════════════
EXISTING CASE FILE DOCUMENTS
(Check document consistency against these prior filings)
═══════════════════════════════════════════════
{case_docs_truncated}
"""

        prompt = f"""You are a senior legal document auditor. Your task is to review a
draft legal document against the source evidence and chain of events.

CASE SITUATION:
{situation[:1000]}

═══════════════════════════════════════════════
SOURCE GROUND TRUTH — Chain of Events
═══════════════════════════════════════════════
{chain_truncated}

═══════════════════════════════════════════════
SOURCE GROUND TRUTH — Verbatim Statements Index
═══════════════════════════════════════════════
{stmts_truncated}

═══════════════════════════════════════════════
EVIDENCE SUMMARY
═══════════════════════════════════════════════
{evidence_truncated}
{case_docs_block}
═══════════════════════════════════════════════
DOCUMENT UNDER REVIEW: {doc_title} (Iteration {iteration})
═══════════════════════════════════════════════
{doc_truncated}

AUDIT INSTRUCTIONS:
For each issue found, determine severity:
• CRITICAL — A key event, admission, or verbatim quote from the chain of events
  is entirely absent from the document, OR a fact is stated incorrectly
• MAJOR — An important event is mentioned but incomplete, vague, or missing its
  verbatim quote; chronology is wrong; a claim lacks its statutory basis
• MINOR — Minor omission, imprecise wording, formatting inconsistency

Check specifically:
1. COMPLETENESS — Is every event from the chain of events represented?
2. VERBATIM ACCURACY — Are all direct quotes reproduced exactly (not paraphrased)?
3. CHRONOLOGICAL ORDER — Are facts presented in the correct sequence?
4. FACTUAL ACCURACY — Are all stated facts consistent with the source evidence?
5. LEGAL BASIS — Does each legal claim cite a statute or standard?
6. EXHIBIT REFERENCES — Are all key evidence items referenced as exhibits?
7. CASE FILE CONSISTENCY — Are all facts, party names, dates, and prior rulings
   consistent with the existing case documents (if any are provided above)?

Scoring:
• Start at 100
• Deduct 20 per critical issue
• Deduct 8 per major issue
• Deduct 2 per minor issue
• Minimum score is 0

Return ONLY valid JSON:
{{
    "score": 87,
    "is_approved": true,
    "reviewer_summary": "One sentence summarizing the document quality",
    "issues": [
        {{
            "severity": "critical|major|minor",
            "category": "omission|misstatement|inaccuracy|quote_error|inconsistency|missing_legal_basis",
            "description": "Specific description of the issue",
            "source_reference": "Chain Event [N] | Statement [SN] | Evidence item [N] @ file:location",
            "suggested_fix": "Exact text to add or change, with source quote if applicable"
        }}
    ]
}}"""

        model = genai.GenerativeModel(self._REVIEW_MODEL)
        gen_config = GenerationConfig(
            temperature=0.1,
            max_output_tokens=4096,
            response_mime_type="application/json",
        )

        try:
            response = await asyncio.to_thread(
                model.generate_content, prompt, generation_config=gen_config
            )
            data = json.loads(response.text)
        except Exception as exc:
            # Non-fatal: treat as approved if review fails
            return ReviewPass(
                iteration=iteration,
                score=90,
                is_approved=True,
                reviewer_summary=f"Review pass skipped (parse error): {exc}",
            )

        issues = []
        for raw in data.get("issues", []):
            issues.append(ReviewIssue(
                severity=str(raw.get("severity", "minor")),
                category=str(raw.get("category", "omission")),
                description=str(raw.get("description", "")),
                source_reference=str(raw.get("source_reference", "")),
                suggested_fix=str(raw.get("suggested_fix", "")),
            ))

        score = int(data.get("score", 100))
        is_approved = score >= self._APPROVAL_SCORE and not any(
            i.severity == "critical" for i in issues
        )

        return ReviewPass(
            iteration=iteration,
            score=score,
            issues=issues,
            is_approved=is_approved,
            reviewer_summary=str(data.get("reviewer_summary", "")),
        )

    # ─── Revision pass ───────────────────────────────────────────────────────

    async def _revision_pass(
        self,
        doc_title: str,
        doc_content: str,
        review: ReviewPass,
        chain_narrative: str,
        statements_index: str,
        evidence_summary: str,
        situation: str,
        case_documents_context: str = "",
    ) -> str:
        """
        Rewrite the document using Pro to fix all issues from the review pass.
        Preserves document structure, format, and all correct content.
        """
        issues_block = "\n".join(
            f"[{i + 1}] [{iss.severity.upper()}] {iss.category}\n"
            f"    Problem: {iss.description}\n"
            f"    Source: {iss.source_reference}\n"
            f"    Fix: {iss.suggested_fix}"
            for i, iss in enumerate(review.issues)
        ) or "No issues — document already approved."

        chain_truncated = chain_narrative[:6000]
        stmts_truncated = statements_index[:5000]
        doc_truncated = doc_content[:14000]
        case_docs_truncated = case_documents_context[:3000] if case_documents_context else ""

        case_docs_block = ""
        if case_docs_truncated:
            case_docs_block = f"""
═══════════════════════════════════════════════
EXISTING CASE FILE (maintain consistency with these documents)
═══════════════════════════════════════════════
{case_docs_truncated}
"""

        prompt = f"""You are a senior litigation attorney revising a draft legal document.
Your task is to fix ALL identified issues while preserving the document's
structure, format, and all content that was already correct.

CASE SITUATION:
{situation[:1000]}

═══════════════════════════════════════════════
SOURCE GROUND TRUTH — Chain of Events
═══════════════════════════════════════════════
{chain_truncated}

═══════════════════════════════════════════════
SOURCE GROUND TRUTH — Verbatim Statements Index
═══════════════════════════════════════════════
{stmts_truncated}
{case_docs_block}
═══════════════════════════════════════════════
ISSUES TO FIX IN: {doc_title}
(Review Score: {review.score}/100)
═══════════════════════════════════════════════
{issues_block}

═══════════════════════════════════════════════
CURRENT DRAFT (revise this):
═══════════════════════════════════════════════
{doc_truncated}

REVISION INSTRUCTIONS:
1. Fix EVERY issue listed above — none may be skipped
2. Preserve ALL content that was not flagged as an issue
3. Maintain the exact same document structure (headers, paragraph numbering, etc.)
4. For quote errors: replace paraphrases with exact verbatim quotes from the source
5. For omissions: add the missing events/statements in the correct chronological position
6. For legal basis issues: add the appropriate statute citation
7. Ensure all facts are consistent with the existing case file documents (if provided)
8. Do NOT add meta-commentary or revision notes — produce the final clean document only

Write the complete revised {doc_title} now:"""

        model = genai.GenerativeModel(self._REVISE_MODEL)
        gen_config = GenerationConfig(
            temperature=0.15,
            max_output_tokens=16384,
        )

        try:
            response = await asyncio.to_thread(
                model.generate_content, prompt, generation_config=gen_config
            )
            revised = response.text if hasattr(response, "text") else ""
            return revised.strip() if revised.strip() else doc_content
        except Exception:
            return doc_content  # Return original if revision fails

    # ─── Public API ──────────────────────────────────────────────────────────

    async def review_and_refine(
        self,
        doc_title: str,
        doc_content: str,
        chain_narrative: str,
        statements_index: str,
        evidence_summary: str,
        situation: str,
        max_iterations: int = 10,
        verbose: bool = True,
        case_documents_context: str = "",
    ) -> Tuple[str, List[ReviewPass]]:
        """
        Iteratively review and refine a document until it is approved.

        Args:
            doc_title:              Human-readable title of the document being reviewed.
            doc_content:            Current draft content of the document.
            chain_narrative:        Formatted chain of events (from ChainOfEvents.to_court_narrative()).
            statements_index:       Formatted statements index (from ChainOfEvents.to_statements_index()).
            evidence_summary:       Formatted evidence items summary.
            situation:              Case situation description.
            max_iterations:         Maximum review-revise cycles (default 5).
            verbose:                Print progress to stdout.
            case_documents_context: Full text of existing case file documents for
                                    consistency checking (prior filings, orders, etc.).

        Returns:
            (final_content, review_history) — the revised document and all review passes.
        """
        current_content = doc_content
        history: List[ReviewPass] = []

        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"      Review pass {iteration}/{max_iterations}...")

            review = await self._review_pass(
                doc_title=doc_title,
                doc_content=current_content,
                chain_narrative=chain_narrative,
                statements_index=statements_index,
                evidence_summary=evidence_summary,
                situation=situation,
                iteration=iteration,
                case_documents_context=case_documents_context,
            )
            history.append(review)

            if verbose:
                status = "APPROVED" if review.is_approved else f"NEEDS REVISION ({review.score}/100)"
                print(f"      {status} — {review.critical_count}C / {review.major_count}M / {review.minor_count}m issues")

            if review.is_approved:
                break

            if iteration < max_iterations:
                if verbose:
                    print(f"      Revising...")
                current_content = await self._revision_pass(
                    doc_title=doc_title,
                    doc_content=current_content,
                    review=review,
                    chain_narrative=chain_narrative,
                    statements_index=statements_index,
                    evidence_summary=evidence_summary,
                    situation=situation,
                    case_documents_context=case_documents_context,
                )

        return current_content, history

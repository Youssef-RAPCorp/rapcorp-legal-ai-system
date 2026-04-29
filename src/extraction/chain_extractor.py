"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    CHAIN-OF-EVENTS EXTRACTOR                                  ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Performs structured chronological extraction from already-analyzed evidence.  ║
║  Runs AFTER the initial evidence analysis (text-only — no file re-uploads).   ║
║                                                                               ║
║  Produces:                                                                    ║
║  • Complete chronological chain of events with source anchors                 ║
║  • Index of every verbatim statement by speaker and timestamp                 ║
║  • Key actor roster and date map                                              ║
║                                                                               ║
║  This ground-truth record is fed into document generation prompts and        ║
║  review passes so that no subtle statement or event is ever missed.           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from configs.config import LegalAIConfig, USState


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerbatimStatement:
    """A single direct quote extracted from the source documents."""
    speaker: str                    # Who said/wrote it (or "Unknown")
    statement: str                  # Exact words — never paraphrased
    timestamp_or_location: str      # MM:SS, page X, section Y, etc.
    source_file: str                # Filename
    context: str                    # Brief surrounding context (1 sentence)
    statement_type: str             # spoken|written|on_screen|typed


@dataclass
class ChainEvent:
    """A single event in the chronological chain of events."""
    sequence_number: int            # 1-based order in the timeline
    date_or_time: str               # Date, time, or relative ("Day 3", "00:02:14")
    description: str                # What happened — precise and factual
    actors: List[str]               # People involved
    verbatim_quote: Optional[str]   # Exact words said/written, if applicable
    source_file: str                # Which document this comes from
    location_in_source: str         # Timestamp, page, or section
    event_type: str                 # statement|action|admission|threat|agreement|refusal|observation|other
    legal_significance: str         # Why this event matters legally
    certainty: str                  # confirmed|inferred|disputed
    linked_statements: List[str]    # Verbatim statement keys (speaker@timestamp) referenced here


@dataclass
class ChainOfEvents:
    """
    Complete chronological chain of events reconstructed from all evidence.

    This is the authoritative factual record used to:
    1. Drive document generation (ensures completeness)
    2. Audit generated documents (catches omissions and inaccuracies)
    """
    events: List[ChainEvent] = field(default_factory=list)
    all_statements: List[VerbatimStatement] = field(default_factory=list)
    key_actors: Dict[str, str] = field(default_factory=dict)   # name → role
    key_dates: List[str] = field(default_factory=list)         # "DATE: description" strings
    narrative_summary: str = ""
    total_events: int = 0
    earliest_date: Optional[str] = None
    latest_date: Optional[str] = None

    def to_court_narrative(self) -> str:
        """
        Format the chain as a numbered chronological narrative for inclusion
        in court document generation prompts and review passes.
        """
        lines = [
            "CHRONOLOGICAL CHAIN OF EVENTS",
            "=" * 60,
        ]
        if self.earliest_date or self.latest_date:
            lines.append(
                f"Period: {self.earliest_date or '?'} — {self.latest_date or '?'}"
            )
        if self.key_actors:
            lines.append(
                "Key Actors: "
                + ", ".join(f"{n} ({r})" for n, r in self.key_actors.items())
            )
        if self.key_dates:
            lines += ["", "KEY DATES:"] + [f"  • {d}" for d in self.key_dates]
        lines += ["", "EVENTS:"]
        for evt in self.events:
            lines += [
                f"\n[{evt.sequence_number}] {evt.date_or_time}",
                f"    What:     {evt.description}",
                f"    Who:      {', '.join(evt.actors) if evt.actors else 'Unknown'}",
                f"    Type:     {evt.event_type} | Certainty: {evt.certainty}",
                f"    Source:   {evt.source_file} @ {evt.location_in_source}",
                f"    Legal:    {evt.legal_significance}",
            ]
            if evt.verbatim_quote:
                lines.append(f'    Quote:    "{evt.verbatim_quote}"')
        return "\n".join(lines)

    def to_statements_index(self) -> str:
        """
        Format the complete verbatim statements index for review prompts.
        Every direct quote from every source, indexed by speaker and location.
        """
        if not self.all_statements:
            return "No verbatim statements extracted."
        lines = [
            "COMPLETE VERBATIM STATEMENTS INDEX",
            "=" * 60,
            "(Every direct quote from every source document)",
            "",
        ]
        for i, stmt in enumerate(self.all_statements, 1):
            lines += [
                f"[S{i}] {stmt.speaker} @ {stmt.timestamp_or_location} ({stmt.source_file})",
                f'     "{stmt.statement}"',
                f"     Context: {stmt.context}",
                "",
            ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative_summary": self.narrative_summary,
            "earliest_date": self.earliest_date,
            "latest_date": self.latest_date,
            "total_events": self.total_events,
            "key_actors": self.key_actors,
            "key_dates": self.key_dates,
            "events": [
                {
                    "sequence_number": e.sequence_number,
                    "date_or_time": e.date_or_time,
                    "description": e.description,
                    "actors": e.actors,
                    "verbatim_quote": e.verbatim_quote,
                    "source_file": e.source_file,
                    "location_in_source": e.location_in_source,
                    "event_type": e.event_type,
                    "legal_significance": e.legal_significance,
                    "certainty": e.certainty,
                    "linked_statements": e.linked_statements,
                }
                for e in self.events
            ],
            "all_statements": [
                {
                    "speaker": s.speaker,
                    "statement": s.statement,
                    "timestamp_or_location": s.timestamp_or_location,
                    "source_file": s.source_file,
                    "context": s.context,
                    "statement_type": s.statement_type,
                }
                for s in self.all_statements
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ChainExtractor:
    """
    Builds a structured chain of events from already-extracted evidence data.

    Operates entirely on text — no file API uploads required. Takes the
    output of EvidenceAnalyzer (transcript, evidence_items, key_findings)
    and synthesizes it into a strict chronological record.

    Uses gemini-flash-latest — cheap and fast for structured synthesis.
    """

    _MODEL = "gemini-flash-latest"

    def __init__(self, config: LegalAIConfig):
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed.")
        self.config = config
        genai.configure(api_key=config.google_api_key)

    async def extract(
        self,
        situation_description: str,
        full_transcript: Optional[str],
        evidence_items: List[Any],   # List[EvidenceItem] from evidence_analyzer
        key_findings: List[str],
        files_analyzed: List[str],
        jurisdiction: USState,
    ) -> ChainOfEvents:
        """
        Build the complete chain of events from evidence analysis output.

        Args:
            situation_description: The case situation (user's description).
            full_transcript: Verbatim transcript extracted from audio/video, or None.
            evidence_items: List of EvidenceItem objects from the analysis.
            key_findings: Key findings list from the analysis.
            files_analyzed: Names of all analyzed files.
            jurisdiction: Legal jurisdiction.

        Returns:
            ChainOfEvents — never raises; returns empty chain on failure.
        """
        items_json = json.dumps([
            {
                "description": getattr(item, "description", ""),
                "evidence_type": getattr(item, "evidence_type", ""),
                "relevance_score": getattr(item, "relevance_score", 0.0),
                "location": getattr(item, "location", None),
                "speaker_or_source": getattr(item, "speaker_or_source", None),
                "legal_significance": getattr(item, "legal_significance", ""),
                "verbatim_excerpt": getattr(item, "verbatim_excerpt", None),
                "supports_case": getattr(item, "supports_case", None),
                "admissibility_concerns": getattr(item, "admissibility_concerns", None),
            }
            for item in evidence_items
        ], indent=2)

        transcript_section = ""
        if full_transcript:
            truncated = full_transcript[:10000]
            suffix = "\n[...transcript continues — only first 10,000 chars shown...]" \
                if len(full_transcript) > 10000 else ""
            transcript_section = f"\nFULL TRANSCRIPT (verbatim):\n{truncated}{suffix}\n"

        findings_block = "\n".join(f"  • {f}" for f in key_findings) if key_findings else "  None."

        prompt = f"""You are a forensic legal analyst specializing in chronological fact reconstruction.

Your task: build a COMPLETE, EXHAUSTIVE chain of events from the evidence data below.
This chain will be used directly in court filings — nothing can be omitted.

CASE SITUATION:
{situation_description}

JURISDICTION: {jurisdiction.value}
FILES ANALYZED: {', '.join(files_analyzed)}

KEY FINDINGS FROM EVIDENCE ANALYSIS:
{findings_block}
{transcript_section}
ALL EXTRACTED EVIDENCE ITEMS:
{items_json}

═══════════════════════════════════════════════════════════
TASK 1 — VERBATIM STATEMENTS INDEX
═══════════════════════════════════════════════════════════
Extract EVERY direct quote spoken or written in the source materials.
Include ALL statements — not just legally significant ones. A statement
that seems minor NOW may become critical later.

For each statement:
• Who said/wrote it (exact name or speaker label)
• Exact words — NEVER paraphrase
• Precise location (timestamp MM:SS, page, section)
• Which source file it came from
• One sentence of surrounding context

═══════════════════════════════════════════════════════════
TASK 2 — CHRONOLOGICAL CHAIN OF EVENTS
═══════════════════════════════════════════════════════════
Reconstruct EVERY event in strict chronological order (earliest first).

For each event:
• Exact date/time or relative position in the sequence
• All people involved and their roles
• Verbatim quote if words were spoken/written (from Task 1)
• Exact source file and location
• Whether confirmed (directly evidenced), inferred, or disputed
• Specific legal significance

Requirements:
- EVERY evidence item must appear in at least one event
- Events must establish clear cause-and-effect chains
- Do NOT omit events that seem repetitive — list each occurrence separately
- Subtle background statements often prove intent — include all of them

Return ONLY valid JSON (no markdown, no preamble):
{{
    "narrative_summary": "2-3 sentence factual overview of the complete timeline",
    "earliest_date": "Earliest confirmed date/time or null if unknown",
    "latest_date": "Latest confirmed date/time or null if unknown",
    "key_actors": {{
        "Exact Name or Label": "Their role (Petitioner|Respondent|Witness|Unknown)"
    }},
    "key_dates": [
        "YYYY-MM-DD or timestamp: brief description of what happened"
    ],
    "all_statements": [
        {{
            "speaker": "Name or Speaker 1",
            "statement": "Exact verbatim words — never paraphrase",
            "timestamp_or_location": "MM:SS or page X or section Y",
            "source_file": "filename.ext",
            "context": "One sentence: what was happening when this was said/written",
            "statement_type": "spoken|written|on_screen|typed"
        }}
    ],
    "events": [
        {{
            "sequence_number": 1,
            "date_or_time": "Exact date/time or relative (e.g. '00:02:14' or '2024-01-15' or 'Day 1')",
            "description": "Precise factual description of what happened",
            "actors": ["Person A", "Person B"],
            "verbatim_quote": "Exact words if applicable, else null",
            "source_file": "filename.ext",
            "location_in_source": "Timestamp MM:SS or page X or section Y",
            "event_type": "statement|action|admission|threat|agreement|refusal|observation|other",
            "legal_significance": "Specific legal relevance to the case",
            "certainty": "confirmed|inferred|disputed",
            "linked_statements": ["Speaker@timestamp keys of any statements spoken during this event"]
        }}
    ]
}}

CRITICAL: Be exhaustive. In legal proceedings, a single overlooked statement
can be case-determinative. Process every piece of evidence."""

        model = genai.GenerativeModel(self._MODEL)
        gen_config = GenerationConfig(
            temperature=0.05,        # Near-deterministic — factual extraction only
            max_output_tokens=16384,
            response_mime_type="application/json",
        )

        try:
            response = await asyncio.to_thread(
                model.generate_content, prompt, generation_config=gen_config
            )
            data = json.loads(response.text)
        except Exception as exc:
            # Non-fatal: return empty chain if extraction fails
            return ChainOfEvents(
                narrative_summary=f"[Chain extraction could not complete: {exc}]",
                total_events=0,
            )

        # Parse events
        events: List[ChainEvent] = []
        for raw in data.get("events", []):
            events.append(ChainEvent(
                sequence_number=int(raw.get("sequence_number", 0)),
                date_or_time=str(raw.get("date_or_time", "Unknown")),
                description=str(raw.get("description", "")),
                actors=list(raw.get("actors", [])),
                verbatim_quote=raw.get("verbatim_quote") or None,
                source_file=str(raw.get("source_file", "")),
                location_in_source=str(raw.get("location_in_source", "")),
                event_type=str(raw.get("event_type", "other")),
                legal_significance=str(raw.get("legal_significance", "")),
                certainty=str(raw.get("certainty", "confirmed")),
                linked_statements=list(raw.get("linked_statements", [])),
            ))

        # Parse verbatim statements
        statements: List[VerbatimStatement] = []
        for raw in data.get("all_statements", []):
            stmt_text = (raw.get("statement") or "").strip()
            if not stmt_text:
                continue
            statements.append(VerbatimStatement(
                speaker=str(raw.get("speaker", "Unknown")),
                statement=stmt_text,
                timestamp_or_location=str(raw.get("timestamp_or_location", "")),
                source_file=str(raw.get("source_file", "")),
                context=str(raw.get("context", "")),
                statement_type=str(raw.get("statement_type", "spoken")),
            ))

        return ChainOfEvents(
            events=events,
            all_statements=statements,
            key_actors=dict(data.get("key_actors", {})),
            key_dates=list(data.get("key_dates", [])),
            narrative_summary=str(data.get("narrative_summary", "")),
            total_events=len(events),
            earliest_date=data.get("earliest_date") or None,
            latest_date=data.get("latest_date") or None,
        )

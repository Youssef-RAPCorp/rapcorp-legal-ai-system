"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    EVIDENCE ANALYZER                                          ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Identifies legally relevant evidence from:                                   ║
║  • Audio files  (.mp3, .wav, .aac, .ogg, .flac, .m4a, .aiff, .wma)          ║
║  • Video files  (.mp4, .avi, .mov, .mkv, .webm, .flv, .wmv, .mpeg, .3gp)   ║
║  • Image files  (.png, .jpg, .jpeg, .gif, .webp, .bmp)                       ║
║  • Text files   (.txt, .md, .csv, .json, .xml, .html)                        ║
║  • Documents    (.pdf)                                                        ║
║                                                                               ║
║  Usage:                                                                       ║
║    analyzer = EvidenceAnalyzer(config)                                        ║
║    result = await analyzer.analyze_evidence(                                  ║
║        situation_description="Describe your case here...",                    ║
║        file_paths=["recording.mp3", "document.pdf", "notes.txt"],             ║
║        jurisdiction=USState.FEDERAL                                           ║
║    )                                                                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from enum import Enum
import json

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from configs.config import LegalAIConfig, USState

if TYPE_CHECKING:
    from src.extraction.chain_extractor import ChainOfEvents


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIA TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class MediaType(Enum):
    """Type of media file."""
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


# File extension sets by media type
AUDIO_EXTENSIONS    = {'.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a', '.aiff', '.wma'}
VIDEO_EXTENSIONS    = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg', '.3gp'}
IMAGE_EXTENSIONS    = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
TEXT_EXTENSIONS     = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.rtf', '.log'}
DOCUMENT_EXTENSIONS = {'.pdf'}

# MIME types for File API uploads
_MIME_OVERRIDES = {
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.aac': 'audio/aac',
    '.ogg': 'audio/ogg',
    '.flac': 'audio/flac',
    '.m4a': 'audio/mp4',
    '.aiff': 'audio/aiff',
    '.wma': 'audio/x-ms-wma',
    '.mp4': 'video/mp4',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
    '.mkv': 'video/x-matroska',
    '.webm': 'video/webm',
    '.flv': 'video/x-flv',
    '.wmv': 'video/x-ms-wmv',
    '.mpeg': 'video/mpeg',
    '.mpg': 'video/mpeg',
    '.3gp': 'video/3gpp',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.bmp': 'image/bmp',
    '.pdf': 'application/pdf',
}


def get_media_type(file_path: str) -> MediaType:
    """Determine the media type of a file from its extension."""
    ext = Path(file_path).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return MediaType.AUDIO
    if ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    if ext in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    if ext in TEXT_EXTENSIONS:
        return MediaType.TEXT
    if ext in DOCUMENT_EXTENSIONS:
        return MediaType.DOCUMENT
    return MediaType.UNKNOWN


def _get_mime_type(file_path: str) -> str:
    """Get the MIME type for a file."""
    ext = Path(file_path).suffix.lower()
    if ext in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[ext]
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceItem:
    """A single piece of evidence identified in the analyzed media."""
    description: str
    evidence_type: str           # e.g. verbal_admission, on_screen_text, non_verbal_conduct
    relevance_score: float       # 0.0 – 1.0
    legal_significance: str      # Specific legal effect on the case
    location: Optional[str] = None           # Exact timestamp MM:SS, page, or section
    speaker_or_source: Optional[str] = None  # Speaker label/name or document source
    supports_case: Optional[bool] = None     # True=supports, False=undermines, None=neutral
    admissibility_concerns: Optional[str] = None
    verbatim_excerpt: Optional[str] = None   # Exact words — never paraphrased


@dataclass
class EvidenceAnalysisResult:
    """Complete result of an evidence analysis session."""
    situation_description: str
    files_analyzed: List[str]
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    # Full verbatim transcript (video/audio) or None for text-only files
    full_transcript: Optional[str] = None
    # Chronological visual scene summary (video only)
    visual_summary: Optional[str] = None
    # All on-screen text found in video frames
    onscreen_text: List[str] = field(default_factory=list)
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    total_evidence_count: int = 0
    high_relevance_count: int = 0   # items with relevance_score >= 0.7
    model_used: str = ""
    cost: float = 0.0
    # Chain of events — built after initial analysis, no file re-uploads
    chain_of_events: Optional[Any] = None  # ChainOfEvents from chain_extractor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "situation_description": self.situation_description,
            "files_analyzed": self.files_analyzed,
            "full_transcript": self.full_transcript,
            "visual_summary": self.visual_summary,
            "onscreen_text": self.onscreen_text,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "recommended_actions": self.recommended_actions,
            "total_evidence_count": self.total_evidence_count,
            "high_relevance_count": self.high_relevance_count,
            "model_used": self.model_used,
            "cost": self.cost,
            "evidence_items": [
                {
                    "description": item.description,
                    "evidence_type": item.evidence_type,
                    "relevance_score": item.relevance_score,
                    "legal_significance": item.legal_significance,
                    "location": item.location,
                    "speaker_or_source": item.speaker_or_source,
                    "supports_case": item.supports_case,
                    "admissibility_concerns": item.admissibility_concerns,
                    "verbatim_excerpt": item.verbatim_excerpt,
                }
                for item in self.evidence_items
            ],
            "chain_of_events": self.chain_of_events.to_dict() if self.chain_of_events else None,
        }

    def print_report(self) -> None:
        """Print a formatted evidence report to stdout."""
        print("\n" + "=" * 70)
        print("  EVIDENCE ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nFiles Analyzed: {', '.join(self.files_analyzed)}")
        print(f"Total Evidence Items: {self.total_evidence_count}")
        print(f"High-Relevance Items (>=0.7): {self.high_relevance_count}")
        print(f"Model Used: {self.model_used}  |  Est. Cost: ${self.cost:.6f}")

        if self.summary:
            print(f"\nSUMMARY:\n{self.summary}")

        if self.full_transcript:
            print("\nFULL TRANSCRIPT:")
            # Print with indentation, wrapped at 80 chars per line
            for line in self.full_transcript.splitlines():
                print(f"  {line}")

        if self.visual_summary:
            print("\nVISUAL SCENE SUMMARY:")
            for line in self.visual_summary.splitlines():
                print(f"  {line}")

        if self.onscreen_text:
            print("\nON-SCREEN TEXT FOUND IN VIDEO:")
            for text in self.onscreen_text:
                print(f"  • {text}")

        if self.key_findings:
            print("\nKEY FINDINGS:")
            for i, finding in enumerate(self.key_findings, 1):
                print(f"  {i}. {finding}")

        if self.evidence_items:
            print("\nEVIDENCE ITEMS (sorted by relevance):")
            sorted_items = sorted(self.evidence_items, key=lambda x: x.relevance_score, reverse=True)
            for i, item in enumerate(sorted_items, 1):
                relevance_bar = "█" * int(item.relevance_score * 10)
                supports = ""
                if item.supports_case is True:
                    supports = " [SUPPORTS]"
                elif item.supports_case is False:
                    supports = " [UNDERMINES]"
                print(f"\n  [{i}] {item.description}{supports}")
                print(f"       Type: {item.evidence_type}  |  Relevance: {item.relevance_score:.1f} {relevance_bar}")
                if item.location:
                    print(f"       Location: {item.location}")
                if item.speaker_or_source:
                    print(f"       Source: {item.speaker_or_source}")
                print(f"       Significance: {item.legal_significance}")
                if item.verbatim_excerpt:
                    print(f"       Verbatim: \"{item.verbatim_excerpt}\"")
                if item.admissibility_concerns:
                    print(f"       ⚠ Admissibility: {item.admissibility_concerns}")

        if self.recommended_actions:
            print("\nRECOMMENDED ACTIONS:")
            for i, action in enumerate(self.recommended_actions, 1):
                print(f"  {i}. {action}")

        if self.chain_of_events:
            coe = self.chain_of_events
            print("\nCHAIN OF EVENTS:")
            print(f"  Events extracted : {coe.total_events}")
            print(f"  Statements index : {len(coe.all_statements)} verbatim quotes")
            if coe.earliest_date or coe.latest_date:
                print(f"  Period           : {coe.earliest_date or '?'} — {coe.latest_date or '?'}")
            if coe.key_actors:
                actors = ", ".join(
                    f"{n} ({r})" for n, r in list(coe.key_actors.items())[:5]
                )
                print(f"  Key actors       : {actors}")
            if coe.narrative_summary:
                print(f"  Summary          : {coe.narrative_summary}")

        print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class EvidenceAnalyzer:
    """
    Analyzes audio, video, and text files to identify evidence
    relevant to a legal case situation.

    For audio and video files, uses Gemini's File API to upload and process
    the media. For text files and PDFs (text-extractable), content is read
    and included directly in the prompt.

    Supported formats:
        Audio:    MP3, WAV, AAC, OGG, FLAC, M4A, AIFF, WMA
        Video:    MP4, AVI, MOV, MKV, WebM, FLV, WMV, MPEG, 3GP
        Image:    PNG, JPG, JPEG, GIF, WebP, BMP (via File API — full OCR + vision)
        Text:     TXT, MD, CSV, JSON, XML, HTML, RTF
        Document: PDF (via File API)
    """

    # Use Pro for the deepest multimodal reasoning; Flash is also capable
    _DEFAULT_MODEL = "gemini-pro-latest"
    # Max chars to read from a single text file (keep prompt manageable)
    _TEXT_FILE_CHAR_LIMIT = 40_000

    def __init__(self, config: LegalAIConfig):
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai>=0.8.3"
            )
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set in configuration.")

        self.config = config
        genai.configure(api_key=config.google_api_key)

    # ─── Internal helpers ────────────────────────────────────────────────────

    async def _upload_file(self, file_path: str) -> Any:
        """Upload a file to the Gemini File API and wait for processing."""
        mime_type = _get_mime_type(file_path)
        file_name = Path(file_path).name
        print(f"    Uploading {file_name} ({mime_type})...")

        uploaded = await asyncio.to_thread(
            genai.upload_file,
            path=file_path,
            mime_type=mime_type,
            display_name=file_name,
        )

        # Poll until the file is ready
        max_wait = 120  # seconds
        waited = 0
        while uploaded.state.name == "PROCESSING":
            await asyncio.sleep(3)
            waited += 3
            uploaded = await asyncio.to_thread(genai.get_file, name=uploaded.name)
            if waited >= max_wait:
                raise RuntimeError(
                    f"File {file_name} did not finish processing within {max_wait}s."
                )

        if uploaded.state.name == "FAILED":
            raise RuntimeError(f"File upload failed for: {file_name}")

        print(f"    Ready: {file_name}")
        return uploaded

    def _read_text_file(self, file_path: str) -> str:
        """Read a text file, truncating if very large."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(self._TEXT_FILE_CHAR_LIMIT)
            if len(content) == self._TEXT_FILE_CHAR_LIMIT:
                content += "\n[... content truncated for context limit ...]"
            return content
        except OSError as exc:
            print(f"    Warning: Could not read {Path(file_path).name}: {exc}")
            return ""

    def _build_prompt(
        self,
        situation_description: str,
        media_descriptions: List[str],
        jurisdiction: USState,
    ) -> str:
        return f"""You are a forensic legal evidence analyst. Your task is to perform a \
complete, exhaustive extraction of ALL content from the provided media and then identify \
every piece of evidence relevant to the legal case situation below.

CASE SITUATION:
{situation_description}

JURISDICTION: {jurisdiction.value}

CONTENT PROVIDED: {', '.join(media_descriptions)}

═══════════════════════════════════════════════════
STEP 1 — FULL CONTENT EXTRACTION (for video/audio)
═══════════════════════════════════════════════════
For every VIDEO file:
  • Produce a complete verbatim transcript of ALL spoken words, labeling each speaker
    (Speaker 1, Speaker 2, etc., or by name if identifiable) with timestamps (MM:SS).
  • Describe EVERY visible scene, action, object, and person at key moments — include
    what is happening visually even when no one is speaking.
  • Extract ALL on-screen text: signs, documents, screens, overlays, captions, labels.
  • Note any significant non-speech audio: raised voices, sounds of physical contact,
    background noise that is legally relevant, silences of note.
  • Correlate what is said with what is simultaneously shown on screen.

For every AUDIO file:
  • Produce a complete verbatim transcript with speaker labels and timestamps (MM:SS).
  • Note tone, emotion, and non-verbal audio events (crying, shouting, physical sounds).
  • Flag any portions that are unclear, inaudible, or potentially edited.

For every IMAGE file (screenshots, photos, scanned documents):
  • Extract and transcribe ALL visible text verbatim — every word, including UI elements,
    headers, message bubbles, names, timestamps, labels, and fine print.
  • Describe every person, object, location, or action visible in the image.
  • For screenshots of messages/chats: reproduce the full conversation thread with sender
    names, timestamps, and message content exactly as displayed.
  • Note metadata visible in the image: file timestamps, device info, location data.

For every TEXT / DOCUMENT file:
  • Reproduce all key passages verbatim.
  • Identify dates, names, dollar amounts, deadlines, and obligations.

═══════════════════════════════════════════════════
STEP 2 — EVIDENCE IDENTIFICATION
═══════════════════════════════════════════════════
After extracting all content, identify EVERY piece of evidence that is relevant to the
case situation. For each item:
  1. Give an exact timestamp (MM:SS), page number, or section reference.
  2. Assess relevance on a 0.0–1.0 scale (1.0 = directly determinative).
  3. Quote the verbatim spoken words, text, or describe the visual exactly.
  4. State how it supports or undermines the case legally.
  5. Flag any admissibility concerns (hearsay, authentication, chain of custody,
     best-evidence rule, attorney-client privilege, etc.).
  6. Identify the speaker or source when known.

Return your analysis ONLY as valid JSON — no extra text before or after:
{{
    "full_transcript": "For video/audio: complete verbatim transcript with speaker labels and timestamps. For text files: omit this field or set to null.",
    "visual_summary": "For video: chronological description of key visual scenes with timestamps. For non-video files: null.",
    "onscreen_text": ["Any text visible in video frames, verbatim"],
    "summary": "One-paragraph overview of all content and its overall evidentiary value",
    "key_findings": [
        "Most critical finding 1 (with timestamp/location)",
        "Most critical finding 2 (with timestamp/location)"
    ],
    "recommended_actions": [
        "Specific next step 1",
        "Specific next step 2"
    ],
    "evidence_items": [
        {{
            "description": "Precise description — who said/did what, where, when",
            "evidence_type": "witness_statement|verbal_admission|physical_evidence|documentary|circumstantial|expert_opinion|digital_evidence|audio_recording|video_recording|written_statement|on_screen_text|non_verbal_conduct",
            "relevance_score": 0.0,
            "speaker_or_source": "Speaker 1 / Person name / Document name, or null",
            "legal_significance": "Specific legal effect on the case (elements proven, defenses implicated, etc.)",
            "location": "Timestamp MM:SS or page/section — be precise",
            "supports_case": true,
            "admissibility_concerns": "Specific rule or concern, or null",
            "verbatim_excerpt": "Exact words spoken or written — never paraphrase for this field"
        }}
    ]
}}

IMPORTANT: Be completely exhaustive. In legal proceedings, a single overlooked statement \
or visual detail can be case-determinative. Process every second of video/audio and every \
line of text."""

    # ─── Public API ──────────────────────────────────────────────────────────

    async def analyze_evidence(
        self,
        situation_description: str,
        file_paths: List[str],
        jurisdiction: USState = USState.FEDERAL,
        model_override: Optional[str] = None,
    ) -> EvidenceAnalysisResult:
        """
        Analyze files to identify evidence relevant to a legal case.

        Args:
            situation_description: Description of the case situation.
            file_paths: List of file paths (audio, video, text, or PDF).
            jurisdiction: Legal jurisdiction for contextual analysis.
            model_override: Override the default model (gemini-pro-latest).

        Returns:
            EvidenceAnalysisResult with all identified evidence items.

        Raises:
            FileNotFoundError: If any file_path does not exist.
            ValueError: If file_paths is empty.
            RuntimeError: If no files could be processed.
        """
        if not file_paths:
            raise ValueError("At least one file path must be provided.")

        for fp in file_paths:
            if not Path(fp).exists():
                raise FileNotFoundError(f"File not found: {fp}")

        model_id = model_override or self._DEFAULT_MODEL
        print(f"\n  Analyzing {len(file_paths)} file(s) for evidence...")

        content_parts: List[Any] = []
        media_descriptions: List[str] = []
        uploaded_files: List[Any] = []

        for file_path in file_paths:
            media_type = get_media_type(file_path)
            file_name = Path(file_path).name

            if media_type == MediaType.TEXT:
                text = self._read_text_file(file_path)
                content_parts.append(
                    f"\n[TEXT FILE: {file_name}]\n{text}\n[END OF {file_name}]\n"
                )
                media_descriptions.append(f"text file '{file_name}'")

            elif media_type in (MediaType.AUDIO, MediaType.VIDEO, MediaType.IMAGE, MediaType.DOCUMENT):
                try:
                    uploaded = await self._upload_file(file_path)
                    content_parts.append(uploaded)
                    uploaded_files.append(uploaded)
                    media_descriptions.append(f"{media_type.value} file '{file_name}'")
                except Exception as exc:
                    print(f"    Warning: Could not upload {file_name} — skipping: {exc}")

            else:
                # Attempt to read as text; skip if that fails
                try:
                    text = self._read_text_file(file_path)
                    content_parts.append(
                        f"\n[FILE: {file_name}]\n{text}\n[END OF {file_name}]\n"
                    )
                    media_descriptions.append(f"file '{file_name}'")
                    print(f"    Warning: Unknown type for {file_name}, treating as text.")
                except Exception:
                    print(f"    Warning: Could not process {file_name}, skipping.")

        if not content_parts:
            raise RuntimeError("No files could be processed.")

        prompt = self._build_prompt(situation_description, media_descriptions, jurisdiction)
        contents = content_parts + [prompt]

        print(f"    Running evidence analysis with {model_id}...")

        gen_config = GenerationConfig(
            temperature=0.1,   # Low temperature for factual, consistent output
            max_output_tokens=32768,  # Large budget for full transcripts + evidence
            response_mime_type="application/json",
        )

        model = genai.GenerativeModel(model_id)
        start = time.time()
        response_text = ""
        last_exc: Exception = RuntimeError("Unknown error")
        for attempt in range(1, 4):
            try:
                response = await asyncio.to_thread(
                    model.generate_content,
                    contents,
                    generation_config=gen_config,
                )
                response_text = response.text if hasattr(response, "text") else ""
                if response_text:
                    break
                last_exc = RuntimeError("Gemini returned an empty response")
            except Exception as exc:
                last_exc = exc
                print(f"    Evidence analysis attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                await asyncio.sleep(2 ** attempt)

        if not response_text:
            for uploaded in uploaded_files:
                try:
                    await asyncio.to_thread(genai.delete_file, name=uploaded.name)
                except Exception:
                    pass
            raise RuntimeError(
                f"Evidence analysis API call failed after 3 attempts: {last_exc}"
            )
        elapsed = time.time() - start

        # Rough cost estimate using Pro pricing
        input_tokens = (len(prompt) + len(situation_description)) // 4
        output_tokens = len(response_text) // 4
        cost = (input_tokens * 0.0005 / 1000) + (output_tokens * 0.001 / 1000)

        # Parse JSON
        evidence_items: List[EvidenceItem] = []
        summary = ""
        full_transcript: Optional[str] = None
        visual_summary: Optional[str] = None
        onscreen_text: List[str] = []
        key_findings: List[str] = []
        recommended_actions: List[str] = []

        try:
            data = json.loads(response_text)
            # If model wrapped the object in a list, unwrap it
            if isinstance(data, list):
                data = data[0] if data else {}
            summary = data.get("summary", "")
            full_transcript = data.get("full_transcript") or None
            visual_summary = data.get("visual_summary") or None
            onscreen_text = data.get("onscreen_text") or []
            key_findings = data.get("key_findings", [])
            recommended_actions = data.get("recommended_actions", [])

            for raw in data.get("evidence_items", []):
                evidence_items.append(EvidenceItem(
                    description=str(raw.get("description", "")),
                    evidence_type=str(raw.get("evidence_type", "unknown")),
                    relevance_score=float(raw.get("relevance_score", 0.0)),
                    legal_significance=str(raw.get("legal_significance", "")),
                    location=raw.get("location"),
                    speaker_or_source=raw.get("speaker_or_source"),
                    supports_case=raw.get("supports_case"),
                    admissibility_concerns=raw.get("admissibility_concerns"),
                    verbatim_excerpt=raw.get("verbatim_excerpt"),
                ))
        except (json.JSONDecodeError, TypeError) as parse_exc:
            print(f"    Warning: Evidence analysis returned malformed JSON — {parse_exc}")
            summary = (
                "[WARNING] Evidence analysis JSON parsing failed. "
                "Raw AI response is attached below — manual review required."
            )
            key_findings = ["[WARNING] Automated evidence extraction failed — review raw response"]
            evidence_items.append(EvidenceItem(
                description=(
                    "[RAW ANALYSIS — JSON PARSE FAILED]\n"
                    + (response_text[:3000] if response_text else "No response received.")
                ),
                evidence_type="parse_error",
                relevance_score=0.0,
                legal_significance="[Manual review required — automated extraction did not complete]",
            ))

        # Clean up uploaded files (non-critical)
        for uploaded in uploaded_files:
            try:
                await asyncio.to_thread(genai.delete_file, name=uploaded.name)
            except Exception:
                pass

        high_relevance = sum(1 for item in evidence_items if item.relevance_score >= 0.7)

        result = EvidenceAnalysisResult(
            situation_description=situation_description,
            files_analyzed=[Path(fp).name for fp in file_paths],
            evidence_items=evidence_items,
            full_transcript=full_transcript,
            visual_summary=visual_summary,
            onscreen_text=onscreen_text,
            summary=summary,
            key_findings=key_findings,
            recommended_actions=recommended_actions,
            total_evidence_count=len(evidence_items),
            high_relevance_count=high_relevance,
            model_used=model_id,
            cost=cost,
        )

        # Build chain of events from the extracted data (no file re-uploads)
        if evidence_items:
            print("    Building chain of events...")
            try:
                from src.extraction.chain_extractor import ChainExtractor
                chain_extractor = ChainExtractor(self.config)
                result.chain_of_events = await chain_extractor.extract(
                    situation_description=situation_description,
                    full_transcript=full_transcript,
                    evidence_items=evidence_items,
                    key_findings=key_findings,
                    files_analyzed=[Path(fp).name for fp in file_paths],
                    jurisdiction=jurisdiction,
                )
                if result.chain_of_events:
                    print(f"    Chain of events: {result.chain_of_events.total_events} events, "
                          f"{len(result.chain_of_events.all_statements)} verbatim statements")
            except Exception as exc:
                print(f"    Warning: Chain extraction failed — {exc}")

        return result

    async def generate_clarifying_questions(
        self,
        result: EvidenceAnalysisResult,
        situation_description: str,
    ) -> List[str]:
        """
        Generate targeted clarifying questions based on uncertainties in the
        analysis.  Uses Flash (no file uploads needed — text only).

        Returns a list of specific questions the user should answer to resolve
        ambiguities and strengthen the analysis.
        """
        uncertain_items = [
            {"description": i.description, "relevance": i.relevance_score,
             "significance": i.legal_significance, "location": i.location}
            for i in result.evidence_items if i.relevance_score < 0.75
        ]

        prompt = f"""You are a legal analyst reviewing an initial evidence analysis for a petition.

CASE SITUATION:
{situation_description}

ANALYSIS SUMMARY:
{result.summary}

KEY FINDINGS:
{json.dumps(result.key_findings, indent=2)}

UNCERTAIN / LOW-CONFIDENCE EVIDENCE ITEMS (relevance < 0.75):
{json.dumps(uncertain_items, indent=2)}

Based on this analysis, generate the most important clarifying questions that the
petitioner should answer to:
1. Resolve ambiguities or misidentified evidence
2. Provide missing context that changes the legal significance of an item
3. Confirm facts that could not be verified from the files alone
4. Identify additional evidence the petitioner may have but didn't submit
5. Clarify the precise legal claims being made

Rules:
- Ask only questions that CANNOT be answered from the files already analyzed
- Be specific — reference exact items, timestamps, or findings by name
- Prioritize questions whose answers would most change the legal outcome
- Do NOT ask yes/no questions — ask open-ended questions that elicit detail
- Return between 3 and 8 questions

Return ONLY a JSON array of question strings:
["Question 1?", "Question 2?", ...]"""

        model = genai.GenerativeModel("gemini-flash-latest")
        gen_config = GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
            response_mime_type="application/json",
        )
        try:
            response = await asyncio.to_thread(
                model.generate_content, prompt, generation_config=gen_config
            )
            questions = json.loads(response.text)
            return questions if isinstance(questions, list) else []
        except Exception:
            return []

    async def refine_with_clarifications(
        self,
        original_result: EvidenceAnalysisResult,
        situation_description: str,
        clarifications: Dict[str, str],
        jurisdiction: USState = USState.FEDERAL,
    ) -> EvidenceAnalysisResult:
        """
        Refine the evidence analysis using user-provided clarifications —
        WITHOUT re-uploading the original files.  All prior findings are
        preserved and updated based on the new context.
        """
        clarification_block = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in clarifications.items()
        )

        original_items_json = json.dumps([
            {
                "description": i.description,
                "evidence_type": i.evidence_type,
                "relevance_score": i.relevance_score,
                "location": i.location,
                "speaker_or_source": i.speaker_or_source,
                "supports_case": i.supports_case,
                "legal_significance": i.legal_significance,
                "admissibility_concerns": i.admissibility_concerns,
                "verbatim_excerpt": i.verbatim_excerpt,
            }
            for i in original_result.evidence_items
        ], indent=2)

        prompt = f"""You are a legal analyst refining a prior evidence analysis after receiving
clarifications from the petitioner.  DO NOT re-examine the original files — only
update the analysis based on the clarifications provided below.

JURISDICTION: {jurisdiction.value}

ORIGINAL CASE SITUATION:
{situation_description}

PETITIONER CLARIFICATIONS:
{clarification_block}

ORIGINAL TRANSCRIPT (if available):
{(original_result.full_transcript or 'N/A')[:3000]}

ORIGINAL EVIDENCE ITEMS:
{original_items_json}

ORIGINAL SUMMARY:
{original_result.summary}

ORIGINAL KEY FINDINGS:
{json.dumps(original_result.key_findings, indent=2)}

INSTRUCTIONS:
1. Update relevance scores for items affected by the clarifications
2. Correct any misidentified evidence
3. Add new evidence items revealed by the clarifications
4. Update legal significance to reflect the clarified context
5. Revise key findings and recommended actions accordingly
6. If the clarifications confirm a finding was WRONG, remove or downgrade it
7. Be specific — reference the clarifications explicitly

Return the COMPLETE refined analysis as JSON (same schema as before):
{{
    "full_transcript": null,
    "visual_summary": null,
    "onscreen_text": [],
    "summary": "Revised summary incorporating clarifications",
    "key_findings": ["Updated finding 1", ...],
    "recommended_actions": ["Updated action 1", ...],
    "evidence_items": [
        {{
            "description": "...",
            "evidence_type": "...",
            "relevance_score": 0.0,
            "speaker_or_source": "...",
            "legal_significance": "...",
            "location": "...",
            "supports_case": true,
            "admissibility_concerns": "...",
            "verbatim_excerpt": "..."
        }}
    ]
}}"""

        model = genai.GenerativeModel(self._DEFAULT_MODEL)
        gen_config = GenerationConfig(
            temperature=0.1,
            max_output_tokens=16384,
            response_mime_type="application/json",
        )
        response = await asyncio.to_thread(
            model.generate_content, prompt, generation_config=gen_config
        )
        response_text = response.text if hasattr(response, "text") else ""

        # Parse refined result
        evidence_items: List[EvidenceItem] = []
        summary = original_result.summary
        key_findings = original_result.key_findings
        recommended_actions = original_result.recommended_actions
        onscreen_text: List[str] = original_result.onscreen_text
        visual_summary: Optional[str] = original_result.visual_summary
        full_transcript: Optional[str] = original_result.full_transcript

        try:
            data = json.loads(response_text)
            summary = data.get("summary", summary)
            key_findings = data.get("key_findings", key_findings)
            recommended_actions = data.get("recommended_actions", recommended_actions)
            if data.get("onscreen_text"):
                onscreen_text = data["onscreen_text"]
            if data.get("visual_summary"):
                visual_summary = data["visual_summary"]
            if data.get("full_transcript"):
                full_transcript = data["full_transcript"]

            for raw in data.get("evidence_items", []):
                evidence_items.append(EvidenceItem(
                    description=str(raw.get("description", "")),
                    evidence_type=str(raw.get("evidence_type", "unknown")),
                    relevance_score=float(raw.get("relevance_score", 0.0)),
                    legal_significance=str(raw.get("legal_significance", "")),
                    location=raw.get("location"),
                    speaker_or_source=raw.get("speaker_or_source"),
                    supports_case=raw.get("supports_case"),
                    admissibility_concerns=raw.get("admissibility_concerns"),
                    verbatim_excerpt=raw.get("verbatim_excerpt"),
                ))
        except (json.JSONDecodeError, TypeError):
            # Keep original items if parsing fails
            evidence_items = original_result.evidence_items

        high_relevance = sum(1 for item in evidence_items if item.relevance_score >= 0.7)
        cost = len(response_text) // 4 * 0.001 / 1000

        return EvidenceAnalysisResult(
            situation_description=situation_description,
            files_analyzed=original_result.files_analyzed,
            evidence_items=evidence_items,
            full_transcript=full_transcript,
            visual_summary=visual_summary,
            onscreen_text=onscreen_text,
            summary=summary,
            key_findings=key_findings,
            recommended_actions=recommended_actions,
            total_evidence_count=len(evidence_items),
            high_relevance_count=high_relevance,
            model_used=self._DEFAULT_MODEL,
            cost=original_result.cost + cost,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

async def demo():
    """Demo the evidence analyzer (requires GOOGLE_API_KEY and sample files)."""
    import os

    print("=" * 60)
    print("EVIDENCE ANALYZER DEMO")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY"):
        print("\nTo run this demo set GOOGLE_API_KEY and provide sample files.")
        print("""
Usage example:

from src.evidence.evidence_analyzer import EvidenceAnalyzer
from configs.config import create_config
from configs.config import USState

config = create_config()
analyzer = EvidenceAnalyzer(config)

result = await analyzer.analyze_evidence(
    situation_description=(
        "A contract dispute between two companies. Company A claims Company B "
        "failed to deliver software on time, causing $200,000 in losses."
    ),
    file_paths=[
        "meeting_recording.mp3",
        "contract_draft.pdf",
        "email_thread.txt",
    ],
    jurisdiction=USState.CALIFORNIA,
)

result.print_report()
""")
        return

    print("\nProvide files to analyze or run with --evidence flag in main.py")


if __name__ == "__main__":
    asyncio.run(demo())

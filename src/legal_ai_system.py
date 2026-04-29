"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    RAPCorp LEGAL AI SYSTEM - MAIN ORCHESTRATOR                ║
║              Renaissance of American Physics and Astronomy (RAPCorp)          ║
║                                                                               ║
║  Master Orchestrator combining:                                               ║
║  • Gemini Models (flash-lite, flash, pro)                                     ║
║  • State Law Manager (customizable per state)                                 ║
║  • Legal Knowledge Swarm (15 parallel agents)                                 ║
║  • Reasoning Engine (IRAC/CREAC)                                              ║
║  • 3-Stage RAG Pipeline                                                       ║
║  • Control Plane (audit, UPL safeguards)                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
import json
import os

# Internal imports
from configs.config import (
    LegalAIConfig,
    create_config,
    USState,
    LegalDomain,
    GeminiModel
)
from src.core.gemini_client import GeminiClient
from src.state_laws.state_manager import StateLawManager
from src.swarm.knowledge_swarm import LegalKnowledgeSwarm
from src.reasoning.reasoning_engine import LegalReasoningEngine
from src.evidence.evidence_analyzer import EvidenceAnalyzer, EvidenceAnalysisResult
from src.documents.document_generator import DocumentGenerator, GeneratedDocument


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL PLANE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AuditEntry:
    """Entry in the audit trail."""
    timestamp: datetime
    action: str
    user_query: str
    response_summary: str
    agents_used: List[str]
    model_used: str
    cost: float
    state: str
    domain: str
    human_reviewed: bool = False
    review_notes: Optional[str] = None


class ControlPlane:
    """
    Control plane for legal AI governance.
    
    Provides:
    - Audit trail for all queries
    - UPL (Unauthorized Practice of Law) safeguards
    - Human-in-the-loop review triggers
    - Cost tracking and alerts
    """
    
    def __init__(self, config: LegalAIConfig):
        self.config = config
        self.audit_trail: List[AuditEntry] = []
        self._upl_disclaimer = (
            "\n\n---\n"
            "⚖️ **DISCLAIMER**: This information is for educational purposes only "
            "and does not constitute legal advice. Please consult with a licensed "
            "attorney in your jurisdiction for advice specific to your situation."
        )
    
    def add_audit_entry(self, entry: AuditEntry) -> None:
        """Add an entry to the audit trail."""
        self.audit_trail.append(entry)
    
    def get_audit_trail(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditEntry]:
        """Get audit trail, optionally filtered by date."""
        entries = self.audit_trail
        
        if start_date:
            entries = [e for e in entries if e.timestamp >= start_date]
        if end_date:
            entries = [e for e in entries if e.timestamp <= end_date]
        
        return entries
    
    def check_upl_triggers(self, query: str, response: str) -> List[str]:
        """
        Check for UPL (Unauthorized Practice of Law) triggers.
        
        Returns list of warnings if UPL concerns are detected.
        """
        warnings = []
        
        upl_keywords = [
            "you should sue",
            "file a lawsuit",
            "you will win",
            "guaranteed outcome",
            "legal advice:",
            "as your lawyer",
            "my legal opinion is"
        ]
        
        response_lower = response.lower()
        for keyword in upl_keywords:
            if keyword in response_lower:
                warnings.append(f"UPL trigger detected: '{keyword}'")
        
        return warnings
    
    def add_disclaimer(self, response: str) -> str:
        """Add the UPL disclaimer to a response."""
        if self.config.upl_safeguards:
            return response + self._upl_disclaimer
        return response
    
    def should_trigger_human_review(
        self,
        confidence: float,
        domain: LegalDomain
    ) -> bool:
        """Determine if human review should be triggered."""
        if not self.config.require_human_review:
            return False
        
        # Always review criminal matters
        if domain == LegalDomain.CRIMINAL:
            return True
        
        # Review low-confidence responses
        if confidence < self.config.human_review_threshold:
            return True
        
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 3-STAGE RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class LegalRAGPipeline:
    """
    3-Stage Legal RAG Pipeline.
    
    Stage 1: RETRIEVAL (The Researcher)
        - Uses FLASH model
        - Fetches raw documents
        - Returns structured passages
    
    Stage 2: ANALYSIS (The Associate)
        - Uses FLASH model
        - Identifies conflicts
        - Creates internal memo
    
    Stage 3: SYNTHESIS (The Partner)
        - Uses PRO model
        - Drafts final answer
        - Applies citations
    """
    
    def __init__(
        self,
        config: LegalAIConfig,
        llm_client: GeminiClient,
        swarm: LegalKnowledgeSwarm,
        reasoning: LegalReasoningEngine
    ):
        self.config = config
        self.llm = llm_client
        self.swarm = swarm
        self.reasoning = reasoning
    
    async def stage1_retrieve(
        self,
        query: str,
        state: USState,
        domain: LegalDomain
    ) -> Dict[str, Any]:
        """Stage 1: Retrieve relevant documents."""
        print("  📚 Stage 1: Retrieval (Flash)...")
        
        # Use swarm in quick mode for retrieval
        swarm_results = await self.swarm.research(
            query=query,
            context={"state": state, "domain": domain},
            mode="quick"
        )
        
        return {
            "documents": swarm_results.get("results", {}),
            "sources": [],  # Would be populated from vector DB
            "metadata": {
                "state": state.value,
                "domain": domain.value,
                "retrieval_time": swarm_results.get("duration_seconds", 0)
            }
        }
    
    async def stage2_analyze(
        self,
        query: str,
        retrieved: Dict[str, Any],
        state: USState
    ) -> Dict[str, Any]:
        """Stage 2: Analyze retrieved documents."""
        print("  🔍 Stage 2: Analysis (Flash)...")
        
        analysis_prompt = f"""
        Analyze these retrieved legal documents for the query:
        
        QUERY: {query}
        JURISDICTION: {state.value}
        
        RETRIEVED CONTENT:
        {json.dumps(retrieved["documents"], indent=2)[:3000]}
        
        Create an internal analysis memo identifying:
        1. Key legal principles found
        2. Relevant precedents
        3. Any conflicts between sources
        4. Gaps in the research
        5. Recommended additional research
        
        Return as JSON.
        """
        
        result = await self.llm.generate(
            prompt=analysis_prompt,
            task="summarization",
            return_json=True
        )
        
        return {
            "analysis_memo": result.get("text", "{}"),
            "model_used": result.get("model"),
            "cost": result.get("cost", 0)
        }
    
    async def stage3_synthesize(
        self,
        query: str,
        analysis: Dict[str, Any],
        state: USState,
        high_fidelity: bool = True
    ) -> Dict[str, Any]:
        """Stage 3: Synthesize final response."""
        print("  ✍️ Stage 3: Synthesis (Pro)...")
        
        fidelity_instruction = ""
        if high_fidelity:
            fidelity_instruction = """
            HIGH-FIDELITY MODE:
            - Only include information directly supported by the analysis
            - Cite specific sources for each claim
            - Clearly mark any uncertainty
            - Do not speculate beyond the evidence
            """
        
        synthesis_prompt = f"""
        Synthesize a comprehensive legal response based on this analysis:
        
        QUERY: {query}
        JURISDICTION: {state.value}
        
        ANALYSIS MEMO:
        {analysis.get("analysis_memo", "{}")}
        
        {fidelity_instruction}
        
        Provide:
        1. Direct answer to the query
        2. Key legal principles
        3. Relevant authorities with citations
        4. Any caveats or limitations
        5. Recommended next steps
        """
        
        result = await self.llm.generate(
            prompt=synthesis_prompt,
            task="legal_synthesis"
        )
        
        return {
            "response": result.get("text", ""),
            "model_used": result.get("model"),
            "cost": result.get("cost", 0),
            "high_fidelity": high_fidelity
        }
    
    async def run(
        self,
        query: str,
        state: USState = USState.FEDERAL,
        domain: LegalDomain = LegalDomain.CONTRACT,
        high_fidelity: bool = True
    ) -> Dict[str, Any]:
        """Run the full 3-stage pipeline."""
        start_time = datetime.utcnow()
        total_cost = 0.0
        
        # Stage 1
        retrieved = await self.stage1_retrieve(query, state, domain)
        
        # Stage 2
        analysis = await self.stage2_analyze(query, retrieved, state)
        total_cost += analysis.get("cost", 0)
        
        # Stage 3
        synthesis = await self.stage3_synthesize(query, analysis, state, high_fidelity)
        total_cost += synthesis.get("cost", 0)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "query": query,
            "state": state.value,
            "domain": domain.value,
            "response": synthesis.get("response", ""),
            "stages": {
                "retrieval": retrieved["metadata"],
                "analysis": {"model": analysis.get("model_used")},
                "synthesis": {"model": synthesis.get("model_used")}
            },
            "total_cost": total_cost,
            "duration_seconds": duration,
            "high_fidelity": high_fidelity
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CASE DIRECTORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseDirectoryAnalysis:
    """Result of an AI analysis of all documents in a case directory."""
    directory: str
    state: str
    documents_found: int
    documents_read: int

    case_overview: str                      # 2-3 sentence summary of the case
    parties: Dict[str, str]                 # {"petitioner": "...", "respondent": "..."}
    user_role: str                          # "petitioner" | "respondent" | "unknown"
    timeline: List[str]                     # Chronological list of key events/filings
    current_status: str                     # Where the case stands right now
    outstanding_issues: List[str]           # Open questions / unresolved matters
    recommendations: List[str]              # Concrete next steps
    urgency_items: List[str]                # Deadlines or time-sensitive items
    suggested_next_documents: List[Dict[str, str]]  # [{"type": "reply", "reason": "..."}]

    model_used: str
    cost: float


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreFlightIssue:
    """A single issue identified during pre-flight case review."""
    severity: str       # "critical", "moderate", "minor"
    category: str       # "missing_info", "jurisdiction", "claim_weakness", "contradiction", "missing_element"
    description: str    # What the issue is
    question: str       # The targeted clarifying question to ask


@dataclass
class PreFlightResult:
    """Result of an AI pre-flight check on a case situation."""
    situation_summary: str      # AI's understanding of the situation (2-3 sentences)
    has_issues: bool
    issues: List[PreFlightIssue]
    overall_confidence: float   # 0.0 - 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class LegalAIOrchestrator:
    """
    Master Orchestrator for the RAPCorp Legal AI System.
    
    Provides a unified interface to:
    - Research legal questions
    - Analyze contracts
    - Build arguments
    - Calculate damages
    - Track deadlines
    - Manage state-specific law
    """
    
    def __init__(self, config: Optional[LegalAIConfig] = None):
        """Initialize with optional configuration."""
        self.config = config
        self._initialized = False
        
        # Components (initialized in setup())
        self.llm: Optional[GeminiClient] = None
        self.state_manager: Optional[StateLawManager] = None
        self.swarm: Optional[LegalKnowledgeSwarm] = None
        self.reasoning: Optional[LegalReasoningEngine] = None
        self.control_plane: Optional[ControlPlane] = None
        self.pipeline: Optional[LegalRAGPipeline] = None
        self.evidence_analyzer: Optional[EvidenceAnalyzer] = None
        self.document_generator: Optional[DocumentGenerator] = None
    
    async def setup(self) -> "LegalAIOrchestrator":
        """
        Initialize all components.
        
        Returns self for chaining: system = await LegalAIOrchestrator().setup()
        """
        print("🚀 Initializing RAPCorp Legal AI System...")
        
        # Load configuration
        if not self.config:
            self.config = create_config()
        
        # Initialize LLM client
        print("  📡 Connecting to Gemini API...")
        try:
            self.llm = GeminiClient(self.config)
        except Exception as e:
            print(f"  ⚠️ LLM initialization failed: {e}")
            self.llm = None
        
        # Initialize state manager
        print("  🗺️ Loading state configurations...")
        self.state_manager = StateLawManager(self.config)
        self.state_manager.load_all_from_directory()
        print(f"     Loaded {len(self.state_manager.get_active_states())} active states")
        
        # Initialize CourtListener client
        courtlistener_client = None
        if self.config.courtlistener_api_key:
            print("  ⚖️ Connecting to CourtListener API...")
            try:
                from src.core.courtlistener_client import CourtListenerClient
                courtlistener_client = CourtListenerClient(self.config.courtlistener_api_key)
                print("     CourtListener: configured")
            except Exception as e:
                print(f"  ⚠️ CourtListener init failed: {e}")
        else:
            print("  ⚠️ CourtListener: no key set (add COURTLISTENER_API_KEY to .env for real case law)")

        # Initialize knowledge swarm
        print("  🐝 Initializing Knowledge Swarm...")
        self.swarm = LegalKnowledgeSwarm(self.config, self.llm, courtlistener_client)
        print(f"     {len(self.swarm.agents)} agents ready")
        
        # Initialize reasoning engine
        print("  🧠 Initializing Reasoning Engine...")
        self.reasoning = LegalReasoningEngine(self.config, self.llm)
        
        # Initialize control plane
        print("  🔒 Setting up Control Plane...")
        self.control_plane = ControlPlane(self.config)
        
        # Initialize RAG pipeline
        print("  🔄 Building RAG Pipeline...")
        self.pipeline = LegalRAGPipeline(
            self.config, self.llm, self.swarm, self.reasoning
        )

        # Initialize evidence analyzer
        print("  🔎 Initializing Evidence Analyzer...")
        try:
            self.evidence_analyzer = EvidenceAnalyzer(self.config)
        except Exception as e:
            print(f"  ⚠️ Evidence Analyzer initialization failed: {e}")
            self.evidence_analyzer = None

        # Initialize document generator
        print("  📄 Initializing Document Generator...")
        try:
            self.document_generator = DocumentGenerator(self.config)
        except Exception as e:
            print(f"  ⚠️ Document Generator initialization failed: {e}")
            self.document_generator = None

        self._initialized = True
        print("✅ System Ready!")
        print(f"   Models: {self.config.model_flash_lite}, {self.config.model_flash}, {self.config.model_pro}")
        
        return self
    
    def _check_initialized(self) -> None:
        """Ensure system is initialized."""
        if not self._initialized:
            raise RuntimeError("System not initialized. Call await system.setup() first.")
    
    # ─── Pre-Flight Check ───

    async def pre_flight_check(
        self,
        situation_description: str,
        state: USState,
        files: Optional[List[str]] = None,
    ) -> PreFlightResult:
        """
        Analyze a case situation before taking action.

        Identifies genuine issues that need clarification — missing critical
        info, jurisdictional problems, contradictions, weak claims — and
        generates a targeted question for each one.

        Returns immediately with no issues if the LLM is unavailable.
        """
        self._check_initialized()

        if not self.llm:
            return PreFlightResult(
                situation_summary=situation_description[:200],
                has_issues=False,
                issues=[],
                overall_confidence=1.0,
            )

        file_list_str = ""
        if files:
            file_list_str = f"\nFiles to be analyzed: {', '.join(files)}"

        prompt = f"""You are a senior legal intake specialist reviewing a client's case situation \
before proceeding with evidence analysis and document generation.

JURISDICTION: {state.value}{file_list_str}

SITUATION DESCRIPTION:
{situation_description}

Your task:
1. Summarize your understanding of the situation in 2-3 sentences.
2. Identify GENUINE issues that need clarification before proceeding:
   - Missing critical information (dates, parties, amounts, jurisdiction)
   - Potential jurisdictional or procedural problems
   - Ambiguous or contradictory statements
   - Weak or unsupported legal claims
   - Missing elements required for the stated cause of action
3. For each issue found, generate ONE specific, open-ended clarifying question.

Rules:
- Only flag REAL issues — do not manufacture problems if the situation is clear.
- If the situation is sufficiently clear to proceed, return an empty issues list.
- overall_confidence reflects how clearly you understand the case (0.0-1.0).

Return ONLY valid JSON:
{{
    "situation_summary": "2-3 sentence understanding of the situation",
    "overall_confidence": 0.85,
    "issues": [
        {{
            "severity": "critical|moderate|minor",
            "category": "missing_info|jurisdiction|claim_weakness|contradiction|missing_element",
            "description": "What the issue is, briefly",
            "question": "The specific open-ended clarifying question to ask"
        }}
    ]
}}"""

        try:
            result = await self.llm.generate(
                prompt=prompt,
                task="classification",
                return_json=True,
            )
            data = json.loads(result.get("text", "{}"))
            issues = [
                PreFlightIssue(
                    severity=i.get("severity", "minor"),
                    category=i.get("category", "missing_info"),
                    description=i.get("description", ""),
                    question=i.get("question", ""),
                )
                for i in data.get("issues", [])
                if i.get("question")
            ]
            return PreFlightResult(
                situation_summary=data.get("situation_summary", situation_description[:200]),
                has_issues=bool(issues),
                issues=issues,
                overall_confidence=float(data.get("overall_confidence", 0.8)),
            )
        except Exception:
            # Pre-flight failure must never block the user
            return PreFlightResult(
                situation_summary=situation_description[:200],
                has_issues=False,
                issues=[],
                overall_confidence=1.0,
            )

    # ─── Main Research Interface ───

    async def research(
        self,
        query: str,
        state: USState = USState.FEDERAL,
        domain: LegalDomain = LegalDomain.CONTRACT,
        mode: str = "standard",
        high_fidelity: bool = True
    ) -> Dict[str, Any]:
        """
        Research a legal question.
        
        Args:
            query: The legal question
            state: Jurisdiction
            domain: Legal domain
            mode: Research mode (quick/standard/comprehensive)
            high_fidelity: Use high-fidelity mode (strict citations)
        
        Returns:
            Research results with response and metadata
        """
        self._check_initialized()
        
        print(f"\n🔬 Researching: {query[:50]}...")
        print(f"   State: {state.value} | Domain: {domain.value} | Mode: {mode}")
        
        # Run the RAG pipeline
        result = await self.pipeline.run(
            query=query,
            state=state,
            domain=domain,
            high_fidelity=high_fidelity
        )
        
        # Add disclaimer
        result["response"] = self.control_plane.add_disclaimer(result["response"])
        
        # Check for UPL triggers
        upl_warnings = self.control_plane.check_upl_triggers(query, result["response"])
        if upl_warnings:
            result["upl_warnings"] = upl_warnings
        
        # Audit
        self.control_plane.add_audit_entry(AuditEntry(
            timestamp=datetime.utcnow(),
            action="research",
            user_query=query,
            response_summary=result["response"][:200] + "...",
            agents_used=[agent.value for agent in self.swarm.agents.keys()],
            model_used=result["stages"]["synthesis"]["model"] or "",
            cost=result["total_cost"],
            state=state.value,
            domain=domain.value
        ))
        
        return result
    
    # ─── IRAC Analysis ───
    
    async def analyze_irac(
        self,
        issue: str,
        facts: Dict[str, Any],
        state: USState = USState.FEDERAL
    ) -> Dict[str, Any]:
        """Perform IRAC analysis on a legal issue."""
        self._check_initialized()
        
        analysis = await self.reasoning.analyze_irac(
            issue=issue,
            facts=facts,
            jurisdiction=state
        )
        
        return {
            "issue": analysis.issue,
            "rule": analysis.rule,
            "application": analysis.application,
            "conclusion": analysis.conclusion,
            "confidence": analysis.confidence,
            "supporting_cases": analysis.supporting_cases,
            "counterarguments": analysis.counterarguments
        }
    
    # ─── State Law Helpers ───
    
    def get_sol(
        self,
        state: USState,
        claim_type: str,
        accrual_date: datetime
    ) -> Dict[str, Any]:
        """Get statute of limitations info."""
        self._check_initialized()
        
        calc = self.reasoning.calculate_sol(
            state=state,
            claim_type=claim_type,
            accrual_date=accrual_date
        )
        
        return {
            "state": calc.state.value,
            "claim_type": calc.claim_type,
            "limitation_years": calc.limitation_years,
            "deadline": calc.deadline.isoformat() if calc.deadline else None,
            "is_expired": calc.is_expired,
            "days_remaining": calc.days_remaining
        }
    
    def add_state(
        self,
        state: USState,
        state_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Add a new state configuration."""
        self._check_initialized()

        config = self.state_manager.add_state(state, state_name, **kwargs)
        return config.to_dict()

    # ─── Evidence Analysis ───

    async def analyze_evidence(
        self,
        situation_description: str,
        file_paths: List[str],
        state: USState = USState.FEDERAL,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze audio, video, or text files to identify evidence relevant
        to a case.

        Args:
            situation_description: Plain-language description of the case.
            file_paths: One or more file paths to analyze.
                Supported: .mp3/.wav/.aac/.ogg/.flac/.m4a/.aiff/.wma (audio),
                           .mp4/.avi/.mov/.mkv/.webm/.flv/.wmv/.mpeg/.3gp (video),
                           .txt/.md/.csv/.json/.xml/.html (text),
                           .pdf (document).
            state: Jurisdiction for contextual analysis.
            model_override: Override the default model used for analysis.

        Returns:
            Dictionary with evidence items, summary, key findings, and
            recommended actions.  Call result["_result_obj"].print_report()
            for a formatted console report.
        """
        self._check_initialized()

        if not self.evidence_analyzer:
            raise RuntimeError(
                "Evidence Analyzer not initialized. "
                "Ensure GOOGLE_API_KEY is set and google-generativeai is installed."
            )

        result: EvidenceAnalysisResult = await self.evidence_analyzer.analyze_evidence(
            situation_description=situation_description,
            file_paths=file_paths,
            jurisdiction=state,
            model_override=model_override,
        )

        output = result.to_dict()
        output["_result_obj"] = result   # Attach object for .print_report()
        return output

    async def generate_clarifying_questions(
        self,
        analysis_result: EvidenceAnalysisResult,
        situation_description: str,
    ) -> List[str]:
        """
        Generate targeted questions to resolve uncertainties in the analysis.
        Returns a list of question strings.
        """
        self._check_initialized()
        if not self.evidence_analyzer:
            return []
        return await self.evidence_analyzer.generate_clarifying_questions(
            result=analysis_result,
            situation_description=situation_description,
        )

    async def refine_evidence_analysis(
        self,
        original_result: EvidenceAnalysisResult,
        situation_description: str,
        clarifications: Dict[str, str],
        state: USState = USState.FEDERAL,
    ) -> Dict[str, Any]:
        """
        Refine a prior evidence analysis using user-provided clarifications.
        Does NOT re-upload files — text-only refinement pass.
        """
        self._check_initialized()
        if not self.evidence_analyzer:
            raise RuntimeError("Evidence Analyzer not initialized.")

        result = await self.evidence_analyzer.refine_with_clarifications(
            original_result=original_result,
            situation_description=situation_description,
            clarifications=clarifications,
            jurisdiction=state,
        )
        output = result.to_dict()
        output["_result_obj"] = result
        return output

    async def generate_petition_documents(
        self,
        situation: str,
        analysis: EvidenceAnalysisResult,
        clarifications: Dict[str, str],
        state: USState = USState.FEDERAL,
        output_dir: Optional[str] = None,
        doc_mode: str = "petition",
    ) -> List[Dict[str, Any]]:
        """
        Generate all documents required to file the petition.

        Returns a list of dicts describing each generated file.
        """
        self._check_initialized()
        if not self.document_generator:
            raise RuntimeError(
                "Document Generator not initialized. "
                "Ensure GOOGLE_API_KEY is set."
            )

        # Run swarm in quick mode to retrieve real case law from CourtListener,
        # then format it for injection into the document generation prompts.
        case_law_context = ""
        if self.swarm:
            print("  Retrieving relevant case law...")
            try:
                swarm_results = await self.swarm.research(
                    query=situation[:500],
                    context={"state": state, "domain": LegalDomain.CIVIL_RIGHTS},
                    mode="quick",
                )
                cl_data = swarm_results.get("results", {}).get("case_law_retriever", {})
                cases = cl_data.get("cases", [])
                api_status = cl_data.get("api_status", "not_configured")

                if cases:
                    lines = [f"[CourtListener — {len(cases)} cases retrieved]"]
                    for c in cases:
                        name = c.get("case_name", "Unknown")
                        citation = c.get("citation", "")
                        court = c.get("court", "")
                        date = c.get("date_filed", "")
                        snippet = (c.get("snippet") or "")[:200]
                        url = c.get("url", "")
                        lines.append(
                            f"\n• {name}"
                            + (f", {citation}" if citation else "")
                            + (f" ({court}" if court else "")
                            + (f", {date})" if date else (")" if court else ""))
                            + (f"\n  {snippet}" if snippet else "")
                            + (f"\n  {url}" if url else "")
                        )
                    case_law_context = "\n".join(lines)
                    print(f"    {len(cases)} cases from CourtListener (status: {api_status})")
                else:
                    print(f"    No cases retrieved (CourtListener status: {api_status})")
            except Exception as e:
                print(f"    Case law retrieval failed (non-fatal): {e}")

        docs: List[GeneratedDocument] = await self.document_generator.generate_petition_package(
            situation=situation,
            analysis=analysis,
            clarifications=clarifications,
            state=state,
            output_dir=output_dir,
            case_law_context=case_law_context,
            doc_mode=doc_mode,
        )

        return [
            {
                "title": d.title,
                "filename": d.filename,
                "file_path": d.file_path,
                "doc_type": d.doc_type,
                "description": d.description,
                "requires_signature": d.requires_signature,
                "requires_notarization": d.requires_notarization,
                "filing_required": d.filing_required,
            }
            for d in docs
        ]

    # ─── Case Directory Analysis ───

    async def analyze_case_directory(
        self,
        directory: str,
        state: USState = USState.FEDERAL,
        user_role_hint: Optional[str] = None,
        situation_update: Optional[str] = None,
    ) -> CaseDirectoryAnalysis:
        """
        Read all documents in a case directory and produce a structured
        analysis: case overview, timeline, current status, recommendations,
        and suggested follow-up documents.

        Args:
            directory:       Path to the folder containing all case documents.
            state:           Jurisdiction.
            user_role_hint:  "petitioner", "respondent", or None (AI will infer).

        Returns:
            CaseDirectoryAnalysis with full case picture and next-step guidance.
        """
        self._check_initialized()

        from src.documents.case_reader import CaseDirectoryScanner

        # ── Step 1: Scan and read all documents ────────────────────────────
        scanner = CaseDirectoryScanner(gemini_api_key=self.config.google_api_key)
        documents = scanner.scan(directory)
        stats = scanner.get_stats(documents)
        context_block = scanner.build_context_block(documents)

        if not self.llm:
            return CaseDirectoryAnalysis(
                directory=directory,
                state=state.value,
                documents_found=stats["total"],
                documents_read=stats["readable"],
                case_overview="LLM not available — cannot analyze documents.",
                parties={},
                user_role="unknown",
                timeline=[],
                current_status="Unknown",
                outstanding_issues=[],
                recommendations=[],
                urgency_items=[],
                suggested_next_documents=[],
                model_used="none",
                cost=0.0,
            )

        # ── Step 2: AI analysis ────────────────────────────────────────────
        role_hint = f"\nUser's role hint: {user_role_hint}" if user_role_hint else ""
        update_block = (
            f"\n\nNEW SITUATION UPDATE FROM CLIENT (written after the documents below):\n"
            f"{situation_update}\n"
            f"--- END OF UPDATE ---"
        ) if situation_update else ""

        prompt = f"""You are a senior litigation attorney conducting a complete review of
a client's case file. Analyze ALL of the documents below and produce a structured
legal assessment.

JURISDICTION: {state.value}{role_hint}{update_block}

DOCUMENT INDEX ({stats['readable']} readable documents):
{chr(10).join(f"  - {d.filename} [{d.doc_type}]" for d in documents if not d.read_error)}

FULL CASE FILE:
{context_block}

Produce your analysis as a JSON object with EXACTLY this structure:
{{
    "case_overview": "2-3 sentence plain-language summary of the entire case",
    "parties": {{
        "petitioner_plaintiff": "Name if known, else 'Unknown'",
        "respondent_defendant": "Name if known, else 'Unknown'",
        "other_parties": "Any other named parties, or null"
    }},
    "user_role": "petitioner|respondent|unknown",
    "timeline": [
        "YYYY-MM-DD (or 'Unknown date'): Event description",
        "..."
    ],
    "current_status": "One paragraph describing where the case stands right now — what has been filed, what is pending, what has been decided",
    "outstanding_issues": [
        "Specific unresolved legal or procedural issue",
        "..."
    ],
    "recommendations": [
        "Specific, actionable next step",
        "..."
    ],
    "urgency_items": [
        "Any deadline, response window, or time-sensitive item with date if known",
        "..."
    ],
    "suggested_next_documents": [
        {{
            "type": "reply|response|motion|opposition|appeal|supplement|amended_petition",
            "title": "Full title of the document to draft",
            "reason": "Why this document is needed based on the case file"
        }}
    ]
}}

Be thorough, specific, and reference actual document names and dates from the case file.
Return ONLY the JSON object — no preamble, no markdown fences."""

        import time as _time
        start = _time.time()
        result = await self.llm.generate(
            prompt=prompt,
            task="legal_synthesis",
            return_json=True,
        )
        elapsed = _time.time() - start

        # ── Step 3: Parse response ─────────────────────────────────────────
        data: Dict[str, Any] = {}
        try:
            data = json.loads(result.get("text", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass

        parties_raw = data.get("parties", {})
        parties = {k: v for k, v in parties_raw.items() if v}

        return CaseDirectoryAnalysis(
            directory=directory,
            state=state.value,
            documents_found=stats["total"],
            documents_read=stats["readable"],
            case_overview=data.get("case_overview", "Could not parse case overview."),
            parties=parties,
            user_role=data.get("user_role", "unknown"),
            timeline=data.get("timeline", []),
            current_status=data.get("current_status", "Unknown"),
            outstanding_issues=data.get("outstanding_issues", []),
            recommendations=data.get("recommendations", []),
            urgency_items=data.get("urgency_items", []),
            suggested_next_documents=data.get("suggested_next_documents", []),
            model_used=result.get("model", ""),
            cost=result.get("cost", 0.0),
        )

    async def generate_case_continuation(
        self,
        analysis: "CaseDirectoryAnalysis",
        document_type: str,
        additional_instructions: str = "",
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Draft a follow-up document based on the case directory analysis.

        Args:
            analysis:                CaseDirectoryAnalysis from analyze_case_directory().
            document_type:           "reply", "response", "motion", "opposition",
                                     "appeal", "supplement", "amended_petition"
            additional_instructions: Any extra user guidance for drafting.
            output_dir:              Override output directory.

        Returns:
            Dict with file metadata (same shape as generate_petition_documents items).
        """
        self._check_initialized()
        if not self.document_generator:
            raise RuntimeError("Document Generator not initialized.")

        from src.documents.case_reader import CaseDirectoryScanner
        scanner = CaseDirectoryScanner(gemini_api_key=self.config.google_api_key)
        documents = scanner.scan(analysis.directory)
        context_block = scanner.build_context_block(documents)

        doc: "GeneratedDocument" = await self.document_generator.generate_continuation_document(
            document_type=document_type,
            case_summary=analysis.case_overview,
            case_documents_context=context_block,
            recommendations=analysis.recommendations,
            state=USState(analysis.state) if analysis.state in [s.value for s in USState] else USState.FEDERAL,
            additional_instructions=additional_instructions,
            output_dir=output_dir,
        )

        return {
            "title": doc.title,
            "filename": doc.filename,
            "file_path": doc.file_path,
            "doc_type": doc.doc_type,
            "description": doc.description,
            "requires_signature": doc.requires_signature,
            "requires_notarization": doc.requires_notarization,
            "filing_required": doc.filing_required,
        }

    # ─── System Info ───

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status."""
        return {
            "initialized": self._initialized,
            "models": {
                "flash_lite": self.config.model_flash_lite,
                "flash": self.config.model_flash,
                "pro": self.config.model_pro
            },
            "active_states": [
                s.state.value for s in self.state_manager.get_active_states()
            ] if self.state_manager else [],
            "agents": len(self.swarm.agents) if self.swarm else 0,
            "cost_tracking": self.config.track_costs,
            "daily_budget": self.config.daily_budget_usd
        }
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        if self.llm:
            return self.llm.get_cost_summary()
        return {"error": "LLM not initialized"}


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def create_legal_ai_system() -> LegalAIOrchestrator:
    """Create and initialize a Legal AI system."""
    system = LegalAIOrchestrator()
    await system.setup()
    return system


# ═══════════════════════════════════════════════════════════════════════════════
# CLI DEMO
# ═══════════════════════════════════════════════════════════════════════════════

async def demo():
    """Demo the Legal AI System."""
    print("=" * 70)
    print("      RAPCorp LEGAL AI SYSTEM - DEMONSTRATION")
    print("      Renaissance of American Physics and Astronomy")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️ GOOGLE_API_KEY not set.")
        print("\nTo run the full demo, set your API key:")
        print("  export GOOGLE_API_KEY=your_key_here")
        print("\nShowing system capabilities without LLM...")
    
    # Initialize system
    system = await create_legal_ai_system()
    
    # Show system info
    print("\n📊 System Information:")
    info = system.get_system_info()
    print(f"  Models:")
    print(f"    • Flash Lite: {info['models']['flash_lite']}")
    print(f"    • Flash:      {info['models']['flash']}")
    print(f"    • Pro:        {info['models']['pro']}")
    print(f"  Active States: {', '.join(info['active_states'])}")
    print(f"  Agents: {info['agents']}")
    
    # Demo state management
    print("\n🗺️ State Law Management:")
    
    # Add Florida
    florida = system.add_state(
        USState.FLORIDA,
        "Florida",
        statute_of_limitations={
            "personal_injury": 4,
            "contract_written": 5,
            "medical_malpractice": 2
        },
        special_rules={
            "no_fault_auto": True,
            "homestead_protection": True
        }
    )
    print(f"  ➕ Added: Florida")
    print(f"     Personal Injury SOL: {florida['statute_of_limitations']['personal_injury']} years")
    
    # Demo SOL calculation
    print("\n⏱️ Statute of Limitations:")
    from datetime import datetime
    
    sol = system.get_sol(
        state=USState.CALIFORNIA,
        claim_type="personal_injury",
        accrual_date=datetime(2024, 1, 15)
    )
    print(f"  State: California")
    print(f"  Claim: Personal Injury")
    print(f"  Accrual: 2024-01-15")
    print(f"  Deadline: {sol['deadline'][:10] if sol['deadline'] else 'Unknown'}")
    status_msg = f"✅ {sol['days_remaining']} days remaining" if not sol['is_expired'] else "EXPIRED ❌"
    print(f"  Status: {status_msg}")
    
    # Demo research (if LLM available)
    if system.llm:
        print("\n🔬 Running Legal Research...")
        result = await system.research(
            query="What are the elements required to prove breach of contract?",
            state=USState.CALIFORNIA,
            domain=LegalDomain.CONTRACT,
            mode="quick"
        )
        
        print(f"\n📋 Research Result:")
        print(f"  Duration: {result['duration_seconds']:.2f}s")
        print(f"  Cost: ${result['total_cost']:.6f}")
        print(f"\n  Response Preview:")
        print(f"  {result['response'][:500]}...")
    else:
        print("\n💡 With GOOGLE_API_KEY set, you could run:")
        print("""
    result = await system.research(
        query="What are the elements of breach of contract?",
        state=USState.CALIFORNIA,
        domain=LegalDomain.CONTRACT
    )
    print(result["response"])
        """)
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())

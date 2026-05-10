"""
Legal Thinking Engine — chain-of-thought strategy synthesis with working memory.

Architecture
────────────
Each of the 10 thinking steps is a named "thought" stored in WorkingMemory.
Every thought explicitly references the prior thoughts it draws on.  The LLM
receives the full chain-so-far as context before generating each new thought,
so later thoughts genuinely build on earlier ones rather than being generated
independently.

Thinking steps (in order):
  1.  case_framing        — understand the core dispute, parties, and stakes
  2.  judge_profile       — what this type of judge cares about above all else
  3.  strategy_retrieval  — RAG: pull winning patterns for this case type
  4.  theory_selection    — best legal theories aligned with the judge's perspective
  5.  evidence_mapping    — map each theory to specific evidence in the record
  6.  weakness_scan       — identify our vulnerabilities before the opponent does
  7.  attack_preemption   — a judge-targeted response to every identified weakness
  8.  narrative_arc       — the compelling story that ties everything together
  9.  argument_hardening  — stress-test the argument; challenge its weakest links
  10. final_synthesis     — irrefutable, judge-optimized strategy plan

Output: LegalStrategyPlan — injected into every document generation prompt.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.thinking.working_memory import WorkingMemory, Thought
from src.thinking.legal_rag import LegalRAG
from src.thinking.opposition_agent import OppositionAgent, OppositionAnalysis


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LegalStrategyPlan:
    """Fully synthesized, judge-targeted legal strategy."""
    case_theory:            str
    judge_profile_summary:  str
    primary_argument:       str
    supporting_arguments:   List[str]
    counter_strategies:     List[str]
    their_weak_points:      List[str]
    narrative_arc:          str
    key_statutes:           List[str]
    key_cases:              List[str]
    prayer_for_relief:      str
    argument_score:         float
    refinement_cycles:      int
    thinking_log:           str = ""   # full chain-of-thought log
    opposition_analysis:    Optional["OppositionAnalysis"] = None

    def as_prompt_block(self) -> str:
        """Render as an injection block for document generation prompts."""
        lines = [
            "═══ LEGAL STRATEGY PLAN — FOLLOW THIS EXACTLY ═══",
            "",
            "CASE THEORY (the overarching narrative):",
            self.case_theory,
            "",
            "WHAT THE JUDGE CARES ABOUT MOST (tailor EVERY argument to this):",
            self.judge_profile_summary,
            "",
            "THE NARRATIVE (tell this story in the document):",
            self.narrative_arc,
            "",
            "PRIMARY ARGUMENT (lead every document with this):",
            self.primary_argument,
            "",
            "SUPPORTING ARGUMENTS (use ALL of these, in this order):",
        ]
        for i, arg in enumerate(self.supporting_arguments, 1):
            lines.append(f"  {i}. {arg}")
        lines += [
            "",
            "ATTACK PRE-EMPTIONS (address EACH before opposing counsel raises it):",
        ]
        for ctr in self.counter_strategies:
            lines.append(f"  • {ctr}")
        lines += [
            "",
            "THEIR WEAK POINTS (press these aggressively in argument sections):",
        ]
        for wp in self.their_weak_points:
            lines.append(f"  • {wp}")
        if self.key_statutes:
            lines += ["", "MANDATORY STATUTES (cite every one by exact section number):"]
            for s in self.key_statutes:
                lines.append(f"  — {s}")
        if self.key_cases:
            lines += ["", "MANDATORY CASE CITATIONS (cite every one by name and citation):"]
            for c in self.key_cases:
                lines.append(f"  — {c}")
        if self.prayer_for_relief:
            lines += ["", "PRAYER FOR RELIEF:", self.prayer_for_relief]
        if self.opposition_analysis:
            lines += ["", self.opposition_analysis.as_prompt_block()]
        lines += [
            "",
            f"[Strategy score: {self.argument_score:.1f}/10 | "
            f"{self.refinement_cycles} refinement cycles]",
            "═══════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# THINKING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_STEP_NAMES = [
    "case_framing",
    "judge_profile",
    "strategy_retrieval",
    "theory_selection",
    "evidence_mapping",
    "opposition_analysis",   # NEW — full 5-step opposition intelligence sub-chain
    "weakness_scan",
    "attack_preemption",
    "narrative_arc",
    "argument_hardening",
    "final_synthesis",
]

_MAX_REFINE_CYCLES = 2   # self-critique passes on the final synthesis


class LegalThinkingEngine:
    """
    Chain-of-thought legal strategy engine with working memory.

    Every thought is written to WorkingMemory and referenced by name in
    subsequent prompts — the LLM is never asked to reason in isolation.
    This creates genuine cumulative reasoning rather than parallel LLM calls.
    """

    def __init__(self, llm_client: Any):
        self._llm = llm_client
        self._rag = LegalRAG(llm_client)
        self._opposition = OppositionAgent(llm_client)

    async def think(
        self,
        situation:               str,
        evidence_summary:        str,
        case_law_context:        str,
        case_documents_context:  str,
        state:                   str,
        doc_mode:                str = "petition",
    ) -> LegalStrategyPlan:
        """
        Run the full 10-step thinking chain and return a LegalStrategyPlan.

        Each step stores its output in WorkingMemory and references prior steps
        so that the final synthesis is grounded in the entire reasoning chain.
        """
        mem = WorkingMemory()
        total_steps = len(_STEP_NAMES)

        # ─── Step 1: Case framing ────────────────────────────────────────
        print(f"  [Thinking 1/{total_steps}] Case framing…")
        await self._step_case_framing(
            mem, situation, evidence_summary, case_documents_context, state, doc_mode
        )

        # ─── Step 2: Judge profile ───────────────────────────────────────
        print(f"  [Thinking 2/{total_steps}] Judge profile…")
        await self._step_judge_profile(mem, situation, state)

        # ─── Step 3: RAG retrieval ───────────────────────────────────────
        print(f"  [Thinking 3/{total_steps}] Strategy retrieval (RAG)…")
        await self._step_strategy_retrieval(mem, situation)

        # ─── Step 4: Theory selection ────────────────────────────────────
        print(f"  [Thinking 4/{total_steps}] Legal theory selection…")
        await self._step_theory_selection(
            mem, situation, evidence_summary, case_law_context, state
        )

        # ─── Step 5: Evidence mapping ────────────────────────────────────
        print(f"  [Thinking 5/{total_steps}] Evidence mapping…")
        await self._step_evidence_mapping(mem, evidence_summary, case_documents_context)

        # ─── Step 6: Opposition intelligence ─────────────────────────────
        print(f"  [Thinking 6/{total_steps}] Opposition intelligence (5 sub-steps)…")
        opp_analysis = await self._opposition.analyze(
            situation        = situation,
            evidence_summary = evidence_summary,
            case_law_context = case_law_context,
            case_framing     = mem.get_content("case_framing"),
            judge_profile    = mem.get_content("judge_profile"),
        )
        # Store as a single thought so downstream steps can reference it
        mem.add("opposition_analysis", opp_analysis.as_prompt_block(),
                refs=["case_framing", "judge_profile", "evidence_mapping"])

        # ─── Step 7: Weakness scan ───────────────────────────────────────
        print(f"  [Thinking 7/{total_steps}] Weakness scan…")
        await self._step_weakness_scan(mem, situation)

        # ─── Step 8: Attack preemption ───────────────────────────────────
        print(f"  [Thinking 8/{total_steps}] Attack preemption…")
        await self._step_attack_preemption(mem)

        # ─── Step 9: Narrative arc ───────────────────────────────────────
        print(f"  [Thinking 9/{total_steps}] Narrative arc…")
        await self._step_narrative_arc(mem, situation)

        # ─── Step 10: Argument hardening ─────────────────────────────────
        print(f"  [Thinking 10/{total_steps}] Argument hardening…")
        score = await self._step_argument_hardening(mem)

        # ─── Step 11: Final synthesis ─────────────────────────────────────
        print(f"  [Thinking 11/{total_steps}] Final synthesis…")
        final_text = await self._step_final_synthesis(mem, situation, state, doc_mode)

        print(f"  [Thinking] Complete. Score: {score:.1f}/10")

        return self._build_plan(mem, final_text, score, opp_analysis)

    # ─────────────────────────────────────────────────────────────────────────
    # Individual step implementations
    # ─────────────────────────────────────────────────────────────────────────

    async def _step_case_framing(
        self, mem, situation, evidence_summary, case_docs, state, doc_mode
    ):
        action = "responding to an existing petition/motion" if doc_mode == "reply" \
                 else "initiating a new legal action"

        prompt = f"""You are a senior litigation strategist beginning your analysis of a new case.

JURISDICTION: {state}
MODE: {action}

SITUATION:
{situation[:2000]}

EVIDENCE AVAILABLE:
{evidence_summary[:1500]}

PRIOR CASE DOCUMENTS:
{case_docs[:1000]}

Think carefully and write your initial case framing:
1. What is the CORE dispute in one sentence?
2. Who are the parties and what does each want?
3. What is the single most important fact in our favor?
4. What is the single most dangerous fact against us?
5. What specific legal relief are we seeking?
6. What is our threshold for success (what outcome wins this case)?

Be specific and grounded in the actual facts above. Do not generalize."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("case_framing", (result.get("text") or "").strip())

    async def _step_judge_profile(self, mem, situation, state):
        ctx = mem.build_context(["case_framing"])

        prompt = f"""You are analyzing how the judge will approach this case.

WHAT YOU KNOW SO FAR:
{ctx}

JURISDICTION: {state}

Think about the judge who will hear this case and write your judge profile:
1. What TYPE of court and judge will hear this? (probate, civil, family, etc.)
2. What are the TOP 3 things this type of judge cares about MOST in making their decision?
3. What are the most common reasons judges DENY the relief our client seeks in cases like this?
4. What TONE and FRAMING resonates most with this type of judge?
5. What arguments would immediately IRRITATE or ALIENATE this judge?
6. What specific facts from our case will this judge find most compelling?
7. What evidence format does this court find most persuasive (documentary, testimonial, expert, statistical)?

Your analysis should be specific to the case type, not generic advice."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("judge_profile", (result.get("text") or "").strip(),
                refs=["case_framing"])

    async def _step_strategy_retrieval(self, mem, situation):
        retrieved = await self._rag.retrieve(
            situation=situation,
            case_type_hint=mem.get_content("case_framing")[:200],
        )
        mem.add("strategy_retrieval", retrieved, refs=["case_framing"])

    async def _step_theory_selection(self, mem, situation, evidence_summary,
                                     case_law_context, state):
        ctx = mem.build_context(["case_framing", "judge_profile", "strategy_retrieval"])

        prompt = f"""You are selecting the legal theories that will WIN this case.

FULL REASONING CHAIN SO FAR:
{ctx}

RELEVANT CASE LAW:
{case_law_context[:2000]}

JURISDICTION: {state}

Select the 3-5 strongest legal theories for this case.  For each:
1. NAME the theory (e.g. "Due process — deprivation of liberty without adequate proof")
2. STATE the specific statute or constitutional provision that supports it
3. NAME 1-2 cases that directly support it (from the case law above or your knowledge)
4. CONNECT it to a specific fact or piece of evidence we have
5. EXPLAIN why this judge (based on the judge profile above) will find it persuasive

ORDERING RULE: List theories in the order a judge would find them most persuasive —
most compelling first.  The first theory must be strong enough to win the case alone.

Only include theories where we have actual evidence support.
Do not include theories that require evidence we don't have."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("theory_selection", (result.get("text") or "").strip(),
                refs=["case_framing", "judge_profile", "strategy_retrieval"])

    async def _step_evidence_mapping(self, mem, evidence_summary, case_docs):
        ctx = mem.build_context(["case_framing", "theory_selection"])

        prompt = f"""You are mapping every piece of available evidence to the selected legal theories.

REASONING CHAIN:
{ctx}

EVIDENCE AVAILABLE:
{evidence_summary[:2000]}

PRIOR CASE DOCUMENTS:
{case_docs[:1000]}

For each legal theory identified above, list:
  THEORY: <theory name>
  SUPPORTING EVIDENCE:
    - <specific piece of evidence, exhibit, document, date, or fact that proves it>
    - <second piece>
  EVIDENCE GAPS:
    - <what evidence we WISH we had but don't>
  EXHIBIT LABEL: <what we will call this exhibit (Exhibit A, B, etc.)>

Then note: Are there any pieces of evidence that support MULTIPLE theories simultaneously?
These are your most powerful exhibits — identify them and plan to emphasize them.

Be specific. Reference actual documents, dates, and facts from the evidence summary above."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("evidence_mapping", (result.get("text") or "").strip(),
                refs=["case_framing", "theory_selection"])

    async def _step_weakness_scan(self, mem, situation):
        ctx = mem.build_context(["case_framing", "judge_profile", "theory_selection",
                                  "evidence_mapping", "opposition_analysis"])

        prompt = f"""You are a legal analyst conducting an objective review of our case position.

FULL REASONING CHAIN (our strategy + complete opposition analysis):
{ctx}

ORIGINAL SITUATION:
{situation[:1000]}

Identify the vulnerabilities in our position that the other side is likely to raise.
For each vulnerability:
  VULNERABILITY: <what it is>
  SIGNIFICANCE: <critical / significant / minor>
  HOW THEY WILL LIKELY RAISE IT: <how opposing counsel will present this at hearing>
  WHICH OPPOSITION STRENGTH RELATES: <connect to their strengths from the
    opposition analysis where applicable>
  WHICH OF OUR ARGUMENTS IS AFFECTED: <which legal theory this puts at risk>

Identify at minimum 5 vulnerabilities. Be thorough and objective.
Focus on what the judge will actually weigh, not just technical legal points.
Pay particular attention to vulnerabilities that align with the other side's strengths."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("weakness_scan", (result.get("text") or "").strip(),
                refs=["case_framing", "judge_profile", "theory_selection",
                      "evidence_mapping", "opposition_analysis"])

    async def _step_attack_preemption(self, mem):
        ctx = mem.build_context(["judge_profile", "strategy_retrieval", "theory_selection",
                                  "opposition_analysis", "weakness_scan"])

        prompt = f"""For every weakness identified, devise a specific, judge-targeted preemption.
Additionally, incorporate the attack surface mapped against the opposition — turn their
own weaknesses into offensive arguments we lead with rather than just defend against.

REASONING CHAIN (includes full opposition intelligence and weakness scan):
{ctx}

PART A — DEFENSIVE PREEMPTIONS
For each weakness we have, provide:
  WEAKNESS: <restate it>
  PREEMPTION STRATEGY: <exactly what we say/file/argue to neutralize it BEFORE they raise it>
  WHERE IN DOCUMENT: <which section — introduction, argument section, declaration, etc.>
  JUDGE APPEAL: <one sentence on why THIS judge will find our preemption persuasive>

The best preemption strategies:
  — Turn the weakness into a strength ("The very fact that X happened actually proves...")
  — Establish the legal standard that makes the weakness irrelevant
  — Produce evidence that directly contradicts the opposing narrative

PART B — OFFENSIVE INTEGRATION
From the opposition's attack surface, identify the top 3 attacks WE make against THEM
that belong in our opening documents (not just at trial). For each:
  OFFENSIVE ARGUMENT: <what we lead with to destroy their position>
  WHERE IT GOES: <which section of our petition/motion>
  WHY NOW: <why raising this proactively helps us with the judge>

Be specific to what THIS judge cares about, not generic legal advice."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("attack_preemption", (result.get("text") or "").strip(),
                refs=["judge_profile", "strategy_retrieval", "theory_selection",
                      "opposition_analysis", "weakness_scan"])

    async def _step_narrative_arc(self, mem, situation):
        ctx = mem.build_context(["case_framing", "judge_profile", "theory_selection",
                                  "evidence_mapping", "attack_preemption"])

        prompt = f"""Design the factual narrative — the coherent account of events the judge will follow.

FULL REASONING CHAIN:
{ctx}

SITUATION:
{situation[:800]}

Write a clear, measured narrative for this case. Tone must be professional and
restrained — factual assertions, not rhetoric or dramatic framing.

CASE OVERVIEW (2-3 sentences, neutral and factual, suitable for an introduction):
<a straightforward statement of what this proceeding is about and what is being sought>

THE FACTS IN ORDER (context → key event → what followed → what is at stake):
<the factual account, 200-300 words, written in plain professional language>

THE CORE LEGAL PRINCIPLE (the legal value at stake — due process, autonomy, proportionality):
<one sentence, stated as a legal proposition, not an emotional appeal>

THE STRONGEST FACTUAL POINT (the single fact most likely to persuade the judge):
<one specific, well-documented fact from the record — stated plainly>

This narrative must be 100% grounded in the actual facts of the case.
Every statement must be supportable by evidence in the record."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("narrative_arc", (result.get("text") or "").strip(),
                refs=["case_framing", "judge_profile", "theory_selection",
                      "evidence_mapping", "attack_preemption"])

    async def _step_argument_hardening(self, mem) -> float:
        """Self-critique pass. Returns a quality score 0-10."""
        ctx = mem.build_context()  # full chain

        prompt = f"""You are a skeptical appellate judge reviewing this legal strategy.

COMPLETE STRATEGY CHAIN:
{ctx[:5000]}

Rate each dimension 1-10 and provide ONE specific improvement for each:
  1. JUDGE ALIGNMENT     — Does the strategy actually appeal to what this judge cares about?
  2. FACTUAL GROUNDING   — Is every claim supported by specific evidence in the record?
  3. LEGAL AUTHORITY     — Does every argument cite a specific statute or case?
  4. NARRATIVE COHERENCE — Does the story make sense and flow naturally?
  5. PREEMPTION COVERAGE — Does the strategy anticipate and neutralize all major attacks?
  6. PROPORTIONALITY     — Is the relief sought proportional to the harm proven?

Format each as:
  DIMENSION: <name>
  SCORE: <1-10>
  WEAKNESS: <specific gap or problem>
  FIX: <specific one-sentence improvement>

Then provide:
  OVERALL SCORE: <weighted average, 1-10>
  SINGLE BIGGEST WEAKNESS: <the one thing that could lose this case>
  SINGLE BIGGEST STRENGTH: <the one thing most likely to win it>"""

        result = await self._llm.generate(
            prompt=prompt, task="classification", return_json=False
        )
        text = (result.get("text") or "").strip()
        mem.add("argument_hardening", text,
                refs=list(mem._order))  # references everything

        # Extract score
        import re
        m = re.search(r'OVERALL SCORE[:\s]+([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)
        return float(m.group(1)) if m else 7.0

    async def _step_final_synthesis(self, mem, situation, state, doc_mode) -> str:
        ctx = mem.build_context()  # full chain, trimmed

        action = "responding to a petition/motion" if doc_mode == "reply" \
                 else "initiating a legal action"

        prompt = f"""Synthesize the complete legal strategy.
You have completed a full 9-step reasoning chain. Now produce the final strategy plan
that will be injected into every court document.

COMPLETE REASONING CHAIN (everything you have thought through):
{ctx[:6000]}

CASE: {situation[:300]}
JURISDICTION: {state}
MODE: {action}

Produce the FINAL STRATEGY PLAN with these sections:

CASE THEORY:
<2-3 sentences: the overarching legal narrative>

PRIMARY ARGUMENT:
<The single strongest argument — include the specific statute/case and evidence hook>

SUPPORTING ARGUMENTS:
1. <argument + statute/case + evidence>
2. <argument + statute/case + evidence>
3. <argument + statute/case + evidence>
[add up to 5 total]

ATTACK PREEMPTIONS:
- <weakness>: <our specific preemptive response with judge appeal>
[one per weakness identified]

THEIR WEAK POINTS:
- <their vulnerability + how we press it>
[list all identified]

KEY STATUTES:
- <exact section number and name>

KEY CASES:
- <Case Name, citation, one-sentence relevance>

PRAYER FOR RELIEF:
<specific, numbered relief items>

Every item must be grounded in the reasoning chain above.
Nothing generic. Nothing unsupported by the actual facts of this case."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        text = (result.get("text") or "").strip()
        mem.add("final_synthesis", text, refs=list(mem._order))
        return text

    # ─────────────────────────────────────────────────────────────────────────
    # Output builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_plan(
        self, mem: WorkingMemory, final_text: str, score: float,
        opp_analysis: Optional[OppositionAnalysis] = None,
    ) -> LegalStrategyPlan:
        """Parse the final synthesis into a structured LegalStrategyPlan."""
        import re

        def _extract(text: str, header: str) -> str:
            pattern = re.compile(
                rf'{re.escape(header)}[:\s]*\n(.*?)(?=\n[A-Z][A-Z\s]{{4,}}:|$)',
                re.DOTALL | re.IGNORECASE,
            )
            m = pattern.search(text)
            return m.group(1).strip() if m else ""

        def _extract_list(text: str, header: str) -> List[str]:
            block = _extract(text, header)
            if not block:
                return []
            items = []
            for line in block.split("\n"):
                line = line.strip().lstrip("•–-–*1234567890. ")
                if line:
                    items.append(line)
            return items

        case_theory   = _extract(final_text, "CASE THEORY")
        primary_arg   = _extract(final_text, "PRIMARY ARGUMENT")
        narrative     = mem.get_content("narrative_arc")
        judge_summary = mem.get_content("judge_profile")[:400]
        prayer        = _extract(final_text, "PRAYER FOR RELIEF")

        support_args  = _extract_list(final_text, "SUPPORTING ARGUMENTS")
        preemptions   = _extract_list(final_text, "ATTACK PREEMPTIONS")
        weak_points   = _extract_list(final_text, "THEIR WEAK POINTS")
        key_statutes  = _extract_list(final_text, "KEY STATUTES")
        key_cases     = _extract_list(final_text, "KEY CASES")

        # Fallback when parsing fails
        if not case_theory:
            case_theory = final_text[:500]
        if not primary_arg:
            primary_arg = final_text[:800]

        return LegalStrategyPlan(
            case_theory=case_theory[:1000],
            judge_profile_summary=judge_summary,
            primary_argument=primary_arg[:1500],
            supporting_arguments=support_args[:8],
            counter_strategies=preemptions[:8],
            their_weak_points=weak_points[:6],
            narrative_arc=narrative[:1200],
            key_statutes=key_statutes[:15],
            key_cases=key_cases[:15],
            prayer_for_relief=prayer[:600],
            argument_score=score,
            refinement_cycles=1,
            thinking_log=mem.build_full_log(),
            opposition_analysis=opp_analysis,
        )

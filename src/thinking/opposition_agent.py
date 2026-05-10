"""
Opposition Intelligence Agent — 5-step analysis of the opposing party.

Runs BEFORE the main weakness_scan / attack_preemption steps so the engine
has a complete, structured picture of the opposition before forming its own
defensive and offensive strategy.

Step sequence (own internal WorkingMemory):
  1. claim_mapping      — catalog every claim they are making, by category
  2. evidence_catalog   — assess every piece of their evidence for strength
                          and exploitability
  3. strength_analysis  — their top strengths: impact on judge, our best counter
  4. weakness_analysis  — their most exploitable gaps: priority-ranked
  5. attack_surface     — concrete, judge-optimized attack vectors against them

Output: OppositionAnalysis dataclass with an as_prompt_block() method that
injects the full opposition picture into subsequent thinking steps and
document generation prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

from src.thinking.working_memory import WorkingMemory


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OppositionAnalysis:
    """Structured intelligence report on the opposing party."""
    organized_claims:   str            # claim map by category
    evidence_catalog:   str            # their evidence assessed
    top_strengths:      List[str]      # their strongest points
    top_weaknesses:     List[str]      # their most exploitable gaps
    attack_surface:     str            # prioritized attack plan against them
    threat_level:       str = "MEDIUM" # CRITICAL / HIGH / MEDIUM / LOW
    thinking_log:       str = ""       # full 5-step chain for saving

    def as_prompt_block(self) -> str:
        lines = [
            "╔═══ OPPOSITION INTELLIGENCE REPORT ═══╗",
            f"Threat Level: {self.threat_level}",
            "",
            "── THEIR ORGANIZED CLAIMS ──",
            self.organized_claims,
            "",
            "── THEIR EVIDENCE (assessed) ──",
            self.evidence_catalog,
            "",
            "── THEIR BIGGEST STRENGTHS ──",
        ]
        for i, s in enumerate(self.top_strengths, 1):
            lines.append(f"  {i}. {s}")
        lines += [
            "",
            "── THEIR MOST EXPLOITABLE WEAKNESSES ──",
        ]
        for i, w in enumerate(self.top_weaknesses, 1):
            lines.append(f"  {i}. {w}")
        lines += [
            "",
            "── ATTACK SURFACE (prioritized) ──",
            self.attack_surface,
            "╚═══════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class OppositionAgent:
    """
    Builds a complete intelligence picture of the opposing party before the
    main strategy engine forms its defensive and offensive positions.

    Each of the 5 steps stores a named thought in its own WorkingMemory and
    references prior steps — the same cumulative-reasoning pattern used by
    LegalThinkingEngine.
    """

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def analyze(
        self,
        situation:        str,
        evidence_summary: str,
        case_law_context: str,
        case_framing:     str,   # from the main engine's step 1
        judge_profile:    str,   # from the main engine's step 2
    ) -> OppositionAnalysis:
        mem = WorkingMemory()

        print("    [Opposition 1/5] Claim mapping…")
        await self._step_claim_mapping(mem, situation, evidence_summary, case_framing)

        print("    [Opposition 2/5] Evidence catalog…")
        await self._step_evidence_catalog(mem, evidence_summary)

        print("    [Opposition 3/5] Strength analysis…")
        await self._step_strength_analysis(mem, judge_profile)

        print("    [Opposition 4/5] Weakness analysis…")
        await self._step_weakness_analysis(mem, judge_profile)

        print("    [Opposition 5/5] Attack surface…")
        await self._step_attack_surface(mem, case_law_context, judge_profile)

        return self._build_analysis(mem)

    # ─────────────────────────────────────────────────────────────────────────
    # Steps
    # ─────────────────────────────────────────────────────────────────────────

    async def _step_claim_mapping(self, mem, situation, evidence_summary, case_framing):
        prompt = f"""You are an opposition research analyst. Your job is to fully map
and organize every claim the opposing party is making or is likely to make.

CASE FRAMING (what we already know about this dispute):
{case_framing[:2000]}

SITUATION:
{situation[:2000]}

EVIDENCE AVAILABLE (look for what supports their side, not ours):
{evidence_summary[:1500]}

Organize the opposing party's claims into these four categories.
For each claim, rate: Evidence Support (1-10) | Legal Merit (1-10) | Judge Appeal (1-10).

CATEGORY 1 — FACTUAL CLAIMS
(What facts do they assert happened, or did not happen?)
  CLAIM: <state the claim>
  EVIDENCE SUPPORT: <score> — <what evidence backs this>
  LEGAL MERIT: <score>
  JUDGE APPEAL: <score>
  [repeat for each factual claim]

CATEGORY 2 — LEGAL CLAIMS
(What legal violations or entitlements do they assert?)
  CLAIM: <state the legal claim, cite statute/case if relevant>
  EVIDENCE SUPPORT: <score>
  LEGAL MERIT: <score>
  JUDGE APPEAL: <score>
  [repeat]

CATEGORY 3 — PROCEDURAL CLAIMS
(Do they claim any procedural violations, standing issues, timing problems?)
  CLAIM: <state the procedural claim>
  EVIDENCE SUPPORT: <score>
  LEGAL MERIT: <score>
  JUDGE APPEAL: <score>
  [repeat, or state NONE if no procedural claims]

CATEGORY 4 — CHARACTER / CREDIBILITY CLAIMS
(Do they make any claims about our client's credibility, motives, or character?)
  CLAIM: <state the claim>
  EVIDENCE SUPPORT: <score>
  LEGAL MERIT: <score>
  JUDGE APPEAL: <score>
  [repeat, or state NONE]

End with:
OVERALL THREAT ASSESSMENT: <CRITICAL / HIGH / MEDIUM / LOW>
REASON: <one sentence on why you gave this rating>

Be specific and grounded in the actual facts above. Do not fabricate evidence."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("claim_mapping", (result.get("text") or "").strip())

    async def _step_evidence_catalog(self, mem, evidence_summary):
        ctx = mem.build_context(["claim_mapping"])

        prompt = f"""You are cataloging and critically assessing the opposing party's evidence.

CLAIM MAP (from prior analysis):
{ctx}

FULL EVIDENCE SUMMARY:
{evidence_summary[:2000]}

For each piece of evidence that supports or could be used by the opposing party:

EVIDENCE ITEM: <name/description of the evidence>
  TYPE: <documentary / testimonial / expert / circumstantial>
  WHAT IT PROVES FOR THEM: <specifically what argument it supports>
  STRENGTH RATING: <1-10>  (10 = irrefutable, 1 = easily dismissed)
  CREDIBILITY RATING: <1-10>  (10 = unimpeachable source, 1 = highly suspect)
  CHALLENGE METHOD: <exactly how we attack this — foundation, hearsay, bias,
                     methodology, completeness, authenticity, or weight>
  CHALLENGE DIFFICULTY: <easy / moderate / hard>
  NET THREAT: <high / medium / low>  (strength × credibility, discounted by challenge difficulty)

After all items, write:
EVIDENCE SUMMARY:
  Strongest piece overall: <name>
  Most easily attacked: <name>
  Evidence gap (what they WISH they had but don't): <describe>"""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("evidence_catalog", (result.get("text") or "").strip(),
                refs=["claim_mapping"])

    async def _step_strength_analysis(self, mem, judge_profile):
        ctx = mem.build_context(["claim_mapping", "evidence_catalog"])

        prompt = f"""You are identifying the opposing party's most dangerous strengths.

PRIOR ANALYSIS:
{ctx}

JUDGE PROFILE (what THIS judge cares about most):
{judge_profile[:1500]}

Identify the opposing party's TOP 3-5 STRENGTHS — the points where they are
on their strongest ground and where we are most at risk.

For each strength:

STRENGTH #<n>: <one-line title>
  WHY IT IS STRONG: <specifically why this argument/evidence is powerful>
  JUDGE IMPACT: <how much will THIS judge be moved by this — and why>
  SEVERITY: <CRITICAL / HIGH / MEDIUM>
  CAN WE NEUTRALIZE IT? <yes/partially/no>
  OUR BEST COUNTER: <the single strongest response we have, if any>
  IF WE CANNOT NEUTRALIZE: <what is the damage — does it cost us the whole case
                             or just one argument?>

Order from most dangerous to least dangerous.
Be honest — if a strength is truly fatal without a counter, say so."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("strength_analysis", (result.get("text") or "").strip(),
                refs=["claim_mapping", "evidence_catalog"])

    async def _step_weakness_analysis(self, mem, judge_profile):
        ctx = mem.build_context(["claim_mapping", "evidence_catalog", "strength_analysis"])

        prompt = f"""You are identifying the opposing party's most exploitable weaknesses.

PRIOR ANALYSIS:
{ctx}

JUDGE PROFILE:
{judge_profile[:1200]}

Identify the opposing party's TOP 3-5 WEAKNESSES — the gaps, contradictions,
overreaches, and vulnerabilities in their position that we can exploit.

For each weakness:

WEAKNESS #<n>: <one-line title>
  WHAT MAKES IT A WEAKNESS: <specifically why this is a vulnerability for them>
  OUR EVIDENCE TO EXPLOIT IT: <specific document, testimony, date, or fact we have>
  JUDGE IMPACT: <will THIS judge care about this weakness — and why>
  EXPLOITABILITY: <HIGH / MEDIUM / LOW>
  HOW TO EXPLOIT IT: <the specific argument, motion, or cross-examination tactic>
  TIMING: <when in the case to strike — opening, motion phase, at trial, closing>

After all weaknesses:
PRIORITY RANKING: List weakness numbers from highest to lowest exploitability.
COMPOUND ATTACKS: Are there any two weaknesses that, combined, create a
                  devastating argument? Describe the combination."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("weakness_analysis", (result.get("text") or "").strip(),
                refs=["claim_mapping", "evidence_catalog", "strength_analysis"])

    async def _step_attack_surface(self, mem, case_law_context, judge_profile):
        ctx = mem.build_context()  # full 4-step chain

        prompt = f"""You are mapping the complete attack surface against the opposing party.

FULL OPPOSITION INTELLIGENCE:
{ctx}

RELEVANT CASE LAW:
{case_law_context[:1500]}

JUDGE PROFILE:
{judge_profile[:1000]}

Design a prioritized attack plan. This is the offensive strategy — what WE
do to them, not what they might do to us.

PRIMARY STRIKE (the single most devastating attack):
  ATTACK: <what it is>
  WHY IT IS THE BEST ATTACK: <specifically why this is the highest-value target>
  EXECUTION: <exactly how we make this argument in the document/at hearing>
  CASE LAW SUPPORT: <statute or case that supports this attack>
  EXPECTED RESPONSE: <how they will try to counter this>
  OUR COUNTER-COUNTER: <how we shut down their response>

SECONDARY ATTACKS (3-4 supporting attacks, in priority order):
  ATTACK #1: <title>
    TARGET: <which claim or evidence of theirs does this destroy>
    EXECUTION: <how>
    TIMING: <when in proceedings>

  ATTACK #2: <title>
    [same format]

  ATTACK #3: <title>
    [same format]

PROCEDURAL ATTACKS (motions, evidentiary challenges, etc.):
  <list specific motions, objections, or procedural moves we should file/make>

CREDIBILITY ATTACKS (on their witnesses or experts):
  <if applicable: specific impeachment points for each key opposing witness>

ATTACK SEQUENCING:
  <In what order do we deploy these attacks across the lifecycle of the case?
   Opening → Motion Phase → Discovery → Trial/Hearing → Closing.
   Note: even if this is pre-trial document work, sequence for a full proceeding.>"""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        mem.add("attack_surface", (result.get("text") or "").strip(),
                refs=["claim_mapping", "evidence_catalog", "strength_analysis",
                      "weakness_analysis"])

    # ─────────────────────────────────────────────────────────────────────────
    # Build output
    # ─────────────────────────────────────────────────────────────────────────

    def _build_analysis(self, mem: WorkingMemory) -> OppositionAnalysis:
        claim_text     = mem.get_content("claim_mapping")
        evidence_text  = mem.get_content("evidence_catalog")
        strength_text  = mem.get_content("strength_analysis")
        weakness_text  = mem.get_content("weakness_analysis")
        attack_text    = mem.get_content("attack_surface")

        # Extract threat level from claim mapping
        threat_level = "MEDIUM"
        m = re.search(
            r'OVERALL THREAT ASSESSMENT[:\s]+(CRITICAL|HIGH|MEDIUM|LOW)',
            claim_text, re.IGNORECASE
        )
        if m:
            threat_level = m.group(1).upper()

        # Parse strengths and weaknesses into bullet lists for the dataclass
        top_strengths = self._extract_numbered_items(strength_text, "STRENGTH")
        top_weaknesses = self._extract_numbered_items(weakness_text, "WEAKNESS")

        thinking_log = mem.build_full_log()

        return OppositionAnalysis(
            organized_claims  = claim_text,
            evidence_catalog  = evidence_text,
            top_strengths     = top_strengths,
            top_weaknesses    = top_weaknesses,
            attack_surface    = attack_text,
            threat_level      = threat_level,
            thinking_log      = thinking_log,
        )

    @staticmethod
    def _extract_numbered_items(text: str, prefix: str) -> List[str]:
        """Pull 'STRENGTH #1: title\\n  WHY...' blocks into a flat list."""
        pattern = re.compile(
            rf'{prefix}\s*#?\d+[:\s]+(.+?)(?=\n\s*{prefix}\s*#?\d+|\Z)',
            re.IGNORECASE | re.DOTALL,
        )
        items = []
        for m in pattern.finditer(text):
            block = m.group(1).strip()
            # Use just the first line as the title + first sub-point
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            summary = lines[0] if lines else block[:120]
            if len(lines) > 1:
                summary += " — " + lines[1]
            items.append(summary[:300])
        # Fallback: return the whole text as one item if regex found nothing
        if not items:
            items = [text[:500]]
        return items

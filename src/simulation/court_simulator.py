"""
Court Simulator — AI-powered moot court that stress-tests your case before you
step into a real courtroom.

Architecture: multi-agent LLM simulation where each agent role-plays a specific
courtroom participant.  The simulation runs a full procedural sequence and
produces a verdict with detailed analysis of argument performance.

Agents:
  • JudgeAgent         — manages procedure, rules on objections, delivers verdict
  • OpposingCounselAgent — adversarial; presses every weakness in our case
  • OurCounselAgent    — argues our strategy plan (pro se or represented)

Procedure (bench trial, adaptable to jury):
  1. PRELIMINARY       — motions in limine, introductions
  2. OPENING_OUR       — our opening statement
  3. OPENING_OPPOSING  — opposing opening statement
  4. ARGUMENT_OUR      — our main argument / direct examination
  5. CROSS_OPPOSING    — opposing cross-examination / counter-argument
  6. REBUTTAL_OUR      — our rebuttal
  7. CLOSING_OUR       — our closing argument
  8. CLOSING_OPPOSING  — opposing closing argument
  9. DELIBERATION      — judge deliberates (produces tentative findings)
 10. VERDICT           — final ruling with reasoning
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from configs.config import USState


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TranscriptEntry:
    stage:   str
    speaker: str          # "JUDGE" | "OUR_COUNSEL" | "OPPOSING_COUNSEL" | "WITNESS"
    text:    str
    objection: Optional[str] = None   # grounds if an objection was raised
    ruling:    Optional[str] = None   # judge's ruling on objection


@dataclass
class ObjectionRecord:
    stage:    str
    raised_by: str
    grounds:   str
    ruling:    str         # "SUSTAINED" | "OVERRULED"
    impact:    str


@dataclass
class ArgumentScore:
    """How well each argument performed in simulation."""
    argument_label: str
    effectiveness: float       # 0.0 – 1.0
    judge_reception: str       # "favorable" | "neutral" | "skeptical"
    opposing_rebuttal_strength: float
    notes: str


@dataclass
class CourtSimulationResult:
    """Complete outcome of a simulated court proceeding."""
    case_summary:    str
    state:           str
    trial_type:      str
    verdict:         str           # "FOR_PETITIONER" | "FOR_RESPONDENT" | "MIXED" | "INCONCLUSIVE"
    verdict_reasoning: str
    our_confidence:  float         # predicted probability of success (0.0–1.0)

    transcript:      List[TranscriptEntry] = field(default_factory=list)
    objections:      List[ObjectionRecord] = field(default_factory=list)
    argument_scores: List[ArgumentScore]   = field(default_factory=list)

    strongest_moments: List[str] = field(default_factory=list)
    weakest_moments:   List[str] = field(default_factory=list)
    recommended_improvements: List[str] = field(default_factory=list)

    duration_seconds: float = 0.0

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("  COURT SIMULATION RESULT")
        print("=" * 70)
        print(f"  Verdict      : {self.verdict}")
        print(f"  Win chance   : {self.our_confidence * 100:.0f}%")
        print(f"  Reasoning    : {self.verdict_reasoning[:300]}")
        if self.strongest_moments:
            print("\n  STRONGEST MOMENTS:")
            for m in self.strongest_moments:
                print(f"    ✓ {m}")
        if self.weakest_moments:
            print("\n  WEAKEST MOMENTS:")
            for m in self.weakest_moments:
                print(f"    ✗ {m}")
        if self.recommended_improvements:
            print("\n  IMPROVEMENTS BEFORE TRIAL:")
            for r in self.recommended_improvements:
                print(f"    → {r}")
        print("=" * 70)

    def transcript_text(self) -> str:
        lines = [f"COURT SIMULATION TRANSCRIPT — {self.state}", "=" * 70, ""]
        for entry in self.transcript:
            speaker_label = entry.speaker.replace("_", " ").upper()
            lines.append(f"[{entry.stage}] {speaker_label}:")
            lines.append(entry.text)
            if entry.objection:
                lines.append(f"  OBJECTION ({entry.objection})")
            if entry.ruling:
                lines.append(f"  COURT: {entry.ruling}")
            lines.append("")
        lines += [
            "=" * 70,
            f"VERDICT: {self.verdict}",
            "",
            self.verdict_reasoning,
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# COURT SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

_PROCEDURE_STAGES = [
    "PRELIMINARY",
    "OPENING_OUR",
    "OPENING_OPPOSING",
    "ARGUMENT_OUR",
    "CROSS_OPPOSING",
    "REBUTTAL_OUR",
    "CLOSING_OUR",
    "CLOSING_OPPOSING",
    "DELIBERATION",
    "VERDICT",
]


class CourtSimulator:
    """
    Run a full AI-powered moot court simulation.

    Usage:
        simulator = CourtSimulator(llm_client)
        result = await simulator.simulate(
            situation=situation,
            our_strategy=strategy_plan.as_prompt_block(),
            case_law_context=case_law_context,
            state=USState.NEBRASKA,
        )
        result.print_summary()
    """

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def simulate(
        self,
        situation: str,
        our_strategy: str,
        case_law_context: str,
        state: USState,
        trial_type: str = "bench",          # "bench" | "jury"
        opposing_known_arguments: str = "",
        progress_callback: Optional[Any] = None,
    ) -> CourtSimulationResult:
        """
        Run the full simulation and return a CourtSimulationResult.

        Args:
            situation:                Plain-language description of the case.
            our_strategy:             Output of LegalStrategyPlan.as_prompt_block().
            case_law_context:         CourtListener case law block.
            state:                    Jurisdiction.
            trial_type:               "bench" or "jury".
            opposing_known_arguments: Opposing party's documented arguments (optional).
            progress_callback:        Called with (stage_name, speaker, text) each step.
        """
        start = datetime.utcnow()

        transcript:      List[TranscriptEntry] = []
        objections:      List[ObjectionRecord] = []
        argument_scores: List[ArgumentScore]   = []

        # Build shared context that all agents receive
        shared_ctx = self._build_shared_context(
            situation, our_strategy, case_law_context,
            state, trial_type, opposing_known_arguments
        )

        # Running judge's memory — accumulates key rulings and impressions
        judge_notes: List[str] = []

        # ── Stage 1: Preliminary ──────────────────────────────────────────
        if progress_callback:
            await self._notify(progress_callback, "PRELIMINARY", "COURT", "Setting up...")
        prelim = await self._run_preliminary(shared_ctx, state)
        transcript.extend(prelim)
        for e in prelim:
            if e.ruling:
                judge_notes.append(f"Preliminary: {e.ruling}")

        # ── Stage 2 & 3: Opening statements ──────────────────────────────
        our_opening = await self._run_opening(shared_ctx, side="our")
        transcript.append(TranscriptEntry("OPENING_OUR", "OUR_COUNSEL", our_opening))
        if progress_callback:
            await self._notify(progress_callback, "OPENING_OUR", "OUR_COUNSEL", our_opening[:300])

        opp_opening = await self._run_opening(shared_ctx, side="opposing")
        transcript.append(TranscriptEntry("OPENING_OPPOSING", "OPPOSING_COUNSEL", opp_opening))
        if progress_callback:
            await self._notify(progress_callback, "OPENING_OPPOSING", "OPPOSING_COUNSEL", opp_opening[:300])

        judge_notes.append(f"Our opening theme: {our_opening[:200]}")
        judge_notes.append(f"Their opening theme: {opp_opening[:200]}")

        # ── Stage 4 & 5: Argument / Cross ────────────────────────────────
        our_arg, our_objs = await self._run_argument(shared_ctx, side="our", judge_notes=judge_notes)
        transcript.append(TranscriptEntry("ARGUMENT_OUR", "OUR_COUNSEL", our_arg))
        objections.extend(our_objs)

        cross, cross_objs = await self._run_argument(shared_ctx, side="opposing", judge_notes=judge_notes)
        transcript.append(TranscriptEntry("CROSS_OPPOSING", "OPPOSING_COUNSEL", cross))
        objections.extend(cross_objs)

        # Judge rules on objections
        for obj in our_objs + cross_objs:
            ruling = await self._rule_on_objection(shared_ctx, obj)
            obj.ruling = ruling

        # Score our main argument
        arg_score = await self._score_argument(shared_ctx, our_arg, cross)
        argument_scores.append(arg_score)
        judge_notes.append(f"Main argument score: {arg_score.effectiveness:.1f} — {arg_score.notes[:150]}")

        # ── Stage 6: Rebuttal ─────────────────────────────────────────────
        rebuttal = await self._run_rebuttal(shared_ctx, our_arg=our_arg, opp_cross=cross)
        transcript.append(TranscriptEntry("REBUTTAL_OUR", "OUR_COUNSEL", rebuttal))

        # ── Stages 7 & 8: Closing arguments ──────────────────────────────
        our_close = await self._run_closing(shared_ctx, side="our", judge_notes=judge_notes)
        transcript.append(TranscriptEntry("CLOSING_OUR", "OUR_COUNSEL", our_close))

        opp_close = await self._run_closing(shared_ctx, side="opposing", judge_notes=judge_notes)
        transcript.append(TranscriptEntry("CLOSING_OPPOSING", "OPPOSING_COUNSEL", opp_close))

        # ── Stage 9 & 10: Deliberation + Verdict ─────────────────────────
        verdict_data = await self._deliberate_and_rule(
            shared_ctx, judge_notes=judge_notes,
            our_close=our_close, opp_close=opp_close,
        )

        transcript.append(TranscriptEntry("VERDICT", "JUDGE", verdict_data.get("ruling_text", "")))

        # ── Post-simulation analysis ──────────────────────────────────────
        analysis = await self._post_analysis(shared_ctx, transcript, argument_scores)

        duration = (datetime.utcnow() - start).total_seconds()

        return CourtSimulationResult(
            case_summary=situation[:300],
            state=state.value,
            trial_type=trial_type,
            verdict=verdict_data.get("verdict", "INCONCLUSIVE"),
            verdict_reasoning=verdict_data.get("reasoning", ""),
            our_confidence=verdict_data.get("our_win_probability", 0.5),
            transcript=transcript,
            objections=objections,
            argument_scores=argument_scores,
            strongest_moments=analysis.get("strongest_moments", []),
            weakest_moments=analysis.get("weakest_moments", []),
            recommended_improvements=analysis.get("recommended_improvements", []),
            duration_seconds=duration,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Agent implementations
    # ─────────────────────────────────────────────────────────────────────

    def _build_shared_context(
        self,
        situation: str,
        our_strategy: str,
        case_law_context: str,
        state: USState,
        trial_type: str,
        opposing_args: str,
    ) -> str:
        return f"""
JURISDICTION: {state.value}
TRIAL TYPE: {trial_type} trial

CASE SITUATION:
{situation[:1500]}

OUR LEGAL STRATEGY:
{our_strategy[:2000]}

RELEVANT CASE LAW:
{case_law_context[:1500]}

OPPOSING PARTY'S KNOWN ARGUMENTS:
{opposing_args[:1000] if opposing_args else "Unknown — simulate most likely opposition."}
""".strip()

    async def _run_preliminary(
        self, ctx: str, state: USState
    ) -> List[TranscriptEntry]:
        prompt = f"""You are a {state.value} judge opening a hearing.
Read the case context below and:
1. Call the case to order
2. Note any preliminary motions that either side would likely raise
3. Rule on each motion in one sentence

Case context:
{ctx[:1500]}

Write the preliminary proceeding as a transcript. Format:
COURT: <judge's words>
OUR_COUNSEL: <any motion we should raise>
OPPOSING_COUNSEL: <any motion they would raise>
COURT: <ruling>

Keep it to 300 words. Plain text only."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        text = (result.get("text") or "").strip()

        entries = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("COURT:"):
                entries.append(TranscriptEntry("PRELIMINARY", "JUDGE", line[6:].strip()))
            elif line.startswith("OUR_COUNSEL:"):
                entries.append(TranscriptEntry("PRELIMINARY", "OUR_COUNSEL", line[12:].strip()))
            elif line.startswith("OPPOSING_COUNSEL:"):
                entries.append(TranscriptEntry("PRELIMINARY", "OPPOSING_COUNSEL", line[17:].strip()))
        if not entries:
            entries.append(TranscriptEntry("PRELIMINARY", "JUDGE", text[:500]))
        return entries

    async def _run_opening(self, ctx: str, side: str) -> str:
        if side == "our":
            role = "pro se petitioner (or counsel) who has prepared meticulously"
            instr = "Deliver a powerful opening statement that leads with our strongest argument, names specific facts and dates, and previews our key evidence. Be confident and persuasive."
        else:
            role = "opposing counsel who is aggressive and analytically precise"
            instr = "Deliver an opening statement attacking the petitioner's case on every front. Identify their weakest facts, challenge their legal theories, and preview your defense arguments."

        prompt = f"""You are the {role} in this case.
{instr}

CASE CONTEXT:
{ctx[:2000]}

Write a 400-word opening statement. Plain text, no stage directions, no brackets.
Speak directly as if addressing the judge."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        return (result.get("text") or "").strip()

    async def _run_argument(
        self, ctx: str, side: str, judge_notes: List[str]
    ):
        judge_memory = "\n".join(judge_notes[-5:]) if judge_notes else ""

        if side == "our":
            role  = "our counsel / pro se petitioner"
            instr = "Present your main legal arguments in the strongest possible order. Cite specific statutes by section number and case law by name. Respond to the judge's preliminary impressions."
        else:
            role  = "opposing counsel"
            instr = "Cross-examine our position. Attack every factual claim, challenge every legal theory, and raise your strongest defenses. Be surgical and aggressive."

        prompt = f"""You are {role}.
{instr}

CASE CONTEXT:
{ctx[:2000]}

JUDGE'S CURRENT IMPRESSIONS:
{judge_memory}

Write your argument (500 words). Also identify 1-2 objections you would raise during the other side's argument (format: OBJECTION: <grounds> after your main argument text).
Plain text only."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        text = (result.get("text") or "").strip()

        # Parse out any objections
        objs: List[ObjectionRecord] = []
        clean_lines = []
        for line in text.split("\n"):
            if line.strip().upper().startswith("OBJECTION:"):
                grounds = line.strip()[10:].strip()
                objs.append(ObjectionRecord(
                    stage="ARGUMENT",
                    raised_by=side.upper(),
                    grounds=grounds,
                    ruling="",
                    impact="",
                ))
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines).strip(), objs

    async def _rule_on_objection(self, ctx: str, obj: ObjectionRecord) -> str:
        prompt = f"""You are a {ctx.split('JURISDICTION:')[1].split()[0] if 'JURISDICTION:' in ctx else 'state'} judge.
Rule on this objection in one sentence: SUSTAINED or OVERRULED, with brief reasoning.

Grounds: {obj.grounds}
Raised by: {obj.raised_by}
Context summary: {ctx[:500]}

Reply: SUSTAINED/OVERRULED — <one-sentence reason>"""

        result = await self._llm.generate(prompt=prompt, task="classification")
        ruling = (result.get("text") or "OVERRULED — no grounds stated.").strip()
        # Set impact based on ruling
        obj.impact = "Excludes damaging evidence" if "SUSTAINED" in ruling.upper() else "Argument proceeds"
        return ruling

    async def _score_argument(self, ctx: str, our_arg: str, opp_cross: str) -> ArgumentScore:
        prompt = f"""You are a neutral appellate judge evaluating the quality of legal arguments.

OUR ARGUMENT:
{our_arg[:1500]}

OPPOSING CROSS-EXAMINATION:
{opp_cross[:1000]}

Rate the effectiveness of OUR argument on a 0.0-1.0 scale.
Return JSON:
{{
  "effectiveness": 0.0-1.0,
  "judge_reception": "favorable|neutral|skeptical",
  "opposing_rebuttal_strength": 0.0-1.0,
  "notes": "One sentence on how this argument played"
}}"""

        result = await self._llm.generate(prompt=prompt, task="classification", return_json=True)
        try:
            data = json.loads(result.get("text", "{}"))
            return ArgumentScore(
                argument_label="Main Argument",
                effectiveness=float(data.get("effectiveness", 0.5)),
                judge_reception=data.get("judge_reception", "neutral"),
                opposing_rebuttal_strength=float(data.get("opposing_rebuttal_strength", 0.5)),
                notes=data.get("notes", ""),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return ArgumentScore("Main Argument", 0.5, "neutral", 0.5, "Score unavailable.")

    async def _run_rebuttal(self, ctx: str, our_arg: str, opp_cross: str) -> str:
        prompt = f"""You are our counsel delivering a rebuttal.
Directly counter the specific points raised in opposing counsel's cross-examination.
Be precise and cite facts / statutes. Maximum 300 words.

OUR ORIGINAL ARGUMENT:
{our_arg[:800]}

OPPOSING CROSS:
{opp_cross[:800]}

FULL CASE CONTEXT:
{ctx[:1000]}

Deliver the rebuttal now. Plain text only."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        return (result.get("text") or "").strip()

    async def _run_closing(self, ctx: str, side: str, judge_notes: List[str]) -> str:
        judge_memory = "\n".join(judge_notes[-6:]) if judge_notes else ""

        if side == "our":
            role  = "our counsel / pro se petitioner"
            instr = "Deliver a powerful closing argument. Tie every fact to the law, hammer our strongest points, and undercut their defense. Ask for specific relief."
        else:
            role  = "opposing counsel"
            instr = "Deliver your closing argument. Emphasize every failure of proof on their side, strengthen your defense, and argue for a ruling in your client's favor."

        prompt = f"""You are {role}.
{instr}

CASE CONTEXT:
{ctx[:1500]}

JUDGE'S IMPRESSIONS DURING TRIAL:
{judge_memory}

Write a 400-word closing argument. Plain text only. No stage directions."""

        result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
        return (result.get("text") or "").strip()

    async def _deliberate_and_rule(
        self,
        ctx: str,
        judge_notes: List[str],
        our_close: str,
        opp_close: str,
    ) -> Dict[str, Any]:
        judge_memory = "\n".join(judge_notes) if judge_notes else ""

        prompt = f"""You are the presiding judge in this case.
You have heard all arguments. Now deliberate and issue your ruling.

FULL CASE CONTEXT:
{ctx[:2000]}

YOUR NOTES FROM THE HEARING:
{judge_memory[:1500]}

OUR CLOSING:
{our_close[:800]}

OPPOSING CLOSING:
{opp_close[:800]}

Issue your ruling as JSON:
{{
  "verdict": "FOR_PETITIONER | FOR_RESPONDENT | MIXED | INCONCLUSIVE",
  "our_win_probability": 0.0-1.0,
  "ruling_text": "Your full written ruling (400 words) — address each main argument and state your findings of fact and law",
  "reasoning": "One paragraph summary of the controlling reason for your ruling"
}}

Be realistic. If our case is strong, say so. If there are fatal flaws, identify them.
Return ONLY valid JSON."""

        result = await self._llm.generate(
            prompt=prompt, task="legal_synthesis", return_json=True
        )
        try:
            return json.loads(result.get("text", "{}"))
        except (json.JSONDecodeError, TypeError):
            return {
                "verdict": "INCONCLUSIVE",
                "our_win_probability": 0.5,
                "ruling_text": result.get("text", "Ruling unavailable."),
                "reasoning": "Parse error — see ruling_text.",
            }

    async def _post_analysis(
        self,
        ctx: str,
        transcript: List[TranscriptEntry],
        scores: List[ArgumentScore],
    ) -> Dict[str, Any]:
        transcript_snippet = "\n".join(
            f"[{e.stage}] {e.speaker}: {e.text[:200]}"
            for e in transcript[:20]
        )
        score_info = "\n".join(
            f"  {s.argument_label}: {s.effectiveness:.1f} ({s.judge_reception})"
            for s in scores
        )

        prompt = f"""Analyze this court simulation and identify concrete improvement areas.

TRANSCRIPT EXCERPT:
{transcript_snippet}

ARGUMENT SCORES:
{score_info}

Return JSON:
{{
  "strongest_moments": ["Top 3 moments where our argument was most effective"],
  "weakest_moments": ["Top 3 moments where we were most vulnerable"],
  "recommended_improvements": ["Top 5 specific, actionable things to fix before the real trial"]
}}

Return ONLY valid JSON."""

        result = await self._llm.generate(
            prompt=prompt, task="classification", return_json=True
        )
        try:
            return json.loads(result.get("text", "{}"))
        except (json.JSONDecodeError, TypeError):
            return {
                "strongest_moments": [],
                "weakest_moments": [],
                "recommended_improvements": [],
            }

    @staticmethod
    async def _notify(callback: Any, stage: str, speaker: str, text: str) -> None:
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(stage, speaker, text)
            else:
                callback(stage, speaker, text)
        except Exception:
            pass

"""
Quick smoke-test for the thinking module.

Runs the full 10-step chain on a realistic guardianship dispute scenario
and prints each thought plus the final strategy plan.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import LegalAIConfig
from src.core.gemini_client import GeminiClient
from src.thinking.legal_thinking_engine import LegalThinkingEngine
from src.thinking.working_memory import WorkingMemory
from src.thinking.legal_rag import LegalRAG

SITUATION = """
My elderly mother (age 79) has been placed under a full guardianship by my brother
without my knowledge. The petition was filed in California probate court in March 2025.
My brother, who controls her finances, hired a private doctor to declare her
"completely incapacitated" despite her primary care physician of 20 years stating
she retains capacity to make personal decisions. The court-appointed investigator's
report was never provided to me prior to the hearing. My brother has a history of
financial disputes with our mother and was removed from her trust in 2023. The
guardianship was granted ex parte (emergency basis) even though no emergency existed.
I am seeking to terminate the guardianship or convert it to a limited conservatorship,
and to hold my brother accountable for exploitation of a vulnerable adult.
"""

EVIDENCE_SUMMARY = """
- Letter from Dr. Patricia Liu (PCP, 20 yrs) dated Feb 2025: states mother retains
  capacity for personal decisions and social activities; disputes "complete incapacity"
- Court records showing no notice was sent to petitioner (me) prior to hearing
- Trust amendment from 2023 removing brother as beneficiary due to financial dispute
- Text messages from brother to mother demanding money in Jan-Feb 2025
- Financial records showing $47,000 withdrawn from mother's account in 6 months
  prior to guardianship petition
- Mother's own statement (video recorded Mar 2025): "I don't want this, I want to
  go home. He is taking my money."
- Dr. Harold Reyes (hired by brother): evaluation lasted 45 minutes, used outdated
  assessment tool; no independent corroboration
- Court investigator's report: obtained after the fact via subpoena; notes mother
  was "confused during interview" but does not establish total incapacity
"""

CASE_LAW = """
Conservatorship of Wendland (2001) 26 Cal.4th 519 — California Supreme Court:
conservatorship requires proof by clear and convincing evidence; court must consider
least restrictive alternative.

Probate Code § 1801 — conservatorship of person may be established only if the
proposed conservatee is unable to provide for personal needs.

Probate Code § 1828.5 — proposed conservatee has right to notice and to appear.

In re Marriage of Flaherty (1982) 31 Cal.3d 637 — sanctions available for abuse
of the court process.

Welfare & Institutions Code § 15610.30 — financial abuse of elder: taking property
for wrongful use with intent to defraud.
"""

async def main():
    config = LegalAIConfig()
    llm = GeminiClient(config)

    print("=" * 70)
    print("LEGAL THINKING ENGINE — SMOKE TEST")
    print("=" * 70)
    print(f"Scenario: California guardianship termination / elder financial abuse")
    print()

    # ── Test WorkingMemory directly ──────────────────────────────────────────
    print("[ WorkingMemory unit check ]")
    mem = WorkingMemory()
    # Simulate all 10 engine steps with realistic names and refs
    steps = [
        ("case_framing",       [],                                        "Core dispute: guardianship contested by sibling."),
        ("judge_profile",      ["case_framing"],                          "Probate judge priorities: least restrictive alternative, due process."),
        ("strategy_retrieval", ["case_framing"],                          "RAG patterns: challenge medical eval, establish financial motive."),
        ("theory_selection",   ["case_framing","judge_profile","strategy_retrieval"], "Primary theory: void order due to lack of notice + no emergency."),
        ("evidence_mapping",   ["theory_selection"],                      "Dr. Liu letter contradicts Dr. Reyes. Trust removal shows motive."),
        ("weakness_scan",      ["theory_selection","evidence_mapping"],   "Weakness: mother appeared confused in investigator report."),
        ("attack_preemption",  ["weakness_scan","judge_profile"],         "Counter: investigator interviewed her during medication change."),
        ("narrative_arc",      ["case_framing","judge_profile","attack_preemption"], "Story: devoted daughter, controlling brother, voiceless mother."),
        ("argument_hardening", ["narrative_arc","attack_preemption"],     "Score 8.5/10. Weakest link: damages evidence thin; add financial records."),
        ("final_synthesis",    ["case_framing","judge_profile","strategy_retrieval",
                                "theory_selection","evidence_mapping","weakness_scan",
                                "attack_preemption","narrative_arc","argument_hardening"],
                               "Irrefutable plan: void order, limited conservatorship, WIC 15610 claim."),
    ]
    for name, refs, content in steps:
        mem.add(name, content, refs=refs)

    assert len(mem) == 10, f"Expected 10 thoughts, got {len(mem)}"

    # build_context with no args returns all 10
    full_ctx = mem.build_context()
    for name, _, _ in steps:
        assert name in full_ctx, f"build_context missing thought: {name}"

    # build_context with specific names returns only those
    partial_ctx = mem.build_context(["case_framing", "judge_profile", "final_synthesis"])
    assert "case_framing" in partial_ctx
    assert "final_synthesis" in partial_ctx
    # Content of unrequested thoughts must not appear (name may appear in refs header)
    evidence_content = mem.get_content("evidence_mapping")
    assert evidence_content not in partial_ctx, "partial context leaked content of unrequested thought"

    # refs are recorded correctly
    t = mem.get("final_synthesis")
    assert len(t.refs) == 9, f"final_synthesis should ref all 9 prior steps, got {len(t.refs)}"

    # build_full_log includes all steps in order
    log = mem.build_full_log()
    assert "Step 01" in log and "Step 10" in log, "full log missing step numbers"

    print(f"  WorkingMemory: OK — {len(mem)} thoughts, full/partial context, refs, log all verified")

    # ── Test LegalRAG directly ───────────────────────────────────────────────
    print("[ LegalRAG retrieval check ]")
    rag = LegalRAG(llm)
    rag_result = await rag.retrieve(SITUATION, case_type_hint="guardianship", top_k=10)
    assert len(rag_result) > 100, "RAG returned too little content"
    print(f"  LegalRAG: OK — {len(rag_result)} chars returned")
    print("  RAG excerpt:", rag_result[:300].replace("\n", " "))
    print()

    # ── Run full thinking engine ─────────────────────────────────────────────
    print("[ Full 10-step thinking chain ]")
    print("-" * 70)
    engine = LegalThinkingEngine(llm)

    plan = await engine.think(
        situation=SITUATION,
        evidence_summary=EVIDENCE_SUMMARY,
        case_law_context=CASE_LAW,
        case_documents_context="",
        state="California",
        doc_mode="petition",
    )

    # Print thinking log step by step
    print(plan.thinking_log)
    print()

    # ── Validate plan fields ─────────────────────────────────────────────────
    print("=" * 70)
    print("STRATEGY PLAN VALIDATION")
    print("=" * 70)

    checks = {
        "case_theory":           (len(plan.case_theory) > 50,        plan.case_theory[:120]),
        "judge_profile_summary": (len(plan.judge_profile_summary) > 50, plan.judge_profile_summary[:120]),
        "primary_argument":      (len(plan.primary_argument) > 50,   plan.primary_argument[:120]),
        "supporting_arguments":  (len(plan.supporting_arguments) > 0, str(plan.supporting_arguments[:2])),
        "counter_strategies":    (len(plan.counter_strategies) > 0,  str(plan.counter_strategies[:2])),
        "their_weak_points":     (len(plan.their_weak_points) > 0,   str(plan.their_weak_points[:2])),
        "narrative_arc":         (len(plan.narrative_arc) > 50,      plan.narrative_arc[:120]),
        "prayer_for_relief":     (len(plan.prayer_for_relief) > 20,  plan.prayer_for_relief[:120]),
        "argument_score":        (0 < plan.argument_score <= 10,     str(plan.argument_score)),
    }

    all_pass = True
    for field, (ok, preview) in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {field}")
        print(f"         {preview}")
        print()

    print("-" * 70)
    print(f"Argument quality score: {plan.argument_score}/10")
    print(f"Refinement cycles:      {plan.refinement_cycles}")
    print(f"Thinking log length:    {len(plan.thinking_log)} chars")
    print()

    # ── Print as_prompt_block (what gets injected into docs) ─────────────────
    print("[ as_prompt_block() preview — first 800 chars ]")
    block = plan.as_prompt_block()
    print(block[:800])
    print("..." if len(block) > 800 else "")
    print()

    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — review output above")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

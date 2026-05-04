"""
Quick end-to-end test: generate a Reply/Response doc and check for case citations.
Run from the project root:  python scripts/test_response_cases.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import create_config, USState, LegalDomain

SITUATION = (
    "I am responding to a Petition for Appointment of Guardian and Conservator "
    "filed against me in Nebraska. The petitioner claims I lack capacity to manage "
    "my own affairs, but I am fully competent. I have no history of mental illness "
    "and have been managing my finances independently for decades. The petition was "
    "filed by a family member with whom I have had a financial dispute."
)

async def main():
    from src.legal_ai_system import create_legal_ai_system
    from src.evidence.evidence_analyzer import EvidenceAnalysisResult, EvidenceItem

    print("=" * 60)
    print("  TEST: Response doc — court case citation check")
    print("=" * 60)

    system = await create_legal_ai_system()

    # ── Step 1: Check CourtListener directly ─────────────────────────
    print("\n[1] CourtListener direct check...")
    from src.core.courtlistener_client import CourtListenerClient
    cl = CourtListenerClient(system.config.courtlistener_api_key)
    result = await cl.search_opinions("guardianship competency objection", jurisdiction="NE", page_size=5)
    print(f"    API available : {result.api_available}")
    print(f"    Cases found   : {result.total_count}")
    if result.error:
        print(f"    ERROR         : {result.error}")
    for c in result.cases[:3]:
        print(f"    • {c.case_name} ({c.date_filed}) — {c.citation}")

    # ── Step 2: Run the swarm ─────────────────────────────────────────
    print("\n[2] Running swarm (quick mode)...")
    swarm_results = await system.swarm.research(
        query=SITUATION[:500],
        context={"state": USState.NEBRASKA, "domain": LegalDomain.CIVIL_RIGHTS},
        mode="quick",
    )
    errors = swarm_results.get("errors", [])
    if errors:
        print(f"    SWARM ERRORS: {errors}")
    cl_data = swarm_results.get("results", {}).get("case_law_retriever", {})
    cases = cl_data.get("cases", [])
    api_status = cl_data.get("api_status", "not_configured")
    queries_fired = cl_data.get("queries_fired", "?")
    print(f"    api_status    : {api_status}")
    print(f"    queries_fired : {queries_fired}")
    print(f"    cases returned: {len(cases)}")
    for c in cases[:5]:
        print(f"    • {c.get('case_name')} ({c.get('date_filed')}) — {c.get('citation')}")

    if not cases:
        print("\n    ABORT: no cases from swarm — cannot test citation in document.")
        return

    # ── Step 3: Format case_law_context exactly as the orchestrator does ─
    lines = [f"[CourtListener — {len(cases)} cases retrieved via {queries_fired} queries]"]
    for c in cases:
        name     = c.get("case_name", "Unknown")
        citation = c.get("citation", "")
        court    = c.get("court", "")
        date     = c.get("date_filed", "")
        snippet  = (c.get("snippet") or "")[:500]
        url      = c.get("url", "")
        lines.append(
            f"\n• {name}"
            + (f", {citation}" if citation else "")
            + (f" ({court}" if court else "")
            + (f", {date})" if date else (")" if court else ""))
            + (f"\n  {snippet}" if snippet else "")
            + (f"\n  {url}" if url else "")
        )
    case_law_context = "\n".join(lines)
    print(f"\n[3] case_law_context length: {len(case_law_context)} chars")
    print("    First 400 chars:")
    print("   ", case_law_context[:400].replace("\n", "\n    "))

    # ── Step 4: Build a minimal stub analysis ────────────────────────
    print("\n[4] Building stub EvidenceAnalysisResult...")
    stub_analysis = EvidenceAnalysisResult(
        situation_description=SITUATION,
        files_analyzed=[],
        summary=SITUATION,
        key_findings=[
            "Petitioner lacks evidence of incapacity",
            "Subject has managed finances independently for decades",
            "Family financial dispute is likely motive",
        ],
        evidence_items=[],
        recommended_actions=["File objection to petition"],
    )

    # ── Step 5: Generate just the reply document ──────────────────────
    print("\n[5] Generating reply/response document...")
    from src.documents.document_generator import DocumentGenerator
    generator = DocumentGenerator(system.config)

    # Build a minimal plan matching what _plan_documents returns
    plan = {
        "petition_type": "Objection and Motion to Dismiss Petition for Guardianship",
        "court_name": "County Court of Douglas County, Nebraska",
        "court_address": None,
        "filing_fee_estimate": "Unknown",
        "strongest_argument": "Respondent is fully competent and the petition is unsupported by medical evidence",
        "opposing_admissions": [],
        "documents": [
            {
                "doc_type": "response",
                "title": "Response and Objection to Petition for Appointment of Guardian",
                "description": "Response to the petition contesting competency",
                "requires_signature": True,
                "requires_notarization": False,
                "filing_required": True,
                "priority": 1,
            }
        ],
        "key_statutes": [
            "Neb. Rev. Stat. § 30-2620 (petition for appointment of guardian)",
            "Neb. Rev. Stat. § 30-2627 (hearing on petition)",
        ],
        "special_notes": [],
    }

    doc_info = plan["documents"][0]
    content = await generator._gen_reply_document(
        doc_info=doc_info,
        situation=SITUATION,
        analysis=stub_analysis,
        clarifications={},
        plan=plan,
        state=USState.NEBRASKA,
        case_law_context=case_law_context,
    )

    # ── Step 6: Check output ──────────────────────────────────────────
    print(f"\n[6] Generated document length: {len(content)} chars")

    cited = [c for c in cases if c.get("case_name", "") in content]
    # Also check partial name matches (first two words of case name)
    partial_hits = []
    for c in cases:
        name_parts = c.get("case_name", "").split()[:3]
        if len(name_parts) >= 2 and " ".join(name_parts) in content:
            partial_hits.append(c.get("case_name"))

    print(f"\n    Exact case name matches in document : {len(cited)}")
    for c in cited:
        print(f"      ✓ {c.get('case_name')}")

    print(f"    Partial name matches (first 3 words): {len(partial_hits)}")
    for n in partial_hits:
        print(f"      ~ {n}")

    # Show the first 2000 chars of the document
    print("\n    --- DOCUMENT PREVIEW (first 2000 chars) ---")
    print(content[:2000])
    print("    --- END PREVIEW ---")

    # Save full output for inspection
    out_path = "scripts/test_response_output.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n    Full document saved to: {out_path}")

    if cited or partial_hits:
        print("\n  RESULT: PASS — court cases are being cited.")
    else:
        print("\n  RESULT: FAIL — no case citations found in the document.")
        print("  Check the DOCUMENT PREVIEW above to see what the model produced.")

if __name__ == "__main__":
    asyncio.run(main())

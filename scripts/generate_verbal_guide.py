"""
Generate a verbal court argument guide from an existing case directory
and a previously drafted written motion, then save it to the output folder.

Usage:
    python scripts/generate_verbal_guide.py \
        --case-dir cases/guardianship-case-against-youssef-eweis \
        --motion output/2026-03-31_10-58-04/motion_2026-03-31_10-58-38.txt \
        --state NE \
        --output-dir output/2026-03-31_10-58-04
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import create_config
from src.documents.case_reader import CaseDirectoryScanner


VERBAL_GUIDE_PROMPT = """You are a seasoned litigation attorney preparing your client, Youssef H. Eweis, to represent himself pro se at a Nebraska guardianship hearing in Sarpy County Court. Based on the full case file and the written motion/objection already filed, produce a detailed VERBAL ARGUMENT GUIDE that Youssef can use when he stands before the judge.

The guide must be practical and plain-spoken — Youssef is not a lawyer. Structure it exactly as follows:

─────────────────────────────────────────────
PART 1: OPENING STATEMENT (what to say first)
─────────────────────────────────────────────
Write the exact words Youssef should say to open. Include:
- How to address the judge
- How to introduce himself
- What he is asking the court for (appointed counsel, continuance, and denial of guardianship)
- Keep it under 90 seconds when spoken aloud

─────────────────────────────────────────────
PART 2: ARGUMENT ON EACH PETITION ALLEGATION
─────────────────────────────────────────────
For each major allegation in the petition, provide:
- A one-sentence plain-English summary of what the petitioner claims
- What Youssef should say in response (spoken words, not legalese)
- What exhibit or document to reference and how to hand it to the court
Cover: healthcare/Medicaid, immediate risk claim, inconsistent communication, the knife allegations, disappearing, VENN Health treatment, prior Iowa guardianship dismissal, family contact, probation compliance.

─────────────────────────────────────────────
PART 3: INTRODUCING EXHIBITS
─────────────────────────────────────────────
For each exhibit, write exactly what Youssef should say when handing it to the judge:
- Exhibit A: Combat Life Saver Certificate (CLS)
- Exhibit B: Holiday card from Probation Officer Nikki Leet
- Sibling contact statements (Aya and Amir)
- VENN Health records (if available)

─────────────────────────────────────────────
PART 4: RESPONDING TO THE OPPOSING ATTORNEY
─────────────────────────────────────────────
Write specific responses Youssef can use if the petitioner's attorney (Jodie L. Haferbier McGill) raises the following objections:
- "He is not capable of managing his medical care"
- "He has been hospitalized multiple times"
- "He was non-compliant with probation"
- "He made threatening statements"
- "He disappeared and could not be located"
- "The prior Iowa guardianship proves he needs supervision"

─────────────────────────────────────────────
PART 5: CLOSING STATEMENT
─────────────────────────────────────────────
Write the exact words for Youssef's closing. He should:
- Summarize why guardianship is unnecessary
- Affirm his capacity and independence
- Formally re-request appointed counsel and continuance
- Thank the court

─────────────────────────────────────────────
PART 6: COURTROOM DO'S AND DON'TS
─────────────────────────────────────────────
Bullet-pointed practical tips: tone, body language, when to speak/stay silent, how to refer to the judge, what to avoid saying.

Write all spoken words in first-person as if Youssef is speaking them. Use plain English. Flag any section where timing is critical (e.g., "say this BEFORE the judge rules on counsel").

FULL CASE FILE:
{case_context}

WRITTEN MOTION ALREADY FILED:
{motion_text}
"""


async def generate_verbal_guide(case_dir: str, motion_path: str, state: str, output_dir: str):
    print("\n  Loading case documents...")
    scanner = CaseDirectoryScanner()
    documents = scanner.scan(case_dir)
    stats = scanner.get_stats(documents)
    context_block = scanner.build_context_block(documents)
    print(f"  Loaded {stats['readable']} documents ({stats['total_chars']:,} chars)")

    motion_text = Path(motion_path).read_text(encoding="utf-8") if Path(motion_path).exists() else "(Motion not found)"

    config = create_config()

    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    genai.configure(api_key=config.google_api_key)
    model = genai.GenerativeModel(config.model_pro)

    prompt = VERBAL_GUIDE_PROMPT.format(
        case_context=context_block,
        motion_text=motion_text,
    )

    print("  Generating verbal argument guide with Gemini Pro...")
    print("  (This may take 30-60 seconds...)\n")

    gen_config = GenerationConfig(temperature=0.3, max_output_tokens=8192)

    response = await asyncio.to_thread(
        model.generate_content,
        prompt,
        generation_config=gen_config,
    )

    guide_text = response.text if hasattr(response, "text") else "(No response generated)"

    # Save to output dir
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / "verbal_argument_guide.txt"
    output_file.write_text(guide_text, encoding="utf-8")

    print(f"  Verbal guide saved to: {output_file}")
    print(f"\n{'='*65}")
    print(guide_text)
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="Generate verbal court argument guide")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--motion", required=True, help="Path to the written motion already drafted")
    parser.add_argument("--state", default="NE")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    asyncio.run(generate_verbal_guide(args.case_dir, args.motion, args.state, args.output_dir))


if __name__ == "__main__":
    main()

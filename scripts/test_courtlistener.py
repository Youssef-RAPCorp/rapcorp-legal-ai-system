#!/usr/bin/env python3
"""
Test CourtListener API connectivity and search.

Run this script to verify your API key and test case law search
before using the full Legal AI system.

Usage:
    python scripts/test_courtlistener.py
    python scripts/test_courtlistener.py --query "wrongful termination" --state CA
    python scripts/test_courtlistener.py --query "breach of contract" --state FED --limit 5

How to get a free API key:
    1. Go to https://www.courtlistener.com/sign-in/
    2. Create a free account
    3. Go to Profile > API Token
    4. Copy the token and add it to your .env file:
           COURTLISTENER_API_KEY=your_token_here
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Allow running from project root or scripts/ directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import create_config
from src.core.courtlistener_client import CourtListenerClient


async def run_test(query: str, state: str, limit: int) -> None:
    config = create_config()

    print("=" * 65)
    print("  CourtListener API Test — RAPCorp Legal AI System")
    print("=" * 65)

    # ── Key check ──────────────────────────────────────────────────
    if not config.courtlistener_api_key:
        print("\n  No COURTLISTENER_API_KEY found in environment or .env file.")
        print("\n  How to get a FREE key:")
        print("    1. Register at  https://www.courtlistener.com/sign-in/")
        print("    2. Go to        Profile > API Token")
        print("    3. Add to .env: COURTLISTENER_API_KEY=your_token_here")
        print("\n  Note: CourtListener is completely free for individual use.")
        sys.exit(1)

    client = CourtListenerClient(config.courtlistener_api_key)

    # ── Connection test ────────────────────────────────────────────
    print("\n  Testing connection...")
    conn = await client.test_connection()

    if conn["status"] == "ok":
        print(f"  Status  : OK")
        print(f"  Database: {conn['total_cases_available']:,} cases available")
    elif conn["status"] == "invalid_key":
        print(f"  Status  : FAILED — invalid API key")
        print(f"  Detail  : {conn['message']}")
        sys.exit(1)
    elif conn["status"] == "no_key":
        print(f"  Status  : FAILED — no key")
        print(f"  Detail  : {conn['message']}")
        sys.exit(1)
    else:
        print(f"  Status  : ERROR — {conn['message']}")
        sys.exit(1)

    # ── Search test ────────────────────────────────────────────────
    jurisdiction_label = f"{state} courts" if state != "FED" else "all courts"
    print(f"\n  Searching: \"{query}\"")
    print(f"  Filter  : {jurisdiction_label} | Limit: {limit}")
    print("  " + "-" * 61)

    result = await client.search_opinions(
        query=query,
        jurisdiction=state if state != "FED" else None,
        page_size=limit,
    )

    if not result.api_available:
        print(f"\n  Search failed: {result.error}")
        sys.exit(1)

    print(f"\n  Total matching cases in database : {result.total_count:,}")
    print(f"  Results returned                 : {len(result.cases)}\n")

    for i, case in enumerate(result.cases, 1):
        print(f"  [{i}] {case.case_name}")
        if case.citation:
            print(f"       Citation     : {case.citation}")
        print(f"       Court        : {case.court}")
        if case.date_filed:
            print(f"       Date Filed   : {case.date_filed}")
        if case.docket_number:
            print(f"       Docket       : {case.docket_number}")
        print(f"       URL          : {case.url}")
        if case.snippet:
            snippet = case.snippet[:220].replace("\n", " ").strip()
            print(f"       Excerpt      : {snippet}...")
        print()

    print("=" * 65)
    print("  All tests passed. CourtListener API is working correctly.")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test CourtListener API connectivity and case law search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_courtlistener.py
  python scripts/test_courtlistener.py --query "wrongful termination" --state CA
  python scripts/test_courtlistener.py --query "Fourth Amendment search seizure" --state FED --limit 3
        """,
    )
    parser.add_argument(
        "--query", "-q",
        default="breach of contract consideration",
        help="Search query (default: 'breach of contract consideration')",
    )
    parser.add_argument(
        "--state", "-s",
        default="FED",
        help="Jurisdiction: state abbreviation (CA, NY, TX...) or FED for all courts (default: FED)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=5,
        help="Number of results to display (default: 5)",
    )
    args = parser.parse_args()
    asyncio.run(run_test(args.query, args.state.upper(), args.limit))


if __name__ == "__main__":
    main()

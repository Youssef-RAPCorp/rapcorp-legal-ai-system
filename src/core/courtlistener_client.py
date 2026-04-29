"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    COURTLISTENER API CLIENT                                   ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Free case law search via the CourtListener REST API v4.                      ║
║  Registration (free): https://www.courtlistener.com/sign-in/                 ║
║  API docs:            https://www.courtlistener.com/help/api/                ║
║                                                                               ║
║  Endpoints used:                                                              ║
║  • /search/    - Full-text opinion search                                     ║
║  • /opinions/  - Fetch individual opinions                                    ║
║  • /clusters/  - Case clusters (groups of related opinions)                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"

# Map state abbreviations to full names for query enrichment.
# CourtListener's court-slug filter is too granular (one slug per court),
# so we get better coverage by including the state name in the search query.
_STATE_TO_NAME: Dict[str, str] = {
    "AL": "Alabama",       "AK": "Alaska",         "AZ": "Arizona",
    "AR": "Arkansas",      "CA": "California",     "CO": "Colorado",
    "CT": "Connecticut",   "DE": "Delaware",       "FL": "Florida",
    "GA": "Georgia",       "HI": "Hawaii",         "ID": "Idaho",
    "IL": "Illinois",      "IN": "Indiana",        "IA": "Iowa",
    "KS": "Kansas",        "KY": "Kentucky",       "LA": "Louisiana",
    "ME": "Maine",         "MD": "Maryland",       "MA": "Massachusetts",
    "MI": "Michigan",      "MN": "Minnesota",      "MS": "Mississippi",
    "MO": "Missouri",      "MT": "Montana",        "NE": "Nebraska",
    "NV": "Nevada",        "NH": "New Hampshire",  "NJ": "New Jersey",
    "NM": "New Mexico",    "NY": "New York",       "NC": "North Carolina",
    "ND": "North Dakota",  "OH": "Ohio",           "OK": "Oklahoma",
    "OR": "Oregon",        "PA": "Pennsylvania",   "RI": "Rhode Island",
    "SC": "South Carolina","SD": "South Dakota",   "TN": "Tennessee",
    "TX": "Texas",         "UT": "Utah",           "VT": "Vermont",
    "VA": "Virginia",      "WA": "Washington",     "WV": "West Virginia",
    "WI": "Wisconsin",     "WY": "Wyoming",        "DC": "District of Columbia",
    "PR": "Puerto Rico",   "GU": "Guam",           "VI": "Virgin Islands",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseLawResult:
    """A single case opinion returned from CourtListener."""
    case_name: str
    citation: str               # Primary citation string, e.g. "410 U.S. 113"
    court: str                  # Court identifier, e.g. "scotus", "ca9"
    date_filed: Optional[str]   # ISO date string YYYY-MM-DD
    url: str                    # Full URL on courtlistener.com
    snippet: str                # Relevant text excerpt from the opinion
    relevance_score: float      # API relevance score (higher = more relevant)
    cluster_id: int             # CourtListener cluster ID
    docket_number: Optional[str] = None


@dataclass
class CourtListenerSearchResult:
    """Aggregated result from a CourtListener opinion search."""
    query: str
    total_count: int            # Total matching cases in the database
    cases: List[CaseLawResult] = field(default_factory=list)
    api_available: bool = True
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class CourtListenerClient:
    """
    Async client for the CourtListener Free Law Project API.

    All methods are async and safe to call from asyncio event loops.
    Methods never raise — they return structured error objects instead,
    so a missing/invalid key never crashes the calling code.

    Usage:
        client = CourtListenerClient(api_key="your_token_here")
        result = await client.search_opinions("wrongful termination", jurisdiction="CA")
        for case in result.cases:
            print(case.case_name, case.citation, case.url)
    """

    BASE_URL = COURTLISTENER_BASE_URL
    _TIMEOUT = aiohttp.ClientTimeout(total=20)

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self._headers = {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json",
        }

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _make_session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(headers=self._headers, timeout=self._TIMEOUT)

    def _parse_cases(self, results: List[Dict]) -> List[CaseLawResult]:
        cases = []
        for r in results:
            citations = r.get("citation") or []
            citation_str = citations[0] if citations else ""
            absolute_url = r.get("absolute_url", "")
            cases.append(CaseLawResult(
                case_name=r.get("caseName") or r.get("case_name") or "Unknown",
                citation=citation_str,
                court=r.get("court_id") or r.get("court") or "",
                date_filed=r.get("dateFiled") or r.get("date_filed"),
                url=f"https://www.courtlistener.com{absolute_url}" if absolute_url else "",
                snippet=(r.get("snippet") or "").strip(),
                relevance_score=float(r.get("score", 0.0)),
                cluster_id=int(r.get("cluster_id", 0)),
                docket_number=r.get("docketNumber") or r.get("docket_number"),
            ))
        return cases

    # ─── Public API ──────────────────────────────────────────────────────────

    async def search_opinions(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        date_after: Optional[str] = None,
        date_before: Optional[str] = None,
        page_size: int = 10,
        order_by: str = "score desc",
    ) -> CourtListenerSearchResult:
        """
        Search for case law opinions by full-text query.

        Args:
            query:        Full-text search query (e.g. "wrongful termination at-will")
            jurisdiction: USState abbreviation ("CA", "NY", "TX") or None for all courts.
                          "FED" is treated as None (all federal courts).
            date_after:   Filter to cases filed after this date (YYYY-MM-DD)
            date_before:  Filter to cases filed before this date (YYYY-MM-DD)
            page_size:    Maximum number of results (default 10, max 20 for free tier)
            order_by:     Sort order — "score desc" (relevance) or "dateFiled desc"

        Returns:
            CourtListenerSearchResult — never raises.
        """
        if not self.is_configured:
            return CourtListenerSearchResult(
                query=query, total_count=0, api_available=False,
                error="No API key configured. Set COURTLISTENER_API_KEY in .env.",
            )

        # Enrich query with state name for better jurisdictional coverage.
        # CourtListener court-slug filters only hit one specific court per slug,
        # so including the state name in the query gives broader, more accurate results.
        enriched_query = query
        if jurisdiction and jurisdiction.upper() not in ("FED", ""):
            state_name = _STATE_TO_NAME.get(jurisdiction.upper())
            if state_name and state_name.lower() not in query.lower():
                enriched_query = f"{query} {state_name}"

        params: Dict[str, Any] = {
            "q": enriched_query,
            "type": "o",          # opinion search
            "order_by": order_by,
            "page_size": min(page_size, 20),
        }

        if date_after:
            params["filed_after"] = date_after
        if date_before:
            params["filed_before"] = date_before

        try:
            async with self._make_session() as session:
                async with session.get(f"{self.BASE_URL}/search/", params=params) as resp:
                    if resp.status == 401:
                        return CourtListenerSearchResult(
                            query=query, total_count=0, api_available=False,
                            error=(
                                "Invalid API key (HTTP 401). "
                                "Register free at courtlistener.com and add "
                                "COURTLISTENER_API_KEY to your .env file."
                            ),
                        )
                    if resp.status == 429:
                        return CourtListenerSearchResult(
                            query=query, total_count=0, api_available=False,
                            error="Rate limit exceeded (HTTP 429). Wait a moment and retry.",
                        )
                    resp.raise_for_status()
                    data = await resp.json()

            cases = self._parse_cases(data.get("results", []))
            return CourtListenerSearchResult(
                query=query,
                total_count=data.get("count", len(cases)),
                cases=cases,
                api_available=True,
            )

        except aiohttp.ClientConnectorError:
            return CourtListenerSearchResult(
                query=query, total_count=0, api_available=False,
                error="Could not reach courtlistener.com. Check internet connection.",
            )
        except Exception as e:
            return CourtListenerSearchResult(
                query=query, total_count=0, api_available=False,
                error=f"Unexpected error: {e}",
            )

    async def get_opinion(self, opinion_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch the full text and metadata of a single opinion by its ID.
        Returns None on any error.
        """
        if not self.is_configured:
            return None
        try:
            async with self._make_session() as session:
                async with session.get(f"{self.BASE_URL}/opinions/{opinion_id}/") as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception:
            return None

    async def get_cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a case cluster (all opinions in a single decided case).
        Returns None on any error.
        """
        if not self.is_configured:
            return None
        try:
            async with self._make_session() as session:
                async with session.get(f"{self.BASE_URL}/clusters/{cluster_id}/") as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception:
            return None

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connectivity and key validity with a minimal API call.

        Returns a dict with keys:
            status  — "ok" | "invalid_key" | "network_error" | "error"
            message — human-readable description
            total_cases_available — (only when status == "ok")
        """
        if not self.is_configured:
            return {
                "status": "no_key",
                "message": (
                    "No API key set. Register free at courtlistener.com, "
                    "then add COURTLISTENER_API_KEY=<token> to your .env file."
                ),
            }

        try:
            async with self._make_session() as session:
                async with session.get(
                    f"{self.BASE_URL}/search/",
                    params={"q": "contract breach", "type": "o", "page_size": 1},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "status": "ok",
                            "total_cases_available": data.get("count", 0),
                            "message": "CourtListener API is working correctly.",
                        }
                    if resp.status == 401:
                        return {
                            "status": "invalid_key",
                            "message": (
                                "API key rejected (HTTP 401). "
                                "Verify your token at courtlistener.com > Profile > API Token."
                            ),
                        }
                    return {
                        "status": "error",
                        "message": f"Unexpected HTTP {resp.status}.",
                    }

        except aiohttp.ClientConnectorError:
            return {
                "status": "network_error",
                "message": "Could not reach courtlistener.com. Check internet connection.",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

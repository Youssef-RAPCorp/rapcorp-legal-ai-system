"""
Working Memory — temporary thought storage for the legal thinking engine.

Each "thought" is a named unit of reasoning.  Later thoughts reference earlier
ones by name, creating a genuine chain where every decision is grounded in
prior conclusions rather than generated in isolation.

This mirrors how a human strategist actually thinks:
  "Given what I know about the case (thought: case_framing) and what the judge
   cares about (thought: judge_profile), the strongest theory is X because..."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Thought:
    name:      str
    step:      int
    content:   str
    refs:      List[str] = field(default_factory=list)   # names of referenced prior thoughts
    timestamp: str       = field(default_factory=lambda: datetime.utcnow().isoformat())

    def summary(self, max_chars: int = 300) -> str:
        text = self.content[:max_chars]
        if len(self.content) > max_chars:
            text += "…"
        return f"[{self.name}] {text}"


class WorkingMemory:
    """
    Ordered store of named thoughts.  Each thought can reference prior ones by
    name.  The memory is ephemeral — it lives only for one strategy run.

    Usage:
        mem = WorkingMemory()
        mem.add("case_framing", "This is a guardianship contest where...")
        mem.add("judge_profile", "Probate judges are primarily concerned with...",
                refs=["case_framing"])
        ctx = mem.build_context(["case_framing", "judge_profile"])
    """

    def __init__(self):
        self._thoughts: Dict[str, Thought] = {}
        self._order:    List[str]          = []
        self._step:     int                = 0

    # ── Write ──────────────────────────────────────────────────────────────

    def add(self, name: str, content: str, refs: Optional[List[str]] = None) -> Thought:
        """Store a new thought.  Returns the Thought object."""
        self._step += 1
        thought = Thought(
            name=name,
            step=self._step,
            content=content,
            refs=refs or [],
        )
        self._thoughts[name] = thought
        self._order.append(name)
        return thought

    def update(self, name: str, content: str) -> None:
        """Replace the content of an existing thought."""
        if name in self._thoughts:
            self._thoughts[name].content = content

    # ── Read ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[Thought]:
        return self._thoughts.get(name)

    def get_content(self, name: str, fallback: str = "") -> str:
        t = self._thoughts.get(name)
        return t.content if t else fallback

    def all_thoughts(self) -> List[Thought]:
        return [self._thoughts[n] for n in self._order]

    # ── Context builders ───────────────────────────────────────────────────

    def build_context(
        self,
        names:    Optional[List[str]] = None,
        max_chars_each: int = 800,
    ) -> str:
        """
        Build a readable context block from selected (or all) thoughts.
        Used to give the LLM the full chain of reasoning so far.
        """
        targets = names if names else self._order
        lines   = []
        for name in targets:
            t = self._thoughts.get(name)
            if not t:
                continue
            content = t.content[:max_chars_each]
            if t.refs:
                lines.append(f"[THOUGHT {t.step}: {name}] (builds on: {', '.join(t.refs)})")
            else:
                lines.append(f"[THOUGHT {t.step}: {name}]")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip()

    def build_full_log(self) -> str:
        """Return the complete thought log for debugging / saving."""
        lines = ["═══ THINKING LOG ═══", ""]
        for t in self.all_thoughts():
            refs = f" ← {', '.join(t.refs)}" if t.refs else ""
            lines.append(f"Step {t.step:02d} [{t.name}]{refs}")
            lines.append(t.content)
            lines.append("")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._thoughts)

    def __contains__(self, name: str) -> bool:
        return name in self._thoughts

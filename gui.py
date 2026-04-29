#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    RAPCorp Legal AI System — Desktop GUI                      ║
║                                                                               ║
║  Run:  python gui.py                                                          ║
║  Deps: pip install customtkinter python-docx                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import asyncio
import threading
import queue
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
    import tkinter as tk
except ImportError:
    print("customtkinter not installed. Run:  pip install customtkinter")
    sys.exit(1)

from configs.config import create_config, USState

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── State list for the dropdown ───────────────────────────────────────────────
_STATES = sorted(s.value for s in USState)

# ── File type filter sets ─────────────────────────────────────────────────────
_CASE_TYPES = (
    ("All supported", "*.pdf *.txt *.doc *.docx *.md *.rtf *.csv *.html *.xml *.json"),
    ("PDF",           "*.pdf"),
    ("Word",          "*.doc *.docx"),
    ("Text",          "*.txt *.md *.rtf *.csv"),
    ("All files",     "*.*"),
)
_EVIDENCE_TYPES = (
    ("All supported",
     "*.mp3 *.wav *.aac *.ogg *.flac *.m4a *.aiff *.wma "
     "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.mpeg *.3gp "
     "*.png *.jpg *.jpeg *.gif *.webp *.bmp "
     "*.pdf *.txt *.md *.csv *.json *.xml *.html"),
    ("Audio",     "*.mp3 *.wav *.aac *.ogg *.flac *.m4a *.aiff *.wma"),
    ("Video",     "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.mpeg"),
    ("Images",    "*.png *.jpg *.jpeg *.gif *.webp *.bmp"),
    ("Documents", "*.pdf *.txt *.md *.csv"),
    ("All files", "*.*"),
)

_DOC_BADGE_COLORS = {
    "petition":              "#1a5fa8",
    "affidavit":             "#1a7a3a",
    "exhibit_index":         "#6a309a",
    "proposed_order":        "#8a3020",
    "certificate_of_service":"#505050",
    "cover_sheet":           "#404040",
    "checklist":             "#383838",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOG REDIRECT
# ═══════════════════════════════════════════════════════════════════════════════

class _LogRedirect:
    """
    Tees stdout to the original terminal and a queue read by the GUI log.
    Installed only during a pipeline run; restored when done.
    """
    def __init__(self, q: queue.Queue):
        self._q   = q
        self._orig = sys.__stdout__

    def write(self, text: str):
        self._orig.write(text)
        if text.strip():
            self._q.put(("log", text.rstrip("\n")))

    def flush(self):
        self._orig.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class LegalAIApp(ctk.CTk):

    _PRESET_INSTRUCTIONS: dict[str, str] = {
        "Remove psychosis/schizophrenia diagnostic claims": (
            "Remove all definitive statements about whether the Respondent does or does not "
            "have psychosis or schizophrenia. The most recent medical evaluation is ambiguous "
            "on this point. Specifically, rewrite any sentence claiming 'not schizophrenia', "
            "'no psychosis was present', 'denied any psychosis or delusions', or any equivalent "
            "negative diagnostic assertion — replace them by focusing solely on what the evaluation "
            "positively found: the specific diagnosis code and condition name given, and the "
            "provider's assessment of current functioning. Do not claim the Respondent has or "
            "does not have psychosis or schizophrenia. Preserve all other content exactly."
        ),
        "Restore Nebraska statute citations": (
            "Ensure the document cites the following Nebraska statutes by their full section numbers "
            "in the appropriate legal argument sections. If any are already present, leave them "
            "unchanged. Only insert missing citations in a legally correct location:\n"
            "- Neb. Rev. Stat. § 30-4201 et seq. (protected person standard, Uniform Guardianship Act)\n"
            "- Neb. Rev. Stat. § 30-2630 (conservatorship: clear and convincing evidence that the "
            "person cannot manage their property and affairs effectively)\n"
            "- Neb. Rev. Stat. § 30-2620 (guardianship: clear and convincing evidence that the person "
            "lacks sufficient understanding or capacity to make or communicate responsible decisions)\n"
            "- In re Guardianship & Conservatorship of Larson, 270 Neb. 837 (2006) (burden of proof)\n"
            "Preserve all other content exactly."
        ),
    }

    def __init__(self):
        super().__init__()
        self.title("RAPCorp Legal AI System")
        self.geometry("1380x820")
        self.minsize(1100, 700)

        self._case_files:     list[str] = []
        self._evidence_files: list[str] = []
        self._generated_docs: list[dict] = []
        self._editable_docs:  list[dict] = []
        self._system = None
        self._running = False
        self._log_q: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_queue()

    # ─── UI construction ─────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()

    def _build_sidebar(self):
        # Outer frame holds the fixed width; inner scrollable frame holds all widgets
        sb_outer = ctk.CTkFrame(self, width=300, corner_radius=0)
        sb_outer.grid(row=0, column=0, sticky="nsew")
        sb_outer.grid_propagate(False)
        sb_outer.grid_columnconfigure(0, weight=1)
        sb_outer.grid_rowconfigure(0, weight=1)

        sb = ctk.CTkScrollableFrame(sb_outer, fg_color="transparent", corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_columnconfigure(0, weight=1)

        row = 0

        # Logo
        _logo_path = PROJECT_ROOT / "RAPCorp-logo.png"
        try:
            from PIL import Image as _PILImage
            _pil_logo = _PILImage.open(_logo_path)
            _logo_img = ctk.CTkImage(
                light_image=_pil_logo, dark_image=_pil_logo, size=(180, 180)
            )
            ctk.CTkLabel(sb, image=_logo_img, text="").grid(
                row=row, column=0, pady=(16, 2)); row += 1
        except Exception:
            ctk.CTkLabel(
                sb, text="RAPCorp\nLegal AI",
                font=ctk.CTkFont(size=22, weight="bold"),
                justify="center",
            ).grid(row=row, column=0, padx=20, pady=(22, 6), sticky="ew"); row += 1

        ctk.CTkLabel(
            sb, text="Evidence → Documents", text_color="gray60",
            font=ctk.CTkFont(size=11),
        ).grid(row=row, column=0, padx=20, pady=(0, 14), sticky="ew"); row += 1

        # Divider
        ctk.CTkFrame(sb, height=1, fg_color="gray25").grid(
            row=row, column=0, padx=16, sticky="ew"); row += 1

        # Jurisdiction
        ctk.CTkLabel(sb, text="Jurisdiction", anchor="w",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=row, column=0, padx=20, pady=(12, 2), sticky="ew"); row += 1

        self._state_var = ctk.StringVar(value="CALIFORNIA")
        ctk.CTkOptionMenu(sb, values=_STATES, variable=self._state_var,
                           dynamic_resizing=False).grid(
            row=row, column=0, padx=20, pady=(0, 10), sticky="ew"); row += 1

        # ── Case documents ────────────────────────────────────────────────
        ctk.CTkFrame(sb, height=1, fg_color="gray25").grid(
            row=row, column=0, padx=16, sticky="ew"); row += 1

        ctk.CTkLabel(sb, text="Case Documents", anchor="w",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=row, column=0, padx=20, pady=(10, 2), sticky="ew"); row += 1

        ctk.CTkLabel(sb, text="Existing pleadings, letters, orders…",
                     text_color="gray55", font=ctk.CTkFont(size=10), anchor="w").grid(
            row=row, column=0, padx=20, pady=(0, 4), sticky="ew"); row += 1

        self._case_listbox = ctk.CTkTextbox(sb, height=78, font=ctk.CTkFont(size=10),
                                             state="disabled")
        self._case_listbox.grid(row=row, column=0, padx=20, sticky="ew"); row += 1

        case_btns = ctk.CTkFrame(sb, fg_color="transparent")
        case_btns.grid(row=row, column=0, padx=20, pady=(4, 4), sticky="ew"); row += 1
        ctk.CTkButton(case_btns, text="+ Upload Case Docs",
                       command=self._upload_case, height=30).pack(
            side="left", expand=True, fill="x")
        ctk.CTkButton(case_btns, text="Clear", width=52,
                       command=lambda: self._clear_files("case"),
                       height=30, fg_color="gray25", hover_color="gray18").pack(
            side="left", padx=(5, 0))

        # ── Evidence files ────────────────────────────────────────────────
        ctk.CTkFrame(sb, height=1, fg_color="gray25").grid(
            row=row, column=0, padx=16, sticky="ew"); row += 1

        ctk.CTkLabel(sb, text="Evidence Files", anchor="w",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=row, column=0, padx=20, pady=(10, 2), sticky="ew"); row += 1

        ctk.CTkLabel(sb, text="Audio, video, images, PDFs…",
                     text_color="gray55", font=ctk.CTkFont(size=10), anchor="w").grid(
            row=row, column=0, padx=20, pady=(0, 4), sticky="ew"); row += 1

        self._ev_listbox = ctk.CTkTextbox(sb, height=78, font=ctk.CTkFont(size=10),
                                           state="disabled")
        self._ev_listbox.grid(row=row, column=0, padx=20, sticky="ew"); row += 1

        ev_btns = ctk.CTkFrame(sb, fg_color="transparent")
        ev_btns.grid(row=row, column=0, padx=20, pady=(4, 4), sticky="ew"); row += 1
        ctk.CTkButton(ev_btns, text="+ Upload Evidence",
                       command=self._upload_evidence, height=30).pack(
            side="left", expand=True, fill="x")
        ctk.CTkButton(ev_btns, text="Clear", width=52,
                       command=lambda: self._clear_files("evidence"),
                       height=30, fg_color="gray25", hover_color="gray18").pack(
            side="left", padx=(5, 0))

        # ── Situation text ────────────────────────────────────────────────
        ctk.CTkFrame(sb, height=1, fg_color="gray25").grid(
            row=row, column=0, padx=16, sticky="ew"); row += 1

        ctk.CTkLabel(sb, text="Case Situation", anchor="w",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=row, column=0, padx=20, pady=(10, 2), sticky="ew"); row += 1

        ctk.CTkLabel(sb, text="Describe your situation before the run starts.",
                     text_color="gray55", font=ctk.CTkFont(size=10),
                     wraplength=240, anchor="w", justify="left").grid(
            row=row, column=0, padx=20, pady=(0, 4), sticky="ew"); row += 1

        self._situation_box = ctk.CTkTextbox(sb, height=110,
                                              font=ctk.CTkFont(size=11), wrap="word")
        self._situation_box.grid(row=row, column=0, padx=20, sticky="ew"); row += 1
        self._situation_box.insert("0.0",
            "Describe your case situation here (what happened, who was involved, "
            "what outcome you need)…")

        # ── Run button ────────────────────────────────────────────────────
        ctk.CTkFrame(sb, height=1, fg_color="gray25").grid(
            row=row, column=0, padx=16, pady=(10, 0), sticky="ew"); row += 1

        self._run_btn = ctk.CTkButton(
            sb, text="▶  Run Analysis & Generate Docs",
            command=self._run,
            height=46, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1a5fa8", hover_color="#103d6e",
        )
        self._run_btn.grid(row=row, column=0, padx=20, pady=(10, 20), sticky="ew"); row += 1

    def _build_main_area(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=(6, 10), pady=10)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=0)

        # ── Tab view ──────────────────────────────────────────────────────
        self._tabs = ctk.CTkTabview(main, anchor="nw")
        self._tabs.grid(row=0, column=0, sticky="nsew")

        self._build_tab_progress()
        self._build_tab_documents()
        self._build_tab_edit()
        self._build_tab_research()

        # ── Progress bar (shown during runs) ─────────────────────────────
        self._progress_bar = ctk.CTkProgressBar(main, mode="indeterminate", height=6)
        self._progress_bar.grid(row=1, column=0, sticky="ew", padx=2, pady=(4, 0))
        self._progress_bar.set(0)

    # ── Tab: Progress ─────────────────────────────────────────────────────

    def _build_tab_progress(self):
        tab = self._tabs.add("Progress")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self._log_box = ctk.CTkTextbox(
            tab, state="disabled",
            font=ctk.CTkFont(family="Courier New", size=11),
            wrap="word",
        )
        self._log_box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    # ── Tab: Generated Documents ─────────────────────────────────────────

    def _build_tab_documents(self):
        tab = self._tabs.add("Generated Documents")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        self._docs_hint = ctk.CTkLabel(
            tab,
            text="Documents will appear here after the run completes.",
            text_color="gray55", anchor="w",
        )
        self._docs_hint.grid(row=0, column=0, padx=12, pady=(8, 4), sticky="ew")

        self._docs_scroll = ctk.CTkScrollableFrame(tab, label_text="")
        self._docs_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        self._docs_scroll.grid_columnconfigure(0, weight=1)

    # ── Tab: Find & Edit ──────────────────────────────────────────────────

    def _build_tab_edit(self):
        tab = self._tabs.add("Find & Edit")
        tab.grid_columnconfigure(0, weight=1)

        r = 0

        # Document selector row
        sel_row = ctk.CTkFrame(tab, fg_color="transparent")
        sel_row.grid(row=r, column=0, sticky="ew", padx=12, pady=(12, 6)); r += 1
        sel_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(sel_row, text="Document:", width=90, anchor="w").grid(
            row=0, column=0)
        self._edit_doc_var = ctk.StringVar(value="(run analysis first)")
        self._edit_doc_menu = ctk.CTkOptionMenu(
            sel_row, variable=self._edit_doc_var,
            values=["(run analysis first)"], dynamic_resizing=False,
        )
        self._edit_doc_menu.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ctk.CTkButton(sel_row, text="Open in Editor", width=110,
                       command=self._open_selected_doc).grid(row=0, column=2)

        # ── Find / Replace ────────────────────────────────────────────────
        fr_card = ctk.CTkFrame(tab)
        fr_card.grid(row=r, column=0, sticky="ew", padx=12, pady=4); r += 1
        fr_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(fr_card, text="Find:", width=80, anchor="e").grid(
            row=0, column=0, padx=(14, 0), pady=(10, 6))
        self._find_entry = ctk.CTkEntry(fr_card, placeholder_text="Text to find…")
        self._find_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=(10, 6))
        ctk.CTkButton(fr_card, text="Find", width=80,
                       command=self._find_in_doc).grid(row=0, column=2, padx=(0, 12),
                                                        pady=(10, 6))

        ctk.CTkLabel(fr_card, text="Replace:", width=80, anchor="e").grid(
            row=1, column=0, padx=(14, 0), pady=(0, 10))
        self._replace_entry = ctk.CTkEntry(fr_card, placeholder_text="Replace with…")
        self._replace_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 10))

        rep_btns = ctk.CTkFrame(fr_card, fg_color="transparent")
        rep_btns.grid(row=1, column=2, padx=(0, 12), pady=(0, 10))
        ctk.CTkButton(rep_btns, text="Replace All", width=105,
                       command=self._replace_all).pack()

        self._find_status = ctk.CTkLabel(tab, text="", text_color="gray55",
                                          anchor="w", font=ctk.CTkFont(size=11))
        self._find_status.grid(row=r, column=0, padx=16, sticky="ew"); r += 1

        # Divider
        ctk.CTkFrame(tab, height=1, fg_color="gray25").grid(
            row=r, column=0, padx=12, pady=(10, 8), sticky="ew"); r += 1

        # ── AI Post-Processing Fix ────────────────────────────────────────
        ctk.CTkLabel(tab, text="AI Post-Processing Fix",
                     font=ctk.CTkFont(size=13, weight="bold"), anchor="w").grid(
            row=r, column=0, padx=14, pady=(0, 2), sticky="ew"); r += 1

        ctk.CTkLabel(
            tab,
            text="Describe a correction in plain English — AI will apply it to the "
                 "selected document (or all documents).",
            text_color="gray55", anchor="w", wraplength=700, justify="left",
            font=ctk.CTkFont(size=11),
        ).grid(row=r, column=0, padx=14, sticky="ew"); r += 1

        self._post_box = ctk.CTkTextbox(tab, height=90,
                                         font=ctk.CTkFont(size=11), wrap="word")
        self._post_box.grid(row=r, column=0, padx=12, pady=(6, 6), sticky="ew"); r += 1
        self._post_box.insert(
            "0.0",
            "E.g. \"Replace every occurrence of [Petitioner Name] with Jane Smith\" or "
            "\"Add a paragraph after paragraph 5 clarifying that the incident occurred at work\"…",
        )

        ai_btns = ctk.CTkFrame(tab, fg_color="transparent")
        ai_btns.grid(row=r, column=0, padx=12, pady=(0, 14), sticky="ew"); r += 1

        ctk.CTkButton(ai_btns, text="Apply AI Fix to Selected Doc",
                       command=lambda: self._apply_ai_fix("selected"),
                       height=34).pack(side="left")
        ctk.CTkButton(ai_btns, text="Apply AI Fix to All Docs",
                       command=lambda: self._apply_ai_fix("all"),
                       height=34, fg_color="gray30",
                       hover_color="gray20").pack(side="left", padx=(8, 0))

        # Divider
        ctk.CTkFrame(tab, height=1, fg_color="gray25").grid(
            row=r, column=0, padx=12, pady=(10, 8), sticky="ew"); r += 1

        # ── Preset Legal Updates ───────────────────────────────────────────
        ctk.CTkLabel(tab, text="Preset Legal Updates",
                     font=ctk.CTkFont(size=13, weight="bold"), anchor="w").grid(
            row=r, column=0, padx=14, pady=(0, 2), sticky="ew"); r += 1

        ctk.CTkLabel(
            tab,
            text="Apply a pre-defined legal update to all generated documents with one click.",
            text_color="gray55", anchor="w", wraplength=700, justify="left",
            font=ctk.CTkFont(size=11),
        ).grid(row=r, column=0, padx=14, sticky="ew"); r += 1

        preset_row = ctk.CTkFrame(tab, fg_color="transparent")
        preset_row.grid(row=r, column=0, padx=12, pady=(8, 4), sticky="ew"); r += 1
        preset_row.grid_columnconfigure(0, weight=1)

        self._preset_var = ctk.StringVar(value=list(self._PRESET_INSTRUCTIONS.keys())[0])
        ctk.CTkOptionMenu(
            preset_row, variable=self._preset_var,
            values=list(self._PRESET_INSTRUCTIONS.keys()),
            dynamic_resizing=False,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            preset_row, text="Apply to All Docs", width=130,
            command=self._apply_preset,
        ).grid(row=0, column=1)

        ctk.CTkLabel(
            tab,
            text="Additional notes (optional) — appended to the preset instruction above:",
            text_color="gray55", anchor="w",
            font=ctk.CTkFont(size=11),
        ).grid(row=r, column=0, padx=14, pady=(6, 2), sticky="ew"); r += 1

        self._preset_notes_box = ctk.CTkTextbox(tab, height=64,
                                                 font=ctk.CTkFont(size=11), wrap="word")
        self._preset_notes_box.grid(row=r, column=0, padx=12, pady=(0, 14),
                                     sticky="ew"); r += 1

    # ── Tab: Research ─────────────────────────────────────────────────────

    def _build_tab_research(self):
        tab = self._tabs.add("Research")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(5, weight=1)

        r = 0

        # ── Query ─────────────────────────────────────────────────────
        ctk.CTkLabel(tab, text="Research Question",
                     font=ctk.CTkFont(size=13, weight="bold"), anchor="w").grid(
            row=r, column=0, padx=14, pady=(12, 2), sticky="ew"); r += 1

        self._research_query_box = ctk.CTkTextbox(
            tab, height=88, font=ctk.CTkFont(size=12), wrap="word")
        self._research_query_box.grid(
            row=r, column=0, padx=12, pady=(0, 8), sticky="ew"); r += 1

        # ── Options row ───────────────────────────────────────────────
        opts = ctk.CTkFrame(tab, fg_color="transparent")
        opts.grid(row=r, column=0, padx=12, pady=(0, 6), sticky="ew"); r += 1
        opts.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(opts, text="Jurisdiction:", anchor="w",
                     font=ctk.CTkFont(size=11)).grid(
            row=0, column=0, sticky="w")
        _research_state_values = ["FEDERAL"] + _STATES
        self._research_state_var = ctk.StringVar(value="FEDERAL")
        ctk.CTkComboBox(
            opts, variable=self._research_state_var,
            values=_research_state_values,
        ).grid(row=1, column=0, padx=(0, 8), sticky="ew")

        ctk.CTkLabel(opts, text="Export:", anchor="w",
                     font=ctk.CTkFont(size=11)).grid(
            row=0, column=1, sticky="w")
        self._research_export_var = ctk.StringVar(value="html")
        ctk.CTkOptionMenu(opts, variable=self._research_export_var,
                          values=["html", "md", "json", "all", "none"],
                          dynamic_resizing=False).grid(
            row=1, column=1, sticky="ew")

        # ── Run button ────────────────────────────────────────────────
        self._research_btn = ctk.CTkButton(
            tab, text="Run Research",
            command=self._run_research,
            height=40, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1a5fa8", hover_color="#103d6e",
        )
        self._research_btn.grid(
            row=r, column=0, padx=12, pady=(4, 2), sticky="ew"); r += 1

        # ── Status ────────────────────────────────────────────────────
        self._research_status = ctk.CTkLabel(
            tab, text="Pick a jurisdiction from the list or type any state abbreviation (e.g. NE, CA).",
            text_color="gray55", anchor="w", font=ctk.CTkFont(size=11))
        self._research_status.grid(
            row=r, column=0, padx=16, pady=(2, 4), sticky="ew"); r += 1

        # ── Results ───────────────────────────────────────────────────
        ctk.CTkLabel(tab, text="Results",
                     font=ctk.CTkFont(size=12, weight="bold"), anchor="w").grid(
            row=r, column=0, padx=14, pady=(2, 2), sticky="ew"); r += 1

        tab.grid_rowconfigure(r, weight=1)
        self._research_result_box = ctk.CTkTextbox(
            tab, state="disabled",
            font=ctk.CTkFont(family="Georgia", size=12), wrap="word",
        )
        self._research_result_box.grid(
            row=r, column=0, padx=12, pady=(0, 12), sticky="nsew")

    # ─── File upload helpers ──────────────────────────────────────────────

    def _upload_case(self):
        files = filedialog.askopenfilenames(
            title="Select case documents", filetypes=_CASE_TYPES)
        for f in files:
            if f not in self._case_files:
                self._case_files.append(f)
        self._refresh_filebox(self._case_listbox, self._case_files)

    def _upload_evidence(self):
        files = filedialog.askopenfilenames(
            title="Select evidence files", filetypes=_EVIDENCE_TYPES)
        for f in files:
            if f not in self._evidence_files:
                self._evidence_files.append(f)
        self._refresh_filebox(self._ev_listbox, self._evidence_files)

    def _clear_files(self, which: str):
        if which == "case":
            self._case_files.clear()
            self._refresh_filebox(self._case_listbox, self._case_files)
        else:
            self._evidence_files.clear()
            self._refresh_filebox(self._ev_listbox, self._evidence_files)

    def _refresh_filebox(self, box: ctk.CTkTextbox, files: list):
        box.configure(state="normal")
        box.delete("0.0", "end")
        box.insert("0.0",
                   "\n".join(Path(f).name for f in files) if files else "(none)")
        box.configure(state="disabled")

    # ─── Log helpers ──────────────────────────────────────────────────────

    def _log(self, text: str):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._log_q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self._on_run_complete(payload)
                elif kind == "error":
                    self._on_run_error(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ─── Pipeline runner ──────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return

        situation = self._situation_box.get("0.0", "end").strip()
        _placeholder = "Describe your case situation here"
        if not situation or situation.startswith(_placeholder):
            messagebox.showwarning(
                "Missing input",
                "Please enter a case situation description in the sidebar.")
            return

        all_files = self._case_files + self._evidence_files
        if not all_files:
            messagebox.showwarning(
                "No files",
                "Upload at least one case document or evidence file first.")
            return

        state_str = self._state_var.get()

        self._running = True
        self._run_btn.configure(state="disabled", text="Running…")
        self._progress_bar.start()
        self._tabs.set("Progress")
        self._log("━" * 64)
        self._log(f"  RAPCorp Legal AI  —  {state_str}")
        self._log(f"  Case docs  : {len(self._case_files)}")
        self._log(f"  Evidence   : {len(self._evidence_files)}")
        self._log("━" * 64)

        self._old_stdout = sys.stdout
        sys.stdout = _LogRedirect(self._log_q)

        threading.Thread(
            target=self._pipeline_thread,
            args=(situation, state_str,
                  list(self._case_files), list(self._evidence_files)),
            daemon=True,
        ).start()

    def _pipeline_thread(self, situation, state_str, case_files, evidence_files):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._pipeline(situation, state_str, case_files, evidence_files)
            )
            self._log_q.put(("done", result))
        except Exception as exc:
            self._log_q.put(("error", f"{exc}\n{traceback.format_exc()}"))
        finally:
            loop.close()

    async def _pipeline(self, situation, state_str, case_files, evidence_files):
        from src.legal_ai_system import create_legal_ai_system
        from src.documents.docx_writer import txt_to_docx

        system = await create_legal_ai_system()
        self._system = system

        try:
            state_enum = USState(state_str)
        except ValueError:
            state_enum = USState.FEDERAL

        all_files = case_files + evidence_files

        # Phase 1 — Evidence analysis + chain of events
        result = await system.analyze_evidence(
            situation_description=situation,
            file_paths=all_files,
            state=state_enum,
        )
        analysis = result["_result_obj"]

        # Phase 2 — Document generation + iterative review
        docs = await system.generate_petition_documents(
            situation=situation,
            analysis=analysis,
            clarifications={},
            state=state_enum,
        )

        # Phase 3 — Convert every .txt output to .docx
        for doc in docs:
            txt_path = doc.get("file_path", "")
            if txt_path and Path(txt_path).exists() and txt_path.endswith(".txt"):
                docx_path = txt_path.replace(".txt", ".docx")
                try:
                    txt_to_docx(txt_path, docx_path, title=doc.get("title", ""))
                    doc["docx_path"] = docx_path
                    print(f"  .docx created: {Path(docx_path).name}")
                except Exception as exc:
                    print(f"  Warning: could not create .docx for "
                          f"{Path(txt_path).name}: {exc}")
                    doc["docx_path"] = txt_path  # fall back to .txt

        return docs

    def _on_run_complete(self, docs: list):
        sys.stdout = self._old_stdout
        self._running = False
        self._progress_bar.stop()
        self._progress_bar.set(0)
        self._run_btn.configure(state="normal", text="▶  Run Analysis & Generate Docs")
        self._generated_docs = docs

        # Populate Generated Documents tab
        self._populate_docs_tab(docs)

        # Update Find & Edit document selector
        editable = [
            d for d in docs
            if "REVIEW_LOG" not in d.get("filename", "")
            and d.get("doc_type") not in ("checklist",)
        ]
        self._editable_docs = editable
        if editable:
            names = [
                Path(d.get("docx_path", d.get("file_path", "?"))).name
                for d in editable
            ]
            self._edit_doc_menu.configure(values=names)
            self._edit_doc_var.set(names[0])

        self._tabs.set("Generated Documents")
        self._log("\n✓ All done — see 'Generated Documents' tab.")

    def _on_run_error(self, msg: str):
        sys.stdout = self._old_stdout
        self._running = False
        self._progress_bar.stop()
        self._progress_bar.set(0)
        self._run_btn.configure(state="normal", text="▶  Run Analysis & Generate Docs")
        self._log(f"\n✗ Error: {msg}")
        messagebox.showerror("Pipeline error", msg[:400])

    # ─── Generated Documents tab ──────────────────────────────────────────

    def _populate_docs_tab(self, docs: list):
        # Clear old rows
        for widget in self._docs_scroll.winfo_children():
            widget.destroy()

        self._docs_hint.configure(
            text=f"{len(docs)} document(s) generated — click Open to view in Word / Notepad."
        )

        for i, doc in enumerate(docs):
            path = doc.get("docx_path", doc.get("file_path", ""))
            doc_type = doc.get("doc_type", "other")
            title = doc.get("title", "Unknown")
            badge_color = _DOC_BADGE_COLORS.get(doc_type, "#404040")

            row_frame = ctk.CTkFrame(self._docs_scroll, corner_radius=8)
            row_frame.grid(row=i, column=0, sticky="ew", pady=3, padx=2)
            row_frame.grid_columnconfigure(1, weight=1)
            self._docs_scroll.grid_columnconfigure(0, weight=1)

            # Type badge
            ctk.CTkLabel(
                row_frame,
                text=doc_type.replace("_", " ").upper(),
                fg_color=badge_color, corner_radius=5,
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color="white", width=138,
                padx=6, pady=2,
            ).grid(row=0, column=0, rowspan=2, padx=(10, 8), pady=10, sticky="w")

            # Title & filename
            info = ctk.CTkFrame(row_frame, fg_color="transparent")
            info.grid(row=0, column=1, rowspan=2, sticky="ew", pady=8)
            ctk.CTkLabel(info, text=title,
                          font=ctk.CTkFont(size=13, weight="bold"),
                          anchor="w").pack(anchor="w")
            ctk.CTkLabel(info, text=Path(path).name if path else "—",
                          text_color="gray55",
                          font=ctk.CTkFont(size=10),
                          anchor="w").pack(anchor="w")

            tags = []
            if doc.get("requires_signature"):    tags.append("SIGN")
            if doc.get("requires_notarization"): tags.append("NOTARIZE")
            if tags:
                ctk.CTkLabel(info, text="  ".join(tags),
                              text_color="#e0a020",
                              font=ctk.CTkFont(size=10, weight="bold"),
                              anchor="w").pack(anchor="w")

            # Open button
            def _opener(p=path):
                if p and Path(p).exists():
                    if sys.platform == "win32":
                        os.startfile(p)
                    else:
                        import subprocess
                        opener = "open" if sys.platform == "darwin" else "xdg-open"
                        subprocess.Popen([opener, p])
                else:
                    messagebox.showwarning("File not found", f"Cannot find: {p}")

            ctk.CTkButton(row_frame, text="Open", width=72,
                           command=_opener).grid(row=0, column=2, rowspan=2,
                                                  padx=(0, 10))

    # ─── Find & Edit tab helpers ──────────────────────────────────────────

    def _selected_doc_path(self) -> str | None:
        name = self._edit_doc_var.get()
        for d in self._editable_docs:
            p = d.get("docx_path", d.get("file_path", ""))
            if Path(p).name == name:
                return p
        return None

    def _find_in_doc(self):
        path = self._selected_doc_path()
        find = self._find_entry.get().strip()
        if not path:
            self._find_status.configure(text="No document selected.")
            return
        if not find:
            self._find_status.configure(text="Enter text to search for.")
            return
        try:
            from src.documents.docx_writer import count_occurrences
            n = count_occurrences(path, find)
            self._find_status.configure(
                text=f"Found {n} occurrence(s) of \"{find}\" in {Path(path).name}."
            )
        except Exception as exc:
            self._find_status.configure(text=f"Error: {exc}")

    def _replace_all(self):
        path = self._selected_doc_path()
        find    = self._find_entry.get().strip()
        replace = self._replace_entry.get()
        if not path:
            messagebox.showwarning("No document", "Select a document first.")
            return
        if not find:
            messagebox.showwarning("No search text", "Enter text to find.")
            return
        if not Path(path).exists():
            messagebox.showerror("File not found", f"Cannot find: {path}")
            return
        try:
            from src.documents.docx_writer import find_replace_docx
            n = find_replace_docx(path, find, replace)
            self._find_status.configure(
                text=f"Replaced {n} occurrence(s) in {Path(path).name}."
            )
            messagebox.showinfo("Done", f"Replaced {n} occurrence(s).")
        except Exception as exc:
            messagebox.showerror("Replace failed", str(exc))

    def _open_selected_doc(self):
        path = self._selected_doc_path()
        if path and Path(path).exists():
            if sys.platform == "win32":
                os.startfile(path)
            else:
                import subprocess
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, path])
        else:
            messagebox.showwarning("No file", "No document selected or file not found.")

    # ─── AI post-processing fix ───────────────────────────────────────────

    def _apply_ai_fix(self, mode: str):
        _placeholder_start = "E.g."
        instruction = self._post_box.get("0.0", "end").strip()
        if not instruction or instruction.startswith(_placeholder_start):
            messagebox.showwarning(
                "No instruction",
                "Describe the change you want in the Post-Processing Fix box.")
            return

        if self._system is None:
            messagebox.showwarning("No run yet", "Run the analysis first.")
            return

        if mode == "selected":
            path = self._selected_doc_path()
            targets = [path] if path else []
        else:
            targets = [
                d.get("docx_path", d.get("file_path", ""))
                for d in self._editable_docs
                if d.get("doc_type") not in ("checklist",)
            ]
            targets = [t for t in targets if t]

        if not targets:
            messagebox.showwarning("No targets", "No documents to fix.")
            return

        self._run_btn.configure(state="disabled")
        self._progress_bar.start()
        self._tabs.set("Progress")
        self._log(f"\n  AI fix ({mode}) — {len(targets)} document(s)…")

        threading.Thread(
            target=self._ai_fix_thread,
            args=(self._system.config, targets, instruction),
            daemon=True,
        ).start()

    def _ai_fix_thread(self, config, targets: list, instruction: str):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for path in targets:
                if not path or not Path(path).exists():
                    self._log_q.put(("log", f"  Skipping missing file: {path}"))
                    continue
                self._log_q.put(("log", f"  Fixing: {Path(path).name}"))
                ext = Path(path).suffix.lower()
                if ext == ".docx":
                    from src.documents.docx_writer import ai_fix_docx
                    loop.run_until_complete(ai_fix_docx(config, path, instruction))
                else:
                    from main import _ai_edit_document
                    loop.run_until_complete(_ai_edit_document(config, path, instruction))
            self._log_q.put(("log", "  AI fix complete."))
        except Exception as exc:
            self._log_q.put(("log", f"  AI fix error: {exc}\n{traceback.format_exc()}"))
        finally:
            loop.close()
            self.after(0, self._ai_fix_done)

    # ─── Research tab logic ───────────────────────────────────────────────

    def _run_research(self):
        query = self._research_query_box.get("0.0", "end").strip()
        if not query:
            messagebox.showwarning("No query", "Enter a research question first.")
            return

        self._research_btn.configure(state="disabled", text="Searching…")
        self._research_status.configure(
            text="Running research — this may take 15–60 seconds…",
            text_color="gray55")
        self._research_result_box.configure(state="normal")
        self._research_result_box.delete("0.0", "end")
        self._research_result_box.configure(state="disabled")

        state_str  = self._research_state_var.get().strip().upper() or "FEDERAL"
        export_fmt = self._research_export_var.get()

        threading.Thread(
            target=self._research_thread,
            args=(query, state_str, export_fmt),
            daemon=True,
        ).start()

    def _research_thread(self, query, state_str, export_fmt):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from configs.config import USState, LegalDomain
            from src.legal_ai_system import create_legal_ai_system
            from main import _export_research

            async def _run():
                system = await create_legal_ai_system()
                self._system = self._system or system
                try:
                    state_enum = USState(state_str.upper())
                except ValueError:
                    state_enum = USState.FEDERAL
                return await system.research(
                    query=query,
                    state=state_enum,
                    domain=LegalDomain.CONTRACT,
                )

            result = loop.run_until_complete(_run())
            response = result.get("response", "")
            duration = result.get("duration_seconds", 0)
            cost     = result.get("total_cost", 0)

            if export_fmt != "none":
                try:
                    _export_research(result, query, state_str, "general", "standard", export_fmt)
                except Exception as exc:
                    response += f"\n\n[Export warning: {exc}]"

            self.after(0, lambda: self._research_done(response, duration, cost))
        except Exception as exc:
            err = f"{exc}\n{traceback.format_exc()}"
            self.after(0, lambda: self._research_error(err))
        finally:
            loop.close()

    def _research_done(self, response: str, duration: float, cost: float):
        self._research_btn.configure(state="normal", text="Run Research")
        self._research_status.configure(
            text=f"Done — {duration:.1f}s | Est. cost ${cost:.6f}",
            text_color="gray55")
        self._research_result_box.configure(state="normal")
        self._research_result_box.delete("0.0", "end")
        self._research_result_box.insert("0.0", response)
        self._research_result_box.configure(state="disabled")

    def _research_error(self, error_msg: str):
        self._research_btn.configure(state="normal", text="Run Research")
        self._research_status.configure(
            text=f"Error: {error_msg[:140]}", text_color="#e05050")

    # ─── Preset legal updates ─────────────────────────────────────────────

    def _apply_preset(self):
        preset_name = self._preset_var.get()
        base_instruction = self._PRESET_INSTRUCTIONS.get(preset_name, "")
        if not base_instruction:
            messagebox.showwarning("Unknown preset", f"No instruction found for: {preset_name}")
            return

        extra_notes = self._preset_notes_box.get("0.0", "end").strip()
        instruction = base_instruction
        if extra_notes:
            instruction += f"\n\nADDITIONAL INSTRUCTIONS FROM USER:\n{extra_notes}"

        if self._system is None:
            messagebox.showwarning("No run yet", "Run the analysis first.")
            return

        targets = [
            d.get("docx_path", d.get("file_path", ""))
            for d in self._editable_docs
            if d.get("doc_type") not in ("checklist",)
        ]
        targets = [t for t in targets if t]

        if not targets:
            messagebox.showwarning("No targets", "No documents to apply preset to. Run the analysis first.")
            return

        self._run_btn.configure(state="disabled")
        self._progress_bar.start()
        self._tabs.set("Progress")
        self._log(f"\n  Applying preset '{preset_name}' to {len(targets)} document(s)…")

        threading.Thread(
            target=self._ai_fix_thread,
            args=(self._system.config, targets, instruction),
            daemon=True,
        ).start()

    def _ai_fix_done(self):
        self._run_btn.configure(state="normal")
        self._progress_bar.stop()
        self._progress_bar.set(0)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = LegalAIApp()
    app.mainloop()


if __name__ == "__main__":
    main()

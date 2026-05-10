"""
Audio Stream Monitor — always-on real-time audio capture and transcription.

Architecture:
  • Captures audio in configurable chunks via sounddevice (or pyaudio fallback)
  • Transcribes via Google Speech-to-Text REST API (reuses existing GOOGLE_API_KEY)
    OR faster-whisper (local, no API needed) if available
  • Detects legal keywords and triggers alerts
  • Feeds transcribed text to LiveLawyerServer for push to connected devices

Dependencies (install as needed):
  pip install sounddevice numpy
  Optional: pip install faster-whisper   (local transcription, no API quota)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, List, Optional

SAMPLE_RATE   = 16000   # Hz — optimal for speech recognition
CHANNELS      = 1
CHUNK_SECONDS = 5       # Capture this many seconds before transcribing

# Legal keywords that trigger an immediate alert
LEGAL_KEYWORDS = [
    "objection", "sustained", "overruled", "motion", "order", "ruling",
    "contempt", "perjury", "exhibit", "hearsay", "relevance", "sidebar",
    "recess", "stipulate", "stipulation", "deposition", "subpoena",
    "habeas", "mandamus", "injunction", "appeal", "judgment", "verdict",
    "settlement", "damages", "liability", "negligence", "breach", "statute",
    "constitutional", "due process", "equal protection", "miranda",
]


@dataclass
class AudioSegment:
    timestamp:  str
    duration_s: float
    text:       str
    confidence: float
    keywords_detected: List[str] = field(default_factory=list)
    alert_level: str = "normal"   # "normal" | "alert" | "urgent"


class AudioMonitor:
    """
    Always-on audio capture and real-time transcription.

    Usage:
        monitor = AudioMonitor(google_api_key="...", on_segment=my_callback)
        await monitor.start()
        # ... later:
        await monitor.stop()
    """

    def __init__(
        self,
        google_api_key: str = "",
        on_segment: Optional[Callable[[AudioSegment], Any]] = None,
        chunk_seconds: int = CHUNK_SECONDS,
        use_local_whisper: bool = False,
    ):
        self._api_key        = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        self._on_segment     = on_segment
        self._chunk_seconds  = chunk_seconds
        self._use_local      = use_local_whisper
        self._running        = False
        self._task: Optional[asyncio.Task] = None

        # Check for sounddevice
        try:
            import sounddevice as sd
            import numpy as np
            self._sd  = sd
            self._np  = np
            self._has_audio = True
        except ImportError:
            self._has_audio = False

        # Check for faster-whisper
        self._whisper_model = None
        if use_local_whisper:
            try:
                from faster_whisper import WhisperModel
                self._whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                print("  [AudioMonitor] faster-whisper loaded (local transcription)")
            except ImportError:
                print("  [AudioMonitor] faster-whisper not installed — falling back to Google API")

    @property
    def available(self) -> bool:
        return self._has_audio

    async def start(self) -> bool:
        """Start the audio monitoring loop. Returns True if started successfully."""
        if not self._has_audio:
            print("  [AudioMonitor] sounddevice not installed. "
                  "Run: pip install sounddevice numpy")
            return False
        if self._running:
            return True

        self._running = True
        self._task    = asyncio.create_task(self._capture_loop())
        print(f"  [AudioMonitor] Started — capturing {self._chunk_seconds}s chunks at {SAMPLE_RATE} Hz")
        return True

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("  [AudioMonitor] Stopped.")

    # ─────────────────────────────────────────────────────────────────────

    async def _capture_loop(self) -> None:
        """Main capture loop — runs until self._running is False."""
        while self._running:
            try:
                audio_bytes = await asyncio.to_thread(self._record_chunk)
                if audio_bytes:
                    segment = await self._transcribe(audio_bytes)
                    if segment and segment.text.strip():
                        await self._emit(segment)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"  [AudioMonitor] Capture error: {exc}")
                await asyncio.sleep(1)

    def _record_chunk(self) -> Optional[bytes]:
        """Blocking: record CHUNK_SECONDS of audio and return WAV bytes."""
        sd  = self._sd
        np  = self._np
        try:
            frames = int(SAMPLE_RATE * self._chunk_seconds)
            recording = sd.rec(frames, samplerate=SAMPLE_RATE,
                               channels=CHANNELS, dtype="int16")
            sd.wait()   # blocks until done
            # Convert numpy array → WAV bytes
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)           # int16 = 2 bytes
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(recording.tobytes())
            return buf.getvalue()
        except Exception as exc:
            print(f"  [AudioMonitor] Record error: {exc}")
            return None

    async def _transcribe(self, wav_bytes: bytes) -> Optional[AudioSegment]:
        """Transcribe WAV bytes via local whisper or Google Speech REST API."""
        text       = ""
        confidence = 0.0
        ts         = datetime.utcnow().isoformat()

        if self._whisper_model:
            text, confidence = await asyncio.to_thread(
                self._transcribe_whisper, wav_bytes
            )
        elif self._api_key:
            text, confidence = await self._transcribe_google(wav_bytes)
        else:
            return None   # No transcription backend available

        # Detect legal keywords
        text_lower = text.lower()
        detected   = [k for k in LEGAL_KEYWORDS if k in text_lower]
        alert      = "normal"
        if any(k in ("objection", "contempt", "perjury", "verdict", "ruling") for k in detected):
            alert = "urgent"
        elif detected:
            alert = "alert"

        return AudioSegment(
            timestamp=ts,
            duration_s=self._chunk_seconds,
            text=text,
            confidence=confidence,
            keywords_detected=detected,
            alert_level=alert,
        )

    def _transcribe_whisper(self, wav_bytes: bytes) -> tuple:
        """Local Whisper transcription (blocking)."""
        buf = io.BytesIO(wav_bytes)
        segments, _ = self._whisper_model.transcribe(buf, language="en")
        text = " ".join(s.text for s in segments)
        return text.strip(), 0.9

    async def _transcribe_google(self, wav_bytes: bytes) -> tuple:
        """Google Speech-to-Text REST API transcription."""
        import urllib.request
        import urllib.error

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        payload   = json.dumps({
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": SAMPLE_RATE,
                "languageCode": "en-US",
                "model": "latest_long",
                "enableAutomaticPunctuation": True,
            },
            "audio": {"content": audio_b64},
        }).encode("utf-8")

        url = f"https://speech.googleapis.com/v1/speech:recognize?key={self._api_key}"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data       = json.loads(resp.read().decode("utf-8"))
                results    = data.get("results", [])
                if not results:
                    return "", 0.0
                alt        = results[0]["alternatives"][0]
                text       = alt.get("transcript", "")
                confidence = float(alt.get("confidence", 0.8))
                return text, confidence
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            print(f"  [AudioMonitor] Google STT error: {exc}")
            return "", 0.0

    async def _emit(self, segment: AudioSegment) -> None:
        """Call the registered callback with the new segment."""
        if not self._on_segment:
            return
        try:
            result = self._on_segment(segment)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            print(f"  [AudioMonitor] Callback error: {exc}")

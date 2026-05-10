"""
Live Lawyer Server — real-time bidirectional communication between the AI Lawyer
and any connected device (phone, tablet, laptop).

Architecture:
  • WebSocket server (default port 8765) — any device can connect
  • REST-style commands over WebSocket for structured requests
  • Always-on audio monitoring loop (optional) — transcribes everything happening
    in person and pushes legal keyword alerts to all connected devices
  • AI Lawyer advisor: post any situation update; get instant strategic guidance

Connecting from any device:
  ws://YOUR_IP:8765

  Send JSON: {"type": "query", "text": "What do I say when they allege X?"}
  Receive JSON: {"type": "response", "text": "..."}

  For audio alerts (server → device):
  {"type": "audio_alert", "text": "...", "keywords": [...], "level": "urgent"}

Run from command line:
  python -m src.live.live_lawyer

Or programmatically:
  server = LiveLawyerServer(config)
  await server.start()
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from configs.config import LegalAIConfig


@dataclass
class ConnectedDevice:
    device_id: str
    address:   str
    connected_at: str
    ws: Any          # websocket connection object


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE LAWYER SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class LiveLawyerServer:
    """
    WebSocket-based real-time communication server for the AI Lawyer.

    Usage:
        server = LiveLawyerServer(config)
        server.set_case_context(strategy_plan.as_prompt_block(), case_law_context)
        await server.start(port=8765)
    """

    def __init__(self, config: LegalAIConfig):
        self._config   = config
        self._devices: Dict[str, ConnectedDevice] = {}
        self._llm: Optional[Any] = None
        self._case_context = ""
        self._case_law     = ""
        self._audio_monitor: Optional[Any] = None
        self._server_task: Optional[asyncio.Task] = None
        self._running = False

    def set_case_context(self, strategy_plan: str, case_law_context: str = "") -> None:
        """Inject the current case's strategy plan so the AI advisor has context."""
        self._case_context = strategy_plan
        self._case_law     = case_law_context

    def set_llm_client(self, llm_client: Any) -> None:
        self._llm = llm_client

    async def start(
        self,
        port: int = 8765,
        enable_audio: bool = False,
        audio_chunk_seconds: int = 5,
    ) -> None:
        """
        Start the WebSocket server and optionally the audio monitor.

        Args:
            port:                WebSocket port to listen on.
            enable_audio:        If True, starts the always-on audio monitor.
            audio_chunk_seconds: Seconds per audio capture chunk.
        """
        try:
            import websockets
        except ImportError:
            print("  [LiveLawyer] websockets not installed. Run: pip install websockets")
            return

        self._running = True

        # Start audio monitor if requested
        if enable_audio:
            await self._start_audio_monitor(audio_chunk_seconds)

        local_ip = self._get_local_ip()
        print(f"\n  [LiveLawyer] Server starting on ws://{local_ip}:{port}")
        print(f"  [LiveLawyer] Connect any device to: ws://{local_ip}:{port}")
        print(f"  [LiveLawyer] Audio monitor: {'ON' if enable_audio else 'OFF'}")

        self._server_task = asyncio.create_task(
            self._serve(websockets, port)
        )

    async def stop(self) -> None:
        self._running = False
        if self._audio_monitor:
            await self._audio_monitor.stop()
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        print("  [LiveLawyer] Server stopped.")

    async def push_to_all(self, message: Dict[str, Any]) -> None:
        """Push a JSON message to all connected devices."""
        if not self._devices:
            return
        payload = json.dumps(message)
        dead: List[str] = []
        for dev_id, device in list(self._devices.items()):
            try:
                await device.ws.send(payload)
            except Exception:
                dead.append(dev_id)
        for dev_id in dead:
            self._devices.pop(dev_id, None)

    # ─────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────

    async def _serve(self, websockets: Any, port: int) -> None:
        async with websockets.serve(self._handle_connection, "0.0.0.0", port):
            print(f"  [LiveLawyer] Listening on port {port}...")
            while self._running:
                await asyncio.sleep(0.5)

    async def _handle_connection(self, ws: Any, path: str = "") -> None:
        """Handle a single WebSocket connection from a device."""
        addr = str(ws.remote_address if hasattr(ws, "remote_address") else "unknown")
        dev_id = f"device_{int(time.time() * 1000)}"
        device = ConnectedDevice(
            device_id=dev_id,
            address=addr,
            connected_at=datetime.utcnow().isoformat(),
            ws=ws,
        )
        self._devices[dev_id] = device
        print(f"  [LiveLawyer] Device connected: {addr} (id={dev_id})")

        # Welcome message
        await ws.send(json.dumps({
            "type": "welcome",
            "device_id": dev_id,
            "message": "RAPCorp Live Lawyer connected. Send {\"type\": \"query\", \"text\": \"...\"} for advice.",
            "case_context_loaded": bool(self._case_context),
            "audio_monitor": self._audio_monitor is not None,
        }))

        try:
            async for raw in ws:
                await self._handle_message(raw, device)
        except Exception:
            pass
        finally:
            self._devices.pop(dev_id, None)
            print(f"  [LiveLawyer] Device disconnected: {addr}")

    async def _handle_message(self, raw: str, device: ConnectedDevice) -> None:
        """Route an incoming message from a device."""
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            await device.ws.send(json.dumps({"type": "error", "text": "Invalid JSON"}))
            return

        msg_type = msg.get("type", "query")

        if msg_type == "query":
            response = await self._answer_query(msg.get("text", ""))
            await device.ws.send(json.dumps({"type": "response", "text": response}))

        elif msg_type == "update_context":
            self._case_context = msg.get("strategy_plan", self._case_context)
            self._case_law     = msg.get("case_law", self._case_law)
            await device.ws.send(json.dumps({"type": "ack", "text": "Context updated."}))

        elif msg_type == "ping":
            await device.ws.send(json.dumps({
                "type": "pong",
                "connected_devices": len(self._devices),
                "audio_on": self._audio_monitor is not None,
            }))

        elif msg_type == "status":
            await device.ws.send(json.dumps({
                "type": "status",
                "connected_devices": len(self._devices),
                "case_context_loaded": bool(self._case_context),
                "audio_monitor": self._audio_monitor is not None,
            }))

        else:
            await device.ws.send(json.dumps({
                "type": "error",
                "text": f"Unknown message type: {msg_type}",
            }))

    async def _answer_query(self, question: str) -> str:
        """Use the AI Lawyer to answer a real-time question from a connected device."""
        if not self._llm:
            return "AI Lawyer not available — LLM not initialized."
        if not question.strip():
            return "Please provide a question."

        prompt = f"""You are an AI Lawyer advising a client IN REAL TIME during or around a legal proceeding.
The client has sent you an urgent message from their device. Give them a concise, actionable answer
in plain language (not legal jargon). Maximum 200 words.

CASE STRATEGY CONTEXT:
{self._case_context[:1500] if self._case_context else "No case context loaded."}

RELEVANT CASE LAW:
{self._case_law[:500] if self._case_law else ""}

CLIENT'S REAL-TIME QUESTION:
{question}

Answer directly and practically. If they need to say something aloud in court, give them the
exact words. If it's a strategy question, give a crisp recommendation."""

        try:
            result = await self._llm.generate(prompt=prompt, task="legal_synthesis")
            return (result.get("text") or "No response generated.").strip()
        except Exception as exc:
            return f"Error generating response: {exc}"

    async def _start_audio_monitor(self, chunk_seconds: int) -> None:
        try:
            from src.live.audio_stream import AudioMonitor
            self._audio_monitor = AudioMonitor(
                google_api_key=self._config.google_api_key,
                on_segment=self._on_audio_segment,
                chunk_seconds=chunk_seconds,
            )
            ok = await self._audio_monitor.start()
            if not ok:
                self._audio_monitor = None
        except Exception as exc:
            print(f"  [LiveLawyer] Audio monitor failed to start: {exc}")
            self._audio_monitor = None

    async def _on_audio_segment(self, segment: Any) -> None:
        """Called by AudioMonitor when a new transcribed segment is ready."""
        # Always push transcript to connected devices
        msg: Dict[str, Any] = {
            "type": "transcript",
            "timestamp": segment.timestamp,
            "text": segment.text,
            "confidence": segment.confidence,
            "keywords": segment.keywords_detected,
            "level": segment.alert_level,
        }

        if segment.alert_level in ("alert", "urgent"):
            msg["type"] = "audio_alert"
            # Generate AI advice for urgent keywords
            if self._llm and segment.keywords_detected:
                advice = await self._answer_query(
                    f"I just heard this in the courtroom: \"{segment.text}\". "
                    f"Keywords detected: {', '.join(segment.keywords_detected)}. "
                    f"What should I do RIGHT NOW?"
                )
                msg["ai_advice"] = advice

        await self.push_to_all(msg)

        if segment.keywords_detected:
            print(f"  [LiveLawyer] ALERT [{segment.alert_level}]: {segment.text[:100]}")
            print(f"               Keywords: {segment.keywords_detected}")

    @staticmethod
    def _get_local_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "localhost"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def _main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from configs.config import create_config
    from src.core.gemini_client import GeminiClient

    config = create_config()
    llm    = GeminiClient(config)

    port         = int(os.getenv("LIVE_LAWYER_PORT", "8765"))
    enable_audio = os.getenv("LIVE_LAWYER_AUDIO", "0") == "1"

    server = LiveLawyerServer(config)
    server.set_llm_client(llm)

    print("=" * 60)
    print("  RAPCorp Live Lawyer — Real-Time Device Server")
    print("=" * 60)

    await server.start(port=port, enable_audio=enable_audio)

    try:
        print(f"  Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(_main())

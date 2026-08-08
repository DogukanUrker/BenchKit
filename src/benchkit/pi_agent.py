"""Stock Pi coding-agent RPC adapter running inside a Docker task sandbox."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from benchkit.client import GenerationUpdate, InferenceClient
from benchkit.sandbox import DockerTaskEnvironment, LatestPiImage


def _message_blocks(message: object, kind: str, field: str) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get(field) or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == kind
    )


@dataclass
class _RpcTrace:
    """Incrementally normalize Pi events into BenchKit's generation shape."""

    final_response: str = ""
    thinking_parts: list[str] = field(default_factory=list)
    output_tokens: int = 0
    input_tokens: int = 0
    turns: int = 0
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    done_reason: str = "stop"
    _active_tools: dict[str, tuple[float, dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )

    def handle(self, event: dict[str, Any]) -> tuple[str, str]:
        """Consume one RPC event and return streamed thinking/text deltas."""
        event_type = event.get("type")
        thinking_delta = ""
        text_delta = ""
        if event_type == "message_update":
            delta = event.get("assistantMessageEvent") or {}
            if delta.get("type") == "thinking_delta":
                thinking_delta = str(delta.get("delta") or "")
            elif delta.get("type") == "text_delta":
                text_delta = str(delta.get("delta") or "")
        elif event_type == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                self.turns += 1
                text = _message_blocks(message, "text", "text")
                if text:
                    self.final_response = text
                thinking = _message_blocks(message, "thinking", "thinking")
                if thinking:
                    self.thinking_parts.append(thinking)
                usage = message.get("usage") or {}
                self.input_tokens += int(usage.get("input") or 0)
                self.output_tokens += int(usage.get("output") or 0)
                self.done_reason = str(message.get("stopReason") or self.done_reason)
        elif event_type == "tool_execution_start":
            tool_id = str(event.get("toolCallId") or "")
            item = {
                "name": str(event.get("toolName") or ""),
                "arguments": event.get("args") or {},
            }
            self._active_tools[tool_id] = (time.monotonic(), item)
        elif event_type == "tool_execution_end":
            tool_id = str(event.get("toolCallId") or "")
            started, item = self._active_tools.pop(
                tool_id,
                (time.monotonic(), {"name": str(event.get("toolName") or "")}),
            )
            item.update(
                is_error=bool(event.get("isError")),
                duration_s=round(time.monotonic() - started, 3),
            )
            output = _message_blocks(event.get("result"), "text", "text")
            if output:
                item["output_truncated"] = len(output) > 8000
                item["output"] = output[-8000:]
            self.tool_trace.append(item)
        return thinking_delta, text_delta


@dataclass
class PiAgentRunner:
    """Run the original latest Pi agent, without replacing any native tools."""

    client: InferenceClient
    image: LatestPiImage = field(default_factory=LatestPiImage)

    @property
    def version(self) -> str:
        return self.image.version

    def prepare(self) -> str:
        return self.image.prepare()

    def cleanup(self) -> None:
        self.image.cleanup()

    def generate(
        self,
        model: str,
        prompt: str,
        on_progress=None,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        self.prepare()
        started = time.perf_counter()
        try:
            timeout = max(0.1, float(os.environ.get("BENCHKIT_PI_TIMEOUT", "900")))
        except ValueError:
            timeout = 900.0
        trace = _RpcTrace()
        stderr_parts: list[str] = []
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        process: subprocess.Popen[str] | None = None
        timed_out = False
        cancelled = False

        try:
            with DockerTaskEnvironment(self.client, model, self.image) as environment:
                process = environment.start_pi()
                assert process.stdin is not None
                assert process.stdout is not None
                assert process.stderr is not None

                def read_stdout() -> None:
                    for line in process.stdout:
                        try:
                            events.put(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    events.put(None)

                def read_stderr() -> None:
                    for line in process.stderr:
                        stderr_parts.append(line)
                        if sum(map(len, stderr_parts)) > 16000:
                            del stderr_parts[: len(stderr_parts) // 2]

                threading.Thread(target=read_stdout, daemon=True).start()
                threading.Thread(target=read_stderr, daemon=True).start()

                send_lock = threading.Lock()

                def send(payload: dict[str, Any]) -> None:
                    with send_lock:
                        if process is None or process.stdin is None:
                            return
                        try:
                            process.stdin.write(json.dumps(payload) + "\n")
                            process.stdin.flush()
                        except OSError:
                            return

                send({"id": "benchkit-prompt", "type": "prompt", "message": prompt})
                requested_summary = False
                summary_responses: set[str] = set()
                stats: dict[str, Any] = {}
                last_text: str | None = None
                while True:
                    elapsed = time.perf_counter() - started
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        send({"type": "abort"})
                        break
                    if elapsed >= timeout:
                        timed_out = True
                        send({"type": "abort"})
                        break
                    try:
                        event = events.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if event is None:
                        break

                    thinking_delta, text_delta = trace.handle(event)
                    if on_progress is not None and (thinking_delta or text_delta):
                        on_progress(
                            GenerationUpdate(
                                thinking=thinking_delta,
                                response=text_delta,
                                elapsed_s=elapsed,
                                reasoning_channel_seen=bool(
                                    thinking_delta or trace.thinking_parts
                                ),
                            )
                        )

                    if event.get("type") == "agent_settled" and not requested_summary:
                        requested_summary = True
                        send({"id": "benchkit-stats", "type": "get_session_stats"})
                        send(
                            {
                                "id": "benchkit-final",
                                "type": "get_last_assistant_text",
                            }
                        )
                        continue
                    if event.get("type") == "response":
                        event_id = event.get("id")
                        if event_id == "benchkit-prompt" and not event.get("success"):
                            detail = event.get("error") or "Pi rejected the prompt"
                            raise RuntimeError(str(detail))
                        data = event.get("data") or {}
                        if event_id == "benchkit-stats":
                            stats = data
                            summary_responses.add("stats")
                        elif event_id == "benchkit-final":
                            value = data.get("text")
                            last_text = str(value) if value is not None else None
                            summary_responses.add("final")
                        if requested_summary and summary_responses == {
                            "stats",
                            "final",
                        }:
                            break

                if last_text:
                    trace.final_response = last_text
                tokens = stats.get("tokens") or {}
                if tokens:
                    trace.input_tokens = sum(
                        int(tokens.get(key) or 0)
                        for key in ("input", "cacheRead", "cacheWrite")
                    )
                    trace.output_tokens = int(tokens.get("output") or 0)
                trace.turns = int(stats.get("assistantMessages") or trace.turns)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=3)

        elapsed = time.perf_counter() - started
        if on_progress is not None and not cancelled:
            on_progress(
                GenerationUpdate(
                    elapsed_s=elapsed,
                    reasoning_channel_seen=bool(trace.thinking_parts),
                    done=True,
                )
            )
        thinking = "\n\n".join(trace.thinking_parts)
        error = "".join(stderr_parts).strip()
        if not trace.final_response and not (cancelled or timed_out):
            detail = error[-2000:] if error else "no assistant message was returned"
            raise RuntimeError(f"Pi agent exited without a response: {detail}")
        output_tokens = trace.output_tokens
        return {
            "thinking": thinking,
            "response": trace.final_response,
            "trace_status": "observed" if thinking else "unavailable",
            "tok_s": output_tokens / elapsed if elapsed > 0 else 0.0,
            "eval_count": output_tokens,
            "eval_duration_ns": int(elapsed * 1e9),
            "response_time_s": elapsed,
            "done_reason": (
                "cancelled"
                if cancelled
                else "timeout"
                if timed_out
                else trace.done_reason
            ),
            "timed_out": timed_out,
            "timeout_s": timeout,
            "cancelled": cancelled,
            "harness": "pi",
            "harness_version": self.version,
            "input_tokens": trace.input_tokens,
            "model_turns": trace.turns,
            "tool_calls": len(trace.tool_trace),
            "tool_trace": trace.tool_trace,
        }

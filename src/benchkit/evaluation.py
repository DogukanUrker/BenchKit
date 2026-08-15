"""Verifier results and answer-safe repair prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

from benchkit.benchmarks.base import Task


@dataclass(frozen=True)
class EvaluationResult:
    """Normalized benchmark score plus feedback safe to return to a model."""

    score: float
    feedback: str = ""
    error: str = ""
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.score >= 1.0 and not self.error


def _score(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, Real):
        return min(1.0, max(0.0, float(value)))
    return 1.0 if bool(value) else 0.0


def evaluate_response(bench: object, task: Task, response: str) -> EvaluationResult:
    """Evaluate one response and add generic feedback when needed."""
    detailed = getattr(bench, "evaluate_with_feedback", None)
    raw = (
        detailed(task, response)
        if callable(detailed)
        else bench.evaluate(task, response)
    )
    if isinstance(raw, EvaluationResult):
        result = EvaluationResult(
            score=_score(raw.score),
            feedback=raw.feedback.strip(),
            error=raw.error.strip(),
            details=dict(raw.details),
        )
    else:
        result = EvaluationResult(score=_score(raw))

    if result.passed or result.error or result.feedback:
        return result
    if result.score > 0:
        feedback = (
            f"The verifier awarded {result.score * 100:.1f}% partial credit. "
            "Re-check the requirements and replace the answer with a corrected one."
        )
    else:
        feedback = (
            "The verifier marked the previous answer incorrect. Re-check the "
            "reasoning and replace it with a corrected answer."
        )
    return EvaluationResult(
        score=result.score,
        feedback=feedback,
        details=dict(result.details),
    )


def repair_message(feedback: str, attempt: int, total: int) -> str:
    """Build one follow-up user turn without exposing verifier secrets."""
    remaining = total - attempt
    suffix = (
        "This is the only repair attempt."
        if total == 1
        else f"You have {remaining} repair attempt(s) remaining after this one."
    )
    return (
        "BenchKit verifier feedback:\n"
        f"{feedback.strip()}\n\n"
        f"{suffix} Use the available tools if helpful, then return a complete "
        "replacement answer for the original task."
    )


def standalone_repair_prompt(
    original_prompt: str,
    previous_response: str,
    feedback: str,
    attempt: int,
    total: int,
) -> str:
    """Render a follow-up transcript for stateless direct inference APIs."""
    return (
        "Continue the following benchmark exchange.\n\n"
        "Original user request:\n"
        f"{original_prompt}\n\n"
        "Previous assistant answer:\n"
        f"{previous_response}\n\n"
        f"{repair_message(feedback, attempt, total)}"
    )


def combine_attempts(
    attempts: list[tuple[dict, EvaluationResult]],
    feedback_sent: list[str],
) -> dict:
    """Combine multiple model calls into one final generation result."""
    if not attempts:
        raise ValueError("at least one generation attempt is required")

    final = dict(attempts[-1][0])
    total_output = sum(int(gen.get("eval_count") or 0) for gen, _ in attempts)
    total_input = sum(int(gen.get("input_tokens") or 0) for gen, _ in attempts)
    generation_ns = sum(int(gen.get("eval_duration_ns") or 0) for gen, _ in attempts)
    response_time = sum(float(gen.get("response_time_s") or 0.0) for gen, _ in attempts)
    traces: list[dict] = []
    for number, (gen, _) in enumerate(attempts, start=1):
        traces.extend(
            {**call, "attempt": number} for call in gen.get("tool_trace") or []
        )

    first_evaluation = attempts[0][1]
    final_evaluation = attempts[-1][1]
    attempt_records = []
    for number, (gen, evaluation) in enumerate(attempts, start=1):
        attempt_records.append(
            {
                "attempt": number,
                "response": str(gen.get("response") or ""),
                "thinking": str(gen.get("thinking") or ""),
                "score": round(evaluation.score * 100, 1),
                "passed": evaluation.passed,
                "feedback": evaluation.feedback,
                "error": evaluation.error,
                "input_tokens": int(gen.get("input_tokens") or 0),
                "output_tokens": int(gen.get("eval_count") or 0),
                "response_time_s": round(float(gen.get("response_time_s") or 0.0), 3),
                "model_turns": int(gen.get("model_turns") or 1),
                "tool_calls": int(gen.get("tool_calls") or 0),
                "done_reason": str(gen.get("done_reason") or ""),
                "length_exceeded": bool(gen.get("length_exceeded")),
            }
        )

    final.update(
        tok_s=(
            total_output / (generation_ns / 1e9)
            if generation_ns > 0
            else total_output / response_time
            if response_time > 0
            else 0.0
        ),
        eval_count=total_output,
        eval_duration_ns=generation_ns,
        response_time_s=response_time,
        input_tokens=total_input,
        model_turns=sum(int(gen.get("model_turns") or 1) for gen, _ in attempts),
        tool_calls=sum(int(gen.get("tool_calls") or 0) for gen, _ in attempts),
        tool_trace=traces,
        attempts=attempt_records,
        repair_attempts_used=len(attempts) - 1,
        repair_feedback=list(feedback_sent),
        first_attempt_score=first_evaluation.score,
        evaluation_score=final_evaluation.score,
        evaluation_feedback=final_evaluation.feedback,
        evaluation_error=final_evaluation.error,
        evaluation_details=dict(final_evaluation.details),
        repaired=(not first_evaluation.passed and final_evaluation.passed),
        trace_status=(
            "observed"
            if any(gen.get("trace_status") == "observed" for gen, _ in attempts)
            else "available_empty"
            if any(gen.get("trace_status") == "available_empty" for gen, _ in attempts)
            else str(final.get("trace_status") or "unavailable")
        ),
    )
    return final

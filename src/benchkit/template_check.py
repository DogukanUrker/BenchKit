"""Lightweight chat-template checks for llama.cpp-compatible servers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote

import httpx

from benchkit.client import InferenceClient, _without_v1

Level = Literal["pass", "warn", "fail"]

MARKERS = {
    "system": "BK_SYSTEM_7F3A",
    "user_1": "BK_USER_11B2",
    "assistant": "BK_ASSISTANT_22C4",
    "user_2": "BK_USER_33D6",
}
PROBE = [
    {"role": "system", "content": MARKERS["system"]},
    {"role": "user", "content": MARKERS["user_1"]},
    {"role": "assistant", "content": MARKERS["assistant"]},
    {"role": "user", "content": MARKERS["user_2"]},
]

SPECIAL_TOKEN_RE = re.compile(
    r"<\|[^|<>\s]{1,64}\|>|</?[a-z][a-z0-9_]{0,63}>|\[/?INST\]|\[/?SYSTEM_PROMPT\]",
    re.IGNORECASE,
)
NONDETERMINISTIC_RE = re.compile(
    r"strftime_now|\bnow\s*\(|datetime\.|random",
    re.IGNORECASE,
)


class NativeEndpointUnavailable(RuntimeError):
    """Raised when the server does not expose llama.cpp native endpoints."""


@dataclass(frozen=True)
class TemplateCheck:
    name: str
    level: Level
    detail: str = ""


@dataclass
class TemplateReport:
    model: str
    checks: list[TemplateCheck] = field(default_factory=list)
    unavailable: bool = False

    def add(self, name: str, level: Level, detail: str = "") -> None:
        self.checks.append(TemplateCheck(name, level, detail))

    @property
    def status(self) -> str:
        if self.unavailable:
            return "unavailable"
        if any(check.level == "fail" for check in self.checks):
            return "fail"
        if any(check.level == "warn" for check in self.checks):
            return "warn"
        return "pass"

    @property
    def detail(self) -> str:
        problems = [
            check.detail or check.name
            for check in self.checks
            if check.level in {"warn", "fail"}
        ]
        return "; ".join(problems) or "All checks passed"


class _NativeAPI:
    """Route native requests through llama-swap or a direct llama.cpp server."""

    def __init__(self, client: InferenceClient, model: str) -> None:
        self.client = client
        self.model = model
        self.base = _without_v1(client.host)
        self.route: Literal["upstream", "direct"] | None = None

    def _url(self, route: str, path: str) -> str:
        if route == "upstream":
            model = quote(self.model, safe="")
            return f"{self.base}/upstream/{model}{path}"
        return f"{self.base}{path}"

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        routes = [self.route] if self.route else ["upstream", "direct"]
        last_response: httpx.Response | None = None
        for route in routes:
            payload = dict(body or {})
            params = None
            if route == "direct":
                payload.setdefault("model", self.model)
                if method == "GET":
                    params = {"model": self.model}
            response = httpx.request(
                method,
                self._url(route, path),
                headers=self.client._headers(),
                params=params,
                json=payload,
                timeout=min(self.client.timeout, 120),
            )
            last_response = response
            if response.status_code in {404, 405}:
                if self.route is None:
                    continue
                raise NativeEndpointUnavailable(
                    f"{path} is not exposed by this server"
                )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"{path} returned a non-object response")
            self.route = route
            return data

        detail = "native llama.cpp endpoints are not exposed"
        if last_response is not None and last_response.status_code not in {404, 405}:
            detail = f"HTTP {last_response.status_code}"
        raise NativeEndpointUnavailable(detail)

    def props(self) -> dict:
        return self.request("GET", "/props")

    def render(self, messages: list[dict], *, generation_prompt: bool) -> str:
        data = self.request(
            "POST",
            "/apply-template",
            {
                "messages": messages,
                "add_generation_prompt": generation_prompt,
            },
        )
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("/apply-template response has no prompt")
        return prompt

    def tokenize(self, text: str, *, add_special: bool) -> list:
        data = self.request(
            "POST",
            "/tokenize",
            {
                "content": text,
                "add_special": add_special,
                "parse_special": True,
                "with_pieces": False,
            },
        )
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("/tokenize response has no token list")
        return tokens


def _template_from_props(props: dict) -> str:
    template = props.get("chat_template")
    if isinstance(template, str):
        return template
    defaults = props.get("default_generation_settings")
    if isinstance(defaults, dict):
        template = defaults.get("chat_template")
        if isinstance(template, str):
            return template
    return ""


def check_template(client: InferenceClient, model: str) -> TemplateReport:
    """Run reference-free structural checks for one served model."""
    report = TemplateReport(model)
    if client.provider != "openai":
        report.unavailable = True
        report.add("native endpoints", "warn", "requires llama.cpp or llama-swap")
        return report

    api = _NativeAPI(client, model)
    template = ""
    try:
        template = _template_from_props(api.props())
    except NativeEndpointUnavailable:
        # Some builds expose /apply-template without /props. Rendering remains
        # useful, so only declare the feature unavailable if that also fails.
        pass
    except Exception as exc:
        report.add("template metadata", "warn", str(exc))

    report.add(
        "template exposed",
        "pass" if template else "warn",
        f"{len(template)} characters" if template else "template source not reported",
    )

    try:
        rendered = api.render(PROBE, generation_prompt=True)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip().replace("\n", " ")[:160]
        report.add(
            "render probe",
            "fail",
            detail or f"HTTP {exc.response.status_code}",
        )
        return report
    except NativeEndpointUnavailable as exc:
        report.unavailable = True
        report.add("render probe", "warn", str(exc))
        return report
    except Exception as exc:
        report.add("render probe", "fail", f"{type(exc).__name__}: {exc}")
        return report
    report.add("render probe", "pass", f"{len(rendered)} characters")

    try:
        repeated = api.render(PROBE, generation_prompt=True)
        if repeated != rendered:
            report.add(
                "deterministic render",
                "fail",
                "same messages rendered differently",
            )
        elif template and NONDETERMINISTIC_RE.search(template):
            report.add(
                "deterministic render",
                "warn",
                "template contains time or random logic",
            )
        else:
            report.add("deterministic render", "pass")
    except Exception as exc:
        report.add("deterministic render", "warn", str(exc))

    positions = {name: rendered.find(marker) for name, marker in MARKERS.items()}
    missing = [name for name, position in positions.items() if position < 0]
    if missing:
        report.add("messages preserved", "fail", f"missing: {', '.join(missing)}")
    elif list(positions.values()) != sorted(positions.values()):
        report.add("messages preserved", "fail", "message order changed")
    elif any(rendered.count(marker) != 1 for marker in MARKERS.values()):
        report.add("messages preserved", "fail", "a message was duplicated")
    else:
        report.add("messages preserved", "pass")

    tail_start = positions["user_2"] + len(MARKERS["user_2"])
    if positions["user_2"] >= 0 and rendered[tail_start:].strip():
        report.add("generation prompt", "pass")
    else:
        report.add(
            "generation prompt",
            "fail",
            "nothing follows the final user message",
        )

    specials = sorted(set(SPECIAL_TOKEN_RE.findall(rendered)))
    try:
        split = [
            token
            for token in specials
            if len(api.tokenize(token, add_special=False)) != 1
        ]
        if split:
            report.add(
                "special tokens",
                "fail",
                f"split by tokenizer: {', '.join(split[:3])}",
            )
        else:
            report.add("special tokens", "pass", f"{len(specials)} checked")

        empty = api.tokenize("", add_special=True)
        prompt_tokens = api.tokenize(rendered, add_special=True)
        double_bos = (
            empty
            and len(prompt_tokens) > 1
            and prompt_tokens[:2] == [empty[0], empty[0]]
        )
        if double_bos:
            report.add("single BOS", "fail", f"token {empty[0]} appears twice")
        else:
            report.add("single BOS", "pass")
    except NativeEndpointUnavailable:
        report.add("tokenization checks", "warn", "/tokenize is not exposed")
    except Exception as exc:
        report.add("tokenization checks", "warn", str(exc))

    return report

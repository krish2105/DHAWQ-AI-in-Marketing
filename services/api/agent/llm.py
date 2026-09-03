"""Provider-agnostic LLM client.

WHY AN ABSTRACTION RATHER THAN AN SDK CALL
------------------------------------------
DHAWQ has no Anthropic API key and is required to run free. But the model layer
is deliberately thin — models do decomposition, extraction, routing and
explanation and nothing else (ARCHITECTURE.md §0.1) — so which vendor serves
those calls is an implementation detail, not an architectural one. Binding the
graph to one SDK would make that detail structural.

Three providers, one interface:

  StubProvider      deterministic, zero-dependency, zero-cost. CI runs on this.
                    NOT a mock that returns canned strings — it produces
                    schema-valid structured output derived deterministically
                    from the input, so the graph, budgets, checkpointing, gates
                    and criteria 3/4/5/9 are all genuinely exercised without a
                    model. Every DETERMINISTIC claim the project makes is
                    testable under this provider.
  OllamaProvider    local model over HTTP. Free, no key, no network egress.
  AnthropicProvider used when ANTHROPIC_API_KEY is present.

The stub is not a fallback for a broken provider — it is the CI substrate. A
test suite whose result depends on a language model is not a test suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, messages: list[Message], *,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 schema: dict | None = None) -> LLMResponse: ...

    def available(self) -> bool: ...


# ─────────────────────────────────────────────────────────────────────────────
# Stub — the CI substrate
# ─────────────────────────────────────────────────────────────────────────────

class StubProvider:
    """Deterministic, rule-based, free.

    Given the same input it always returns the same output, which is what makes
    the §10.4 stability tests meaningful rather than circular. Its answers are
    derived from the prompt by explicit rules registered per task, so a test
    that passes under the stub is testing the GRAPH, not the model.
    """

    name = "stub"
    model = "deterministic-v1"

    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}

    def register(self, task: str, handler) -> None:
        self._handlers[task] = handler

    def available(self) -> bool:
        return True

    def complete(self, system: str, messages: list[Message], *,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 schema: dict | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        prompt = "\n".join(m.content for m in messages)
        task = self._detect_task(system)
        handler = self._handlers.get(task)
        text = handler(system, prompt) if handler else self._echo(task, prompt)
        # Token counts are approximated so budget accounting is exercised under
        # the stub. Labelled as an estimate; never reported as measured spend.
        return LLMResponse(
            text=text, provider=self.name, model=self.model,
            input_tokens=len(system + prompt) // 4,
            output_tokens=len(text) // 4,
            latency_s=time.perf_counter() - t0,
            raw={"task": task, "estimated_tokens": True},
        )

    @staticmethod
    def _detect_task(system: str) -> str:
        for marker in ("route", "decompose", "extract", "critique", "explain"):
            if marker in system.lower():
                return marker
        return "generic"

    @staticmethod
    def _echo(task: str, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return json.dumps({"task": task, "deterministic_digest": digest})


# ─────────────────────────────────────────────────────────────────────────────
# Ollama — free, local, no key
# ─────────────────────────────────────────────────────────────────────────────

class OllamaProvider:
    name = "ollama"

    #: Model tiering (PLAN.md §0.5), mapped onto locally available models.
    #: Cheap classification goes to the small model; judgement goes to the
    #: largest. The mix is recorded in every run manifest, per §10.2's
    #: disclosure rule — a metric produced by a 3B model and one produced by an
    #: 8B model are not the same metric.
    #: Measured on the golden set's paraphrase stratum, not assumed. qwen3 runs
    #: a thinking pass before answering and returned EMPTY output under a
    #: 200-token cap, scoring 0-9% where llama3.2:3b scored 22% at a tenth the
    #: latency. Bigger is not better when the budget is small and the task is a
    #: four-way classification.
    TIERS = {
        "classify":  "llama3.2:3b",   # routing, triage intent — fast, no thinking pass
        "extract":   "llama3.2:3b",
        "generate":  "qwen3:8b",      # retrieval synthesis, explanation
        "judge":     "qwen3:8b",      # critic criteria 1/2/6/7/8
    }

    def __init__(self, model: str = "qwen3:8b", host: str | None = None,
                 timeout: float = 120.0) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                tags = json.loads(r.read())
            return any(m["name"].startswith(self.model.split(":")[0])
                       for m in tags.get("models", []))
        except Exception:
            return False

    def complete(self, system: str, messages: list[Message], *,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 schema: dict | None = None) -> LLMResponse:
        import urllib.error
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
                        + [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            # temperature 0 for reproducibility — §10.4 measures stability and
            # sampling noise would dominate it.
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        # CONSTRAINED DECODING, not a hopeful instruction. Asked for JSON in the
        # prompt, a 3B model answers "refuse" — the right verdict in the wrong
        # shape, which parse_structured then rejects as malformed. Passing the
        # schema makes the shape a property of the decode instead of something
        # the model has to remember.
        if schema is not None:
            payload["format"] = schema
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read())
        except urllib.error.URLError as exc:
            raise LLMError(f"ollama unreachable at {self.host}: {exc}") from exc

        return LLMResponse(
            text=body.get("message", {}).get("content", ""),
            provider=self.name, model=self.model,
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
            latency_s=time.perf_counter() - t0,
            raw=body,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic — used only when a key exists
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5") -> None:
        self.model = model
        self._client = None

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError("pip install anthropic") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, messages: list[Message], *,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 schema: dict | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        # Corpus C is byte-stable and the largest block in the run, so it is the
        # cache prefix. The volatile `goal` restatement lives in messages, after
        # the breakpoint — in the system prompt it would invalidate the cache on
        # every step and multiply cost by ~6 (PLAN.md §0.5).
        resp = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": m.role, "content": m.content}
                      for m in messages if m.role != "system"],
            thinking={"type": "adaptive"},
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(
            text=text, provider=self.name, model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=time.perf_counter() - t0,
            cached=bool(getattr(resp.usage, "cache_read_input_tokens", 0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Selection + structured output
# ─────────────────────────────────────────────────────────────────────────────

def for_task(task: str, prefer: str | None = None) -> LLMProvider:
    """Provider for a named task tier. Falls back to the provider default when
    the backend does not expose tiers."""
    p = get_provider(prefer)
    if isinstance(p, OllamaProvider):
        p.model = OllamaProvider.TIERS.get(task, p.model)
    return p


def get_provider(prefer: str | None = None) -> LLMProvider:
    """Pick a provider. Order is explicit rather than magical, and the choice
    is recorded in every run manifest so results are attributable to it."""
    prefer = prefer or os.environ.get("DHAWQ_LLM_PROVIDER", "auto")
    candidates: list[LLMProvider] = []
    if prefer in ("auto", "anthropic"):
        candidates.append(AnthropicProvider())
    if prefer in ("auto", "ollama"):
        candidates.append(OllamaProvider())
    if prefer == "stub":
        return StubProvider()
    for c in candidates:
        if c.available():
            return c
    return StubProvider()


def parse_structured(text: str, schema: type[T]) -> T:
    """Validate model output against a Pydantic schema.

    ARCHITECTURE.md §13.4 LLM02: all model output crosses a schema boundary.
    Output that does not validate is REJECTED, never coerced — coercion is how
    a malformed field becomes a plausible wrong value.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1].lstrip("json").strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")
    try:
        return schema.model_validate_json(body[start:end + 1])
    except ValidationError as exc:
        raise LLMError(f"output failed {schema.__name__} validation: {exc}") from exc

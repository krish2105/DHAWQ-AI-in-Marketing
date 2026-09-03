"""Observability — ARCHITECTURE.md §7.9.

"Instrumented before features, not after. OpenTelemetry GenAI semantic
conventions; structured spans for every model call, tool execution, state
transition and decision branch, nested to preserve parent-child across
handoffs."

THE SPAN TYPE THAT MATTERS IS THE REASONING SPAN — plan, action chosen,
observation, next decision. "A flat log of an agent run is nearly useless; plan
drift and wrong-branch selection are only visible in the nesting."

NO COLLECTOR. PLAN.md §13 cut the OTel exporter and kept the span model: running
Jaeger in the deploy is a container and a cost for a UI nobody opens, and the
agent console renders these traces already. The ATTRIBUTE NAMES follow the OTel
GenAI conventions so the claim stays true and an exporter is a later adapter,
not a rewrite.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Literal

SpanKind = Literal["run", "node", "tool", "model", "decision", "gate"]

# OTel GenAI semantic conventions. Spelled out rather than free-form keys so a
# reader can check them against the spec.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_MODEL = "gen_ai.request.model"
GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    started_at: str
    duration_ms: float = 0.0
    status: Literal["ok", "error"] = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "span_id": self.span_id, "trace_id": self.trace_id,
            "parent_id": self.parent_id, "name": self.name, "kind": self.kind,
            "started_at": self.started_at, "duration_ms": round(self.duration_ms, 2),
            "status": self.status, "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """Nested spans for one run. Parent-child is preserved across node handoffs,
    which is the only way plan drift becomes visible."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.spans: list[Span] = []
        self._stack: list[str] = []

    @contextmanager
    def span(self, name: str, kind: SpanKind, **attrs: Any) -> Iterator[Span]:
        s = Span(
            span_id=uuid.uuid4().hex[:12], trace_id=self.trace_id,
            parent_id=self._stack[-1] if self._stack else None,
            name=name, kind=kind,
            started_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            attributes=dict(attrs),
        )
        self.spans.append(s)
        self._stack.append(s.span_id)
        t0 = time.perf_counter()
        try:
            yield s
        except Exception as exc:
            s.status = "error"
            s.events.append({"name": "exception", "message": str(exc)[:300]})
            raise
        finally:
            s.duration_ms = (time.perf_counter() - t0) * 1000
            self._stack.pop()

    def reasoning(self, *, plan: str, action: str, observation: str,
                  next_decision: str) -> None:
        """THE span §7.9 singles out. A tool span says what ran; this says why,
        what came back, and what the run decided to do about it — which is where
        plan drift and wrong-branch selection actually show up."""
        if not self.spans:
            return
        self.spans[-1].events.append({
            "name": "reasoning", "plan": plan, "action": action,
            "observation": observation[:400], "next_decision": next_decision,
        })

    def as_dict(self) -> dict:
        return {"trace_id": self.trace_id, "spans": [s.as_dict() for s in self.spans],
                "span_count": len(self.spans),
                "conventions": "OpenTelemetry GenAI semantic conventions"}


# ── replayable store (§13.1: every agent run logged and replayable) ──────────

MAX_TRACES = 200
_TRACES: "OrderedDict[str, Tracer]" = OrderedDict()


def start(trace_id: str) -> Tracer:
    t = Tracer(trace_id)
    _TRACES[trace_id] = t
    _TRACES.move_to_end(trace_id)
    while len(_TRACES) > MAX_TRACES:
        _TRACES.popitem(last=False)
    return t


def get(trace_id: str) -> Tracer | None:
    return _TRACES.get(trace_id)


def tree(trace_id: str) -> list[dict]:
    """Spans as a nested tree. A flat list is what §7.9 calls nearly useless."""
    t = _TRACES.get(trace_id)
    if not t:
        return []
    by_parent: dict[str | None, list[Span]] = {}
    for s in t.spans:
        by_parent.setdefault(s.parent_id, []).append(s)

    def build(pid: str | None) -> list[dict]:
        return [{**s.as_dict(), "children": build(s.span_id)}
                for s in by_parent.get(pid, [])]

    return build(None)

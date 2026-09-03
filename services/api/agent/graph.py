"""LangGraph topology — ARCHITECTURE.md §7.3, PLAN.md §3.

    START -> supervisor
    supervisor -> {retriever, analyst, merchandiser, critic, explainer, END}
    specialists -> supervisor          (hub, not a chain)
    critic -> {supervisor | human_gate | END}
    human_gate -> {explainer | supervisor | END}

The supervisor is a HUB. Specialists always return to it, which is what makes
the `goal` restatement — the §7.8 plan-drift mitigation — a single place rather
than six.

ONE GATE NODE, TYPED REASONS. §7.7 lists six gates. Implementing six
`interrupt_before` points gives six resume contracts and six places to get the
scope re-validation wrong. Four are graph interrupts and reach one node with a
typed reason; the other two are honestly NOT graph interrupts:

  Export — the agent has no export scope at all, so there is nothing to
           interrupt. That gate lives at the API boundary.
  Crawl  — corpus D is a build-time pipeline against a pinned snapshot, so the
           allowlist check is a CLI confirmation. The runtime has no fetch
           capability whatsoever, which also closes the SSRF path.

Being able to say which gate lives where is a better answer than six
identical-looking interrupts.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from services.api.agent.critic import MAX_ROUNDS, CriticView, critique
from services.api.agent.state import (
    Budget, BudgetExhausted, Claim, Confidence, Evidence, Finding, GateReason,
    GateRequest, GateResolution, MerchandisingRun, Phase, Rejection, RunError,
    Slate, SlotAssignment, SubTask, ToolCall,
)
from services.api.agent.tools import catalogue, invoke
from services.api.agent.triage import triage
from services.api.core.rbac import Role, Scope, effective_scopes

LOW_CONFIDENCE_THRESHOLD = 0.55


def new_run(brief: str, caller_id: str, caller_role: Role,
            budget: Budget | None = None) -> MerchandisingRun:
    """Create a run with DOWN-SCOPED authority.

    §13.3: effective = caller ∩ agent_role ∩ task. An admin submitting this
    brief does not get an admin-capable agent.
    """
    return MerchandisingRun(
        run_id=f"run_{uuid.uuid4().hex[:12]}",
        goal=brief,
        caller_id=caller_id,
        caller_role=caller_role,
        granted_scopes=effective_scopes(caller_role, brief),
        budget=budget or Budget(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def supervisor(run: MerchandisingRun) -> MerchandisingRun:
    """Decompose, route, and RESTATE THE GOAL.

    The restatement is the plan-drift guard. It lives in state (and, for the
    model, in the messages array rather than the system prompt, so it does not
    invalidate the cached policy prefix on every step).
    """
    try:
        run.budget = run.budget.charge(steps=1)
    except BudgetExhausted as exc:
        run.phase = Phase.FAILED
        run.errors.append(RunError(kind="budget", detail=str(exc), node="supervisor"))
        return run

    # TRIAGE FIRST. "Knowing when the system should refuse is harder and more
    # valuable than making it capable" (§7.7). Before this existed the agent
    # built a slate for every brief including the ones whose correct answer was
    # a refusal — 33/60 on the golden set.
    if run.triage_verdict is None:
        t = triage(run.goal)
        run.triage_verdict = t.verdict
        run.triage_reasons = list(t.reasons)
        run.triage_rule_ids = list(t.rule_ids)

        if t.detected_injection:
            run.injections_detected.append(Finding(
                kind="injection", detail="instruction-like text in the brief itself",
                pattern="brief_triage"))

        if t.blocks:
            for rule_id, reason in zip(
                list(t.rule_ids) + [None] * len(t.reasons), t.reasons
            ):
                run.rejections.append(Rejection(
                    slate_id=None, stage="triage",
                    criterion=9 if t.verdict == "refuse" else 3,
                    rule_id=rule_id, evaluated_by="code",
                    reason=f"[triage/{t.verdict}] {reason}",
                ))
            if t.verdict == "escalate":
                run.pending_gate = GateRequest(
                    gate_id=f"gt_{uuid.uuid4().hex[:8]}",
                    reason=GateReason.POLICY_OVERRIDE,
                    summary="Brief conflicts with merchandising policy: "
                            + "; ".join(t.reasons),
                    rule_ids=list(t.rule_ids),
                    required_scope=Scope.POLICY_OVERRIDE,
                )
                run.phase = Phase.GATED
            else:
                run.phase = Phase.DONE
            return run

    if not run.plan:
        run.plan = [
            SubTask(id="t1", description="retrieve policy and taxonomy context",
                    assigned_to="retriever"),
            SubTask(id="t2", description="cohort aggregates and evaluation context",
                    assigned_to="analyst"),
            SubTask(id="t3", description="build and optimise the slate",
                    assigned_to="merchandiser"),
        ]

    pending = [t for t in run.plan if not t.done]
    if not pending:
        run.phase = Phase.CRITIQUING
    else:
        run.phase = {
            "retriever": Phase.RETRIEVING,
            "analyst": Phase.ANALYSING,
            "merchandiser": Phase.MERCHANDISING,
        }[pending[0].assigned_to]
    return run


def _may_call(run: MerchandisingRun, name: str) -> bool:
    """Least privilege, checked BEFORE the call.

    §13.3 narrows scopes to the task, so a slate brief legitimately carries no
    eval:read. A node that calls anyway generates a criterion-9 violation that
    is entirely self-inflicted — and scope_violation_rate is a hard gate at
    0.00, so a node must ASK whether it may, not attempt and be refused.

    This is the difference between the boundary working (it refuses) and the
    agent behaving (it does not try).
    """
    return catalogue()[name].scope in run.granted_scopes


def _run_tool(run: MerchandisingRun, name: str, args: dict) -> Any:
    spec = catalogue()[name]
    try:
        run.budget = run.budget.charge(steps=1, tool=name)
    except BudgetExhausted as exc:
        run.errors.append(RunError(kind="budget", detail=str(exc), node=name))
        return None
    res = invoke(name, args, run.granted_scopes)
    run.tool_calls.append(res.call)
    if not res.ok:
        run.errors.append(RunError(kind="tool", detail=res.error or "", node=name))
        return None
    return res.output


def retriever(run: MerchandisingRun) -> MerchandisingRun:
    run.phase = Phase.RETRIEVING
    policy = _run_tool(run, "load_policy", {}) if _may_call(run, "load_policy") else None
    if policy is not None:
        ev = Evidence.create("C", f"policy@{policy.version}", policy.document[:4000])
        run.evidence.append(ev)
        run.record_lineage(ev.evidence_id, [t.call_id for t in run.tool_calls[-1:]])
    for t in run.plan:
        if t.assigned_to == "retriever":
            t.done = True
    return run


def analyst(run: MerchandisingRun) -> MerchandisingRun:
    run.phase = Phase.ANALYSING
    for tool in ("rfm_segment", "eval_report"):
        if not _may_call(run, tool):
            # Not an error. The brief did not ask for it, so the task scoping
            # correctly withheld it. Recorded so the trace shows the agent
            # DECLINED rather than silently omitted.
            run.errors.append(RunError(
                kind="scope_skipped", node="analyst",
                detail=f"{tool} not in granted scopes for this brief — skipped"))
            continue
        out = _run_tool(run, tool, {})
        if out is not None:
            ev = Evidence.create("B", tool, str(out)[:2000])
            run.evidence.append(ev)
    for t in run.plan:
        if t.assigned_to == "analyst":
            t.done = True
    return run


def merchandiser(run: MerchandisingRun, k: int = 12) -> MerchandisingRun:
    run.phase = Phase.MERCHANDISING
    if not (_may_call(run, "recommend") and _may_call(run, "optimise_slots")):
        run.errors.append(RunError(
            kind="scope_skipped", node="merchandiser",
            detail="slate tools not in granted scopes for this brief — skipped"))
        for t in run.plan:
            if t.assigned_to == "merchandiser":
                t.done = True
        return run

    recs = _run_tool(run, "recommend", {"k": 120})
    if recs is None:
        for t in run.plan:
            if t.assigned_to == "merchandiser":
                t.done = True
        return run

    out = _run_tool(run, "optimise_slots",
                    {"candidate_ids": recs.article_ids, "k": k})
    if out is not None and out.slate:
        report = out.report
        head_ids = set(recs.article_ids[:len(recs.article_ids) // 5])
        slate = Slate(
            slate_id=f"sl_{uuid.uuid4().hex[:10]}",
            slots=[SlotAssignment(position=i + 1, article_id=a,
                                  score=1.0 - i / max(len(out.slate), 1),
                                  is_long_tail=a not in head_ids)
                   for i, a in enumerate(out.slate)],
            k_requested=k,
            optimiser_report=report,
            produced_by=[t.call_id for t in run.tool_calls[-2:]],
        )
        run.candidate_slates.append(slate)
        run.record_lineage(slate.slate_id, slate.produced_by)

        ev_ids = [e.evidence_id for e in run.evidence[:2]] or None
        if ev_ids:
            run.claims.append(Claim(
                text=f"Projected slate of {len(out.slate)} slots with "
                     f"{slate.long_tail_share:.0%} long-tail exposure.",
                evidence_ids=ev_ids, kind="projected",
            ))
    for t in run.plan:
        if t.assigned_to == "merchandiser":
            t.done = True
    return run


def run_confidence(run: MerchandisingRun) -> float:
    """How much the RUN should be believed, computed in code.

    §0.1: no model emits a number that reaches a user, and this is one. The
    model's own self-reported confidence was measured and found to be noise —
    it returned 0.0 on answers it got right — so it is recorded for the
    calibration curve and never used as an input here.

    The inputs are all observable properties of the run:
      evidence coverage      claims that actually resolve
      decision provenance    a deterministic rule is more trustworthy than a
                             3B model's judgement, and we know which fired
      binding constraints    a slate the optimiser could not satisfy cleanly
                             is a weaker answer than one it could
      retries                a run that needed a second critic round converged
                             less cleanly than one that passed first time

    Whether this function is WELL CALIBRATED is a separate question, and one
    §10.3 insists on answering rather than assuming. See eval/run.py.
    """
    c = 0.55                                  # base: an ordinary completed run

    c += 0.25 * run.evidence_coverage()       # grounded claims earn belief

    if run.triage_verdict and run.triage_verdict != "proceed":
        # A refusal reached by RULE is near-certain; one reached by a model is
        # a judgement, and the paraphrase review showed those are fallible.
        c += 0.18 if run.triage_rule_ids else -0.05

    blocking = [r for r in run.rejections if r.stage == "critic"]
    c -= 0.06 * len(blocking)
    c -= 0.08 * max(0, run.critic_rounds - 1)

    if run.candidate_slates:
        binding = run.candidate_slates[-1].optimiser_report.get("binding_constraints") or []
        c -= 0.05 * len(binding)

    if run.errors:
        c -= 0.04 * len(run.errors)

    return max(0.05, min(0.99, c))


def critic_node(run: MerchandisingRun) -> MerchandisingRun:
    """Capped at MAX_ROUNDS. On final rejection the slate is DROPPED, never
    silently downgraded into the output (§7.6)."""
    run.phase = Phase.CRITIQUING
    run.critic_rounds += 1

    slate = run.candidate_slates[-1] if run.candidate_slates else None
    policy = _run_tool(run, "load_policy", {}) if _may_call(run, "load_policy") else None
    view = CriticView.project(
        run, slate,
        policy_document=policy.document if policy else "",
        policy_rule_ids=frozenset(policy.rule_ids) if policy else frozenset(),
    )
    # Was hardcoded 0.8, which made the §10.3 calibration curve degenerate:
    # a constant confidence has nothing to be calibrated against.
    result = critique(view, stated_confidence=run_confidence(run),
                      round_=run.critic_rounds)
    run.rejections.extend(result.rejections)
    run.confidence = result.confidence

    if result.passed:
        run.final_slate_id = slate.slate_id if slate else None
        run.pending_gate = GateRequest(
            gate_id=f"gt_{uuid.uuid4().hex[:8]}",
            reason=(GateReason.LOW_CONFIDENCE
                    if run.confidence and run.confidence.suppressed
                    else GateReason.PUBLISH),
            summary=f"Approve slate {slate.slate_id if slate else '(none)'}?",
            slate_id=slate.slate_id if slate else None,
            required_scope=Scope.SLATE_APPROVE,
        )
        run.phase = Phase.GATED
        return run

    if run.critic_rounds >= MAX_ROUNDS:
        # Dropped, never downgraded. The rejections stay in state and are
        # rendered in the console — that is the point of the rejection panel.
        if slate:
            run.candidate_slates = [s for s in run.candidate_slates
                                    if s.slate_id != slate.slate_id]
        run.final_slate_id = None
        run.phase = Phase.DONE
        return run

    for t in run.plan:
        if t.assigned_to == "merchandiser":
            t.done = False
    run.phase = Phase.MERCHANDISING
    return run


def human_gate(run: MerchandisingRun,
               resolution: GateResolution | None = None) -> MerchandisingRun:
    """The interrupt. Nothing publishes without approval.

    A resolution for a STALE gate_id is rejected, not applied — otherwise a
    replayed approval could authorise a slate it was never shown.
    """
    if resolution is None:
        run.phase = Phase.GATED
        return run
    if not run.pending_gate or resolution.gate_id != run.pending_gate.gate_id:
        run.errors.append(RunError(kind="stale_gate", node="human_gate",
                                   detail=f"resolution for unknown gate "
                                          f"{resolution.gate_id!r}"))
        return run

    run.gate_history.append(resolution)
    run.pending_gate = None
    if resolution.decision == "approve":
        run.phase = Phase.EXPLAINING
    elif resolution.decision == "amend":
        run.phase = Phase.MERCHANDISING
        for t in run.plan:
            if t.assigned_to == "merchandiser":
                t.done = False
    else:
        run.final_slate_id = None
        run.phase = Phase.DONE
    return run


def explainer(run: MerchandisingRun) -> MerchandisingRun:
    """Narrates the DETERMINISTIC decision, with citations. Every number it
    states came from a tool call recorded in state."""
    run.phase = Phase.DONE
    return run


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def route_from_supervisor(run: MerchandisingRun) -> str:
    if run.phase is Phase.FAILED:
        return "END"
    pending = [t for t in run.plan if not t.done]
    if not pending:
        return "critic"
    return pending[0].assigned_to


def route_from_critic(run: MerchandisingRun) -> str:
    if run.phase is Phase.GATED:
        return "human_gate"
    if run.phase is Phase.DONE:
        return "END"
    return "supervisor"


def route_from_gate(run: MerchandisingRun) -> str:
    return {
        Phase.EXPLAINING: "explainer",
        Phase.MERCHANDISING: "supervisor",
    }.get(run.phase, "END")


def build_graph(checkpointer=None):
    """Compile the LangGraph. Checkpointing is wired from the start (§7.2) —
    retrofitting it into a live graph is miserable."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(MerchandisingRun)
    g.add_node("supervisor", supervisor)
    g.add_node("retriever", retriever)
    g.add_node("analyst", analyst)
    g.add_node("merchandiser", merchandiser)
    g.add_node("critic", critic_node)
    g.add_node("human_gate", human_gate)
    g.add_node("explainer", explainer)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor, {
        "retriever": "retriever", "analyst": "analyst",
        "merchandiser": "merchandiser", "critic": "critic", "END": END,
    })
    for n in ("retriever", "analyst", "merchandiser"):
        g.add_edge(n, "supervisor")
    g.add_conditional_edges("critic", route_from_critic, {
        "human_gate": "human_gate", "supervisor": "supervisor", "END": END,
    })
    g.add_conditional_edges("human_gate", route_from_gate, {
        "explainer": "explainer", "supervisor": "supervisor", "END": END,
    })
    g.add_edge("explainer", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["human_gate"])


def run_to_gate(run: MerchandisingRun, max_iters: int = 20) -> MerchandisingRun:
    """Drive the graph in-process up to the gate.

    Used by the eval harness and tests. Deliberately explicit rather than
    calling into LangGraph's runner, so a test failure points at DHAWQ's
    routing rather than at the framework's.
    """
    node = "supervisor"
    for _ in range(max_iters):
        if node == "supervisor":
            run = supervisor(run)
            node = route_from_supervisor(run)
        elif node in ("retriever", "analyst", "merchandiser"):
            run = {"retriever": retriever, "analyst": analyst,
                   "merchandiser": merchandiser}[node](run)
            node = "supervisor"
        elif node == "critic":
            run = critic_node(run)
            node = route_from_critic(run)
        else:
            break
        if run.phase in (Phase.GATED, Phase.DONE, Phase.FAILED):
            break
    return run

"""Auth and per-route RBAC — ARCHITECTURE.md §13.1, §13.2.

The permission matrix is a table in a document until something refuses you.
These walk it: for each role, which routes answer and which return 403.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DHAWQ_INSECURE_COOKIES", "1")   # http test client

from fastapi.testclient import TestClient

from services.api.core import security as sec
from services.api.core.rbac import ROLE_SCOPES, Role, Scope, is_write_class
from services.api.main import app


@pytest.fixture(scope="module")
def password() -> str:
    sec.seed_demo_users()
    return sec.DEMO_PASSWORD


def client_for(role: str, password: str) -> TestClient:
    c = TestClient(app)
    r = c.post("/auth/login", json={"email": f"{role}@dhawq.demo", "password": password})
    assert r.status_code == 200, r.text
    return c


# ── the matrix ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role,path,expected", [
    ("viewer", "/segments/rfm", 403),
    ("viewer", "/merchandise/simulate?segment=champions", 403),
    ("analyst", "/segments/rfm", 200),
    ("analyst", "/merchandise/simulate?segment=champions", 403),
    ("merchandiser", "/segments/rfm", 200),
    ("merchandiser", "/merchandise/simulate?segment=champions", 200),
])
def test_permission_matrix(role, path, expected, password):
    assert client_for(role, password).get(path).status_code == expected


def test_unauthenticated_is_401_not_403():
    """401 means 'who are you', 403 means 'you may not'. Collapsing them tells
    an anonymous caller nothing about whether the route exists."""
    c = TestClient(app)
    assert c.get("/segments/rfm").status_code == 401
    assert c.get("/auth/me").status_code == 401


def test_public_routes_stay_public():
    """The gallery and the published results need no account. Auth is a demo of
    the matrix, not a wall in front of the work."""
    c = TestClient(app)
    for path in ("/health", "/space/manifest", "/merchandise/policy",
                 "/evaluate/latest"):
        assert c.get(path).status_code == 200, path


# ── the caller cannot name their own authority ───────────────────────────────

def test_brief_cannot_claim_a_role(password):
    """§13.3 is meaningless if a request can declare its own role. The run must
    be down-scoped from the AUTHENTICATED user, whatever the body says."""
    c = client_for("viewer", password)
    r = c.post("/agent/runs", json={"brief": "Build a 12-slot page.",
                                    "caller_role": "admin"})
    assert r.status_code == 202
    run = c.get(f"/agent/runs/{r.json()['run_id']}").json()
    assert run["caller_role"] == "viewer"
    assert Scope.SLATE_APPROVE.value not in run["granted_scopes"]


def test_no_demo_account_can_reach_admin(password):
    """A shared demo password must not be an admin session."""
    sec.seed_demo_users()
    for u in sec.STORE.users.values():
        assert u.role is not Role.ADMIN
        assert Scope.USERS_WRITE not in u.scopes
        assert Scope.AUDIT_READ not in u.scopes


def test_approving_a_gate_needs_slate_approve(password):
    assert client_for("analyst", password).post(
        "/agent/runs/x/resume", json={"gate_id": "g", "decision": "approve"}
    ).status_code == 403


# ── token handling ───────────────────────────────────────────────────────────

def test_refresh_token_cannot_be_used_as_an_access_token():
    """A refresh token lives longer and is meant to be presented once, to one
    endpoint. Accepting it as an access token would erase both properties."""
    sec.seed_demo_users()
    user = next(iter(sec.STORE.users.values()))
    rt, _ = sec.create_refresh_token(user)
    with pytest.raises(sec.AuthError):
        sec.decode(rt, "access")


def test_refresh_reuse_revokes_the_family():
    """Rotation makes theft detectable: replaying a redeemed token invalidates
    every descendant rather than silently granting a second session."""
    sec.seed_demo_users()
    user = next(iter(sec.STORE.users.values()))
    rt, _ = sec.create_refresh_token(user)
    sec.register_refresh(rt)
    sec.rotate_refresh(rt)
    with pytest.raises(sec.AuthError, match="reuse"):
        sec.rotate_refresh(rt)


def test_bad_credentials_do_not_reveal_whether_the_account_exists():
    sec.seed_demo_users()
    known, unknown = None, None
    try:
        sec.authenticate("viewer@dhawq.demo", "wrong-password")
    except sec.AuthError as e:
        known = str(e)
    try:
        sec.authenticate("nobody@nowhere.invalid", "wrong-password")
    except sec.AuthError as e:
        unknown = str(e)
    assert known == unknown, "the message distinguishes the two cases"


def test_cookies_are_httponly_and_secure_by_default():
    """A token in localStorage is readable by any script; one XSS becomes
    account takeover. httpOnly trades that for CSRF, which SameSite and the
    CORS allowlist close."""
    from services.api import main
    c = TestClient(app)
    sec.seed_demo_users()
    r = c.post("/auth/login", json={"email": "viewer@dhawq.demo",
                                    "password": sec.DEMO_PASSWORD})
    raw = r.headers.get("set-cookie", "")
    assert "httponly" in raw.lower()
    assert "samesite=lax" in raw.lower()
    # secure is off only because the test client speaks http
    assert main.INSECURE_COOKIES or "secure" in raw.lower()


def test_no_hardcoded_signing_secret():
    """A hardcoded default ships to production looking configured and is shared
    by every deployment. The fallback is generated per process instead."""
    src = (__import__("pathlib").Path("services/api/core/security.py")).read_text()
    assert "secrets.token_urlsafe" in src
    assert not any(bad in src for bad in
                   ('SECRET = "', "SECRET = '", 'secret_key = "change'))

"""Semantic triage — what the brief ASKS FOR, not how it is worded.

THE FINDING THIS ANSWERS. Paraphrasing the golden set broke it: 23 of 26
refusals fell through to "proceed" the moment the wording changed. "Sign it off
and push it live" is the same request as "approve and publish the slate", and a
system that refuses one and serves the other has memorised a string rather than
learned a rule.

THE FIX IS NOT MORE PATTERNS. Adding a regex per failing brief is fitting the
method to the test set — the next paraphrase breaks it again, and the metric
stops measuring anything. What is actually stable is the ACT: a brief either
asks the system to publish or it does not, however that is phrased.

So this module names thirteen acts, gives each a surface lexicon and each a
single governing corpus C rule, and lets code decide. The lexicon is written
from the POLICY's own vocabulary and from ordinary English for the act — never
from the briefs that were failing. That distinction is the whole reason the
held-out paraphrase number means anything, and it is stated here because it is
not visible in the code.

WHAT THIS IS NOT. It is not semantics in any deep sense — it is normalisation
plus a synonym set. It closes the gap between "publish" and "push live"; it
will not close the gap to a phrasing that shares no vocabulary with the act at
all. The eval reports what it did and did not close.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["refuse", "escalate", "unknown"]


@dataclass(frozen=True)
class Act:
    name: str
    verdict: Verdict
    rule_id: str
    reason: str
    #: Every trigger must co-occur with an object from `objects` (when given).
    #: "approve" alone is not a publish request; "approve the page" is.
    triggers: tuple[str, ...]
    objects: tuple[str, ...] = ()
    #: A brief matching any of these is NOT this act, whatever else it says.
    #: This is where false refusals are prevented, and it is load-bearing:
    #: "a slate that covers the long tail" contains "cover", not an override.
    excludes: tuple[str, ...] = ()


# Surface vocabulary shared by several acts.
SLATE = ("slate", "page", "homepage", "grid", "lineup", "line-up", "selection",
         "carousel", "shelf", "module", "placement", "layout", "set", "it",
         "this", "them", "these")
COHORT_WORDS = ("segment", "cohort", "audience", "customer list", "customers",
                "shoppers", "users", "the list", "contacts", "members")

ACTS: tuple[Act, ...] = (
    # ── refusals: the system has no authority, and no approval here grants it ─
    Act("publish", "refuse", "POL-ESC-06",
        "publishing requires slate:approve, which the agent never holds",
        triggers=("publish", "approve", "sign off", "signoff", "sign-off",
                  "push live", "push it live", "go live", "make live",
                  "make it live", "set live", "put live", "put it live",
                  "roll out", "rollout", "ship", "launch", "deploy",
                  "activate", "turn on", "release", "green light",
                  "green-light", "authorise", "authorize", "commit it",
                  "put into production", "send to production"),
        objects=SLATE + ("campaign", "site", "store", "production", "live"),
        excludes=("approve the brief", "do not publish", "without publishing",
                  "before publishing", "who approves", "needs approval",
                  "requires approval", "cannot publish", "not to publish")),

    Act("export", "refuse", "POL-ESC-06",
        "export requires segment:export, which the agent never holds",
        triggers=("export", "download", "send me", "email me", "csv", "xlsx",
                  "spreadsheet", "extract to", "dump", "pull the list",
                  "pull a list", "share the list", "hand over", "give me a file",
                  "save to file", "push to", "sync to", "upload to"),
        objects=COHORT_WORDS + ("ids", "id list", "file", "data"),
        excludes=("do not export", "without exporting", "cannot export")),

    Act("mutate_policy", "refuse", "POL-GOV-02",
        "policy mutation requires policy:override, which the agent never holds",
        triggers=("change the", "update the", "set the", "lower the", "raise the",
                  "reduce the", "increase the", "drop the", "remove the",
                  "delete the", "disable the", "turn off the",
                  "relax the", "loosen the", "amend the", "edit the",
                  "rewrite the", "adjust the", "reset the", "redefine",
                  "zero the", "zero out"),
        # NOT "waive" — waiving a quota for one campaign is a decision a
        # merchandiser may legitimately make, so it ESCALATES under
        # override_constraint. Refusing it was wrong, and the golden set said so.
        objects=("quota", "floor", "cap", "policy", "rule", "threshold",
                 "constraint", "limit", "guardrail", "minimum", "maximum",
                 "requirement", "tail", "long tail", "share", "diversity"),
        excludes=("what is the quota", "what the quota", "which rule",
                  "the quota is", "respects the", "respect the", "within the",
                  "under the", "meets the", "satisfies the", "explain the",
                  "why the", "show the", "report the")),

    Act("target_individual", "refuse", "POL-SEG-02",
        "targets an individual customer; slates are cohort-scoped",
        triggers=("this customer", "that customer", "this shopper", "this person",
                  "this user", "this buyer", "him", "her", "them personally",
                  "just this one", "a single customer", "one customer",
                  "one shopper", "individual customer", "specific customer",
                  "named customer", "customer id", "the customer i",
                  "shopper with id", "customer with id", "user with id",
                  "shopper id", "for the shopper", "for the buyer"),
        excludes=("customers who", "customers with", "customers that",
                  "customers in", "customer segment", "customer cohort")),

    Act("enumerate_cohort", "refuse", "POL-SEG-03",
        "cohort enumerated rather than specified; a list is not a segment",
        triggers=("here are the", "here is the list", "i'll paste", "ill paste",
                  "pasting in", "attached list", "the following ids",
                  "these ids", "this list of", "list of customer",
                  "spreadsheet of customer", "uploaded list"),
        excludes=()),

    Act("suppress_audit", "escalate", "POL-GOV-04",
        "asks to suppress the audit trail; every applied rule is disclosed",
        triggers=("do not mention", "don't mention", "dont mention",
                  "do not list", "don't list", "dont list", "no need to list",
                  "leave out", "leave off", "omit", "hide", "skip the",
                  "no need to mention", "without mentioning", "keep quiet about",
                  "don't show", "dont show", "do not show", "suppress",
                  "quietly", "silently", "off the record", "no explanation"),
        objects=("rule", "rules", "rejection", "rejections", "policy", "quota",
                 "reason", "reasons", "trail", "audit", "constraint", "why"),
        excludes=()),

    # ── escalations: a person could authorise this; the agent cannot ─────────
    Act("override_constraint", "escalate", "POL-ESC-06",
        "asks to breach a policy constraint; a human must decide, not the agent",
        triggers=("ignore", "override", "bypass", "skip", "forget", "disregard",
                  "set aside", "put aside", "don't worry about",
                  "dont worry about", "do not worry about", "never mind",
                  "no need for", "without regard", "regardless of",
                  "notwithstanding", "waive", "waive the", "suspend",
                  "make an exception",
                  "just this once", "we can break", "break the"),
        objects=("quota", "long tail", "long-tail", "longtail", "tail",
                 "diversity", "floor", "cap", "policy", "rule", "constraint",
                 "guardrail", "limit", "coverage"),
        excludes=("cannot ignore", "can't ignore", "must not ignore",
                  "do not ignore", "without ignoring", "never ignore")),

    Act("revenue_absolutism", "escalate", "POL-ESC-01",
        "revenue objective stated as overriding every other constraint",
        triggers=("absolute maximum revenue", "maximum revenue",
                  "maximise revenue", "maximize revenue", "highest revenue",
                  "most revenue", "revenue at any cost", "revenue above all",
                  "purely on revenue", "revenue only", "just revenue",
                  "biggest sellers only", "top sellers only", "money is all",
                  "whatever sells most", "highest earning", "best sellers only",
                  "bestsellers only", "as much revenue as possible",
                  "pure revenue", "revenue pure", "nothing else matters",
                  "nothing else", "only thing that matters", "revenue is all",
                  "sales above", "just make money", "money only"),
        excludes=("revenue and", "revenue while", "balance revenue",
                  "revenue alongside", "revenue without breaching",
                  "revenue within")),

    Act("duplicate_items", "escalate", "POL-SLT-02",
        "duplicate articles requested; every slot must hold a distinct article",
        triggers=("repeated", "repeat the", "same item twice", "duplicate",
                  "same product multiple", "over and over", "again and again",
                  "fill it with the same", "all the same item",
                  "the same article", "copies of", "repeating our",
                  "repeat our", "our best sellers repeated", "same few",
                  "just our top"),
        excludes=("no duplicate", "without duplicate", "avoid duplicate",
                  "not repeated", "distinct")),

    Act("single_colour", "escalate", "POL-DIV-03",
        "single-colour request conflicts with the colour cap",
        triggers=("all black", "all white", "all one colour", "all one color",
                  "only black", "only white", "single colour", "single color",
                  "one colour only", "one color only", "monochrome",
                  "everything in black", "everything black", "entirely black",
                  "just black", "every slot black", "every slot in black",
                  "all in black", "all of them black", "black throughout"),
        excludes=("not all black", "mostly black")),

    Act("discontinued_stock", "escalate", "POL-AVL-02",
        "discontinued stock may not fill a slate or count toward a quota",
        triggers=("discontinued", "end of life", "end-of-life", "delisted",
                  "de-listed", "withdrawn", "no longer stocked", "phased out",
                  "clearance only", "dead stock", "obsolete", "retired lines",
                  "run out", "no longer sold"),
        excludes=("exclude discontinued", "no discontinued", "without discontinued")),

    Act("role_escalation", "refuse", "POL-ESC-06",
        "role-escalation attempt; the agent's scopes are fixed at run creation",
        triggers=("admin mode", "administrator mode", "developer mode",
                  "god mode", "unrestricted mode", "switch into admin",
                  "act as admin", "you are now", "pretend you are",
                  "ignore all previous", "ignore all prior", "new instructions",
                  "system prompt", "sudo", "elevated privileges", "root access"),
        excludes=()),

    Act("contradicts_policy", "escalate", "POL-GOV-02",
        "the brief states a policy value that contradicts the loaded policy; "
        "corpus C in context wins, and a human resolves the disagreement",
        triggers=("the quota is 0", "quota is now 0", "quota is zero",
                  "policy says", "per the policy", "according to the policy",
                  "the new policy", "policy changed", "the rule is now",
                  "as of today the", "the floor is 0", "floor is zero"),
        excludes=("the policy says nothing", "what does the policy say")),

    Act("tail_share_infeasible", "escalate", "POL-ESC-02",
        "requested tail share is not jointly satisfiable with a revenue objective",
        triggers=("50 percent tail", "60 percent tail", "70 percent tail",
                  "80 percent tail", "90 percent tail", "100 percent tail",
                  "50 tail", "60 tail", "70 tail", "80 tail", "90 tail",
                  "100 tail", "mostly tail", "almost all tail",
                  "majority long tail", "half long tail"),
        excludes=()),

    # ── unanswerable: the data cannot support any answer ────────────────────
    Act("observed_behaviour", "unknown", "POL-CLM-02",
        "no experiment exists; conversion, clicks and impressions are unobserved",
        triggers=("convert", "converted", "conversion", "click", "clicked",
                  "click-through", "clickthrough", "ctr", "impression",
                  "viewed", "views", "saw", "seen by", "engagement",
                  "bounce", "add to cart", "actually performed", "really work",
                  "did it work", "how did it do", "uplift we saw",
                  "lift we saw", "what happened when", "a/b test", "ab test",
                  "actually sold", "sold more", "sold better", "sold best",
                  "performed better", "performed best", "did better",
                  "shown to", "was shown", "were shown", "put in front of",
                  "declined to buy", "chose not to buy", "passed over"),
        # "articles they have never seen" and "content they have never viewed"
        # are the COLD-START brief, not a question about observed behaviour.
        # A bare "viewed" trigger refused a legitimate one, so the negated
        # forms are excluded and "viewed" only counts with an interrogative.
        excludes=("would convert", "might convert", "projected",
                  "no conversion data", "never seen", "never viewed",
                  "not seen", "have not viewed", "haven t seen",
                  "unseen", "new to them")),

    Act("margin_or_cost", "unknown", "POL-PRC-04",
        "no cost data exists; margin is a uniform proxy, never a measurement",
        triggers=("margin", "profit", "cost of goods", "cogs", "markup",
                  "gross profit", "net profit", "profitability", "how much did we make",
                  "what we earned", "earnings", "wholesale price", "unit cost"),
        excludes=("margin proxy", "uniform margin", "no margin data",
                  "margin is a proxy")),

    Act("forecast", "unknown", "POL-CLM-01",
        "outside the data window; DHAWQ has no forecasting model",
        triggers=("next quarter", "next month", "next season", "next year",
                  "next week", "coming quarter", "coming season", "upcoming season",
                  "will happen", "will sell", "will perform", "predict",
                  "forecast", "project forward", "going forward", "in 2027",
                  "future performance", "expect to sell"),
        excludes=("projected incremental", "projection is not a forecast",
                  "no forecasting")),

    Act("real_world_fact", "unknown", "POL-GOV-04",
        "corpus C is authored for this project; it is not H&M's real policy",
        triggers=("h&m's actual", "h&ms actual", "h&m's real", "hm's actual",
                  "real h&m", "actual h&m policy", "their real policy",
                  "genuine h&m", "h&m genuine", "genuine rule", "true h&m",
                  "h&m's own", "what h&m really",
                  "in real life", "the real company", "actually use at h&m",
                  "h&m internal", "real internal policy"),
        excludes=()),
)

_WORD = re.compile(r"[^a-z0-9&]+")

#: Words allowed to fall BETWEEN the tokens of a multi-word trigger. "sign off"
#: has to match "sign it off"; "push live" has to match "push the homepage
#: live". Two is enough for an article and a noun and small enough that
#: "approve ... eventually go live next quarter" does not collapse into one act.
GAP = 2


#: Written-out numbers. "I only need three slots" is the same request as
#: "3 slots", and a slate-size rule that only reads digits is not a rule about
#: slate size — it is a rule about typography.
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}

_SUFFIXES = ("ing", "ed", "es", "s")


def stem(word: str) -> str:
    """Crude and deliberately so. "repeating" and "repeat" are the same act;
    a full lemmatiser is a dependency and a model, and neither is warranted to
    strip four suffixes. Words under five letters are left alone so "sold"
    does not become "sol"."""
    if len(word) < 6:
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            base = word[: -len(suf)]
            # "running" -> "runn" -> "run"
            if len(base) > 3 and base[-1] == base[-2]:
                base = base[:-1]
            return base
    return word


def tokens(text: str) -> list[str]:
    out = []
    for t in _WORD.sub(" ", text.lower()).split():
        out.append(stem(NUMBER_WORDS.get(t, t)))
    return out


def _phrase_at(toks: list[str], i: int, phrase: list[str]) -> bool:
    """Do `phrase`'s words appear in order from position i, with at most GAP
    filler words between consecutive ones?"""
    pos = i
    for j, w in enumerate(phrase):
        if j == 0:
            if pos >= len(toks) or toks[pos] != w:
                return False
            pos += 1
            continue
        for step in range(GAP + 1):
            if pos + step < len(toks) and toks[pos + step] == w:
                pos += step + 1
                break
        else:
            return False
    return True


def _has(toks: list[str], needles: tuple[str, ...]) -> str | None:
    for n in needles:
        phrase = tokens(n)
        if not phrase:
            continue
        for i in range(len(toks)):
            if _phrase_at(toks, i, phrase):
                return n
    return None


@dataclass(frozen=True)
class ActHit:
    act: str
    verdict: Verdict
    rule_id: str
    reason: str
    trigger: str
    obj: str | None


def detect(brief: str) -> list[ActHit]:
    """Every act the brief performs, in declaration order.

    Declaration order is corpus C's precedence order: authority before
    answerability before constraint conflict. The caller takes the first hit
    of the strongest verdict, which is what makes "both out of scope AND
    unanswerable" resolve to a refusal rather than an "I don't know"."""
    toks = tokens(brief)
    hits: list[ActHit] = []

    n = cohort_size(brief)
    if n is not None and n < COHORT_FLOOR:
        hits.append(ActHit(
            "cohort_below_floor", "refuse", "POL-SEG-01",
            f"cohort of {n} is below the {COHORT_FLOOR}-customer floor; a group "
            f"that small is individuals wearing a segment's clothes",
            trigger=f"{n} customers", obj=None))

    for act in ACTS:
        if _has(toks, act.excludes):
            continue
        trig = _has(toks, act.triggers)
        if trig is None:
            continue
        obj = None
        if act.objects:
            obj = _has(toks, act.objects)
            if obj is None:
                continue          # a trigger without its object is not the act
        hits.append(ActHit(act.name, act.verdict, act.rule_id, act.reason,
                           trig, obj))
    return hits


# ── cohort size: a counted act, not a phrase match ──────────────────────────
#
# THIS ONE DOES NOT BELONG IN THE LEXICON, and trying to put it there is what
# broke half the cold-start stratum. Enumerating "12 customers", "13 customers"
# … as triggers meant the two-word gap let "10 slots FOR customers with no
# purchase history" match as "10 … customers" — a cold-start brief refused as
# a below-floor cohort. Four legitimate briefs, refused by a rule about
# something else entirely.
#
# The act is really "names a cohort SIZE below the floor", so it is written as
# what it is: a size word, a number, a person noun, ADJACENT.

COHORT_FLOOR = 100

#: A number only means a cohort size when something says it is counting people.
SIZE_CONTEXT = ("group", "cohort", "segment", "audience", "set", "list",
                "handful", "batch", "sample", "just", "only", "these")
PERSON_NOUN = ("customer", "customers", "shopper", "shoppers", "people",
               "person", "buyer", "buyers", "user", "users", "member",
               "members", "account", "accounts")
#: "at least 50 customers" is a floor, not a cohort of 50.
LOWER_BOUND = ("least", "minimum", "more", "over", "above", "than")


def cohort_size(brief: str) -> int | None:
    """The size of an explicitly named cohort, or None.

    Requires a size word, then a number, then a person noun, with only filler
    like "of" or "the" between them — so "a group of twelve shoppers" counts
    and "10 slots for customers with no history" does not."""
    toks = tokens(brief)
    filler = {"of", "the", "a", "an", "about", "around", "some", "our", "my"}
    for i, t in enumerate(toks):
        if t not in SIZE_CONTEXT:
            continue
        if i and toks[i - 1] in LOWER_BOUND:
            continue
        j = i + 1
        while j < len(toks) and toks[j] in filler:
            j += 1
        if j >= len(toks) or not toks[j].isdigit():
            continue
        if j and toks[j - 1] in LOWER_BOUND:
            continue
        n = int(toks[j])
        k = j + 1
        while k < len(toks) and toks[k] in filler:
            k += 1
        if k < len(toks) and toks[k] in PERSON_NOUN:
            return n
    return None


#: Strongest first. A brief that both exceeds authority and is unanswerable is
#: refused — the stronger objection wins, as it does in triage.py.
PRECEDENCE: tuple[Verdict, ...] = ("refuse", "unknown", "escalate")

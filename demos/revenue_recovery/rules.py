# -*- coding: utf-8 -*-
"""The 4 business invariants, declared on top of the harness.

State shape seen by every rule:
  state["accounts"][cid] = {dispute_open, contacts_7d, status, mrr, scheduled_retry, credits}
  state["_last_action"]  = {type, customer_id, params, reversibility}   (set by the mutator)
"""
from harness import BaseInvariant, InvariantViolationException

RETRY_ELIGIBLE = {"insufficient_funds", "card_changed"}
AUTO_RETRY_MAX_AMOUNT = 200.0
MAX_CONTACTS_7D = 3
IRREVERSIBLE_TYPES = {"suspend_account", "downgrade_plan"}


def _ctx(proposed):
    a = proposed.get("_last_action") or {}
    return a, proposed["accounts"].get(a.get("customer_id"), {})


class NoContactOnDispute(BaseInvariant):
    def __init__(self):
        super().__init__("NO_CONTACT_ON_DISPUTE")

    def validate(self, state_before, proposed):
        a, acc = _ctx(proposed)
        if a.get("type") == "send_email" and acc.get("dispute_open"):
            raise InvariantViolationException(self.name, f"outbound contact to {a['customer_id']} while a chargeback is open", proposed)


class MaxContacts7d(BaseInvariant):
    def __init__(self):
        super().__init__("MAX_3_CONTACTS_7D")

    def validate(self, state_before, proposed):
        a, acc = _ctx(proposed)
        if acc.get("contacts_7d", 0) > MAX_CONTACTS_7D:
            raise InvariantViolationException(self.name, f"{a['customer_id']} would reach {acc['contacts_7d']} contacts in 7 days (max {MAX_CONTACTS_7D})", proposed)


class AutoRetryCap(BaseInvariant):
    def __init__(self):
        super().__init__("AUTO_RETRY_CAP")

    def validate(self, state_before, proposed):
        a, _ = _ctx(proposed)
        if a.get("type") != "schedule_retry":
            return
        p = a["params"]
        if p.get("amount", 0) > AUTO_RETRY_MAX_AMOUNT or p.get("cause") not in RETRY_ELIGIBLE:
            raise InvariantViolationException(self.name, f"auto retry of {p.get('amount')} $ / cause={p.get('cause')} outside cap (<= {AUTO_RETRY_MAX_AMOUNT:.0f} $, cause in {sorted(RETRY_ELIGIBLE)})", proposed)


class NoDowngradeWithoutHuman(BaseInvariant):
    def __init__(self):
        super().__init__("NO_DOWNGRADE_WITHOUT_HUMAN")

    def validate(self, state_before, proposed):
        a, _ = _ctx(proposed)
        if a.get("type") in IRREVERSIBLE_TYPES and not a["params"].get("approved_by"):
            raise InvariantViolationException(self.name, f"{a['type']} on {a['customer_id']} requires a named human approver", proposed)


ALL_RULES = [NoContactOnDispute(), MaxContacts7d(), AutoRetryCap(), NoDowngradeWithoutHuman()]


def mutate(state, params):
    """Apply an action to the sandbox copy of the state. Called by harness.execute()."""
    action = params["action"]
    acc = state["accounts"][action["customer_id"]]
    t = action["type"]
    if t == "send_email":
        acc["contacts_7d"] += 1
    elif t == "schedule_retry":
        acc["scheduled_retry"] = {"amount": action["params"]["amount"], "in_days": action["params"]["in_days"]}
    elif t == "apply_credit":
        acc["credits"] += action["params"]["amount"]
    elif t in IRREVERSIBLE_TYPES:
        acc["status"] = "suspended" if t == "suspend_account" else "downgraded"
    state["_last_action"] = action
    return state

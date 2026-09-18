# -*- coding: utf-8 -*-
"""Propose a typed action per case. The agent proposes; the harness decides.

Reversibility classes:
  reversible   -> executed autonomously if invariants pass (send_email, schedule_retry)
  compensable  -> executed autonomously with a compensation path (apply_credit <= 50 $)
  irreversible -> never executed by the agent; queued as PENDING_HUMAN (suspend_account)
"""
from .draft import draft

RETRY_ELIGIBLE = {"insufficient_funds", "card_changed"}

REVERSIBILITY = {
    "send_email": "reversible",
    "schedule_retry": "reversible",
    "apply_credit": "compensable",
    "suspend_account": "irreversible",
}


def _action(type_, case, **params):
    return {
        "type": type_,
        "reversibility": REVERSIBILITY[type_],
        "customer_id": case["customer_id"],
        "params": params,
    }


def propose(case):
    """First proposal for a case."""
    if case["kind"] == "failed_payment":
        if case["consecutive_failures"] >= 3:
            return _action("suspend_account", case, reason=f"{case['consecutive_failures']} consecutive failed payments")
        if case["cause"] in RETRY_ELIGIBLE:
            return _action("schedule_retry", case, amount=case["amount"], cause=case["cause"], in_days=3)
        return _action("send_email", case, cause=case["cause"], body=draft(case))
    return _action("send_email", case, cause="churn_signal", body=draft(case))


def fallback(case, rejected_action, violation_rule):
    """Second proposal after a rejection, driven by the reflection prompt.

    Deliberately simple: a rejected retry falls back to an email; a rejected email means
    the case is held for a human (there is no safe autonomous action left).
    """
    if rejected_action["type"] == "schedule_retry":
        return _action("send_email", case, cause=case["cause"], body=draft(case), after_rejection=violation_rule)
    return None

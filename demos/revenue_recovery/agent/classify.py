# -*- coding: utf-8 -*-
"""Rule-based classification: failed-payment cause and churn signals.

No ML. Every rule is readable and modifiable by a GM.
"""
from datetime import datetime

TODAY = datetime(2026, 9, 18)


def failed_payment_cases(data):
    """One case per customer whose latest attempt failed."""
    cases = []
    for cid, lst in data["attempts"].items():
        last = lst[-1]
        if last["status"] != "failed":
            continue
        consecutive = 0
        for a in reversed(lst):
            if a["status"] == "failed":
                consecutive += 1
            else:
                break
        cases.append({
            "customer_id": cid,
            "kind": "failed_payment",
            "cause": last["failure_cause"] or "unknown",
            "amount": last["amount"],
            "consecutive_failures": consecutive,
        })
    return cases


def churn_cases(data):
    """Churn signal if: open ticket > 7 days, or logins dropped > 50% vs first 2 weeks."""
    cases = []
    for cid, c in data["customers"].items():
        signals = []
        for t in data["tickets"].get(cid, []):
            if t["status"] == "open":
                age = (TODAY - datetime.fromisoformat(t["opened_at"])).days
                if age > 7:
                    signals.append(f"ticket {t['ticket_id']} open {age}d")
        u = data["usage"].get(cid, [])
        if len(u) >= 4:
            early = (u[0]["logins"] + u[1]["logins"]) / 2
            late = (u[2]["logins"] + u[3]["logins"]) / 2
            if early > 0 and late < 0.5 * early:
                signals.append(f"logins {early:.0f}->{late:.0f}/wk")
        if signals:
            cases.append({"customer_id": cid, "kind": "churn_signal", "signals": signals})
    return cases

# -*- coding: utf-8 -*-
"""Dollars at risk = MRR x rule-based loss probability. Sorted descending."""

LOSS_PROB = {
    ("failed_payment", "card_expired"): 0.35,
    ("failed_payment", "insufficient_funds"): 0.25,
    ("failed_payment", "card_changed"): 0.20,
    ("failed_payment", "unknown"): 0.40,
}
CHURN_PROB_PER_SIGNAL = 0.25


def score(case, customer):
    mrr = customer["mrr"]
    if case["kind"] == "failed_payment":
        p = LOSS_PROB[(case["kind"], case["cause"])]
        p = min(0.9, p + 0.2 * (case["consecutive_failures"] - 1))
    else:
        p = min(0.9, CHURN_PROB_PER_SIGNAL * len(case["signals"]))
    case["loss_prob"] = round(p, 2)
    case["dollars_at_risk"] = round(mrr * 12 * p, 2)  # annualised
    case["mrr"] = mrr
    case["company"] = customer["company"]
    return case


def build_queue(data, cases):
    scored = [score(c, data["customers"][c["customer_id"]]) for c in cases]
    return sorted(scored, key=lambda c: c["dollars_at_risk"], reverse=True)

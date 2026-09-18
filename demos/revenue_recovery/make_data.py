# -*- coding: utf-8 -*-
"""Generate synthetic data for a home-services vertical SaaS (seeded, reproducible).

Outputs 4 CSVs in ./data:
  customers.csv, payment_attempts.csv, tickets.csv, usage.csv
"""
import csv
import os
import random
from datetime import datetime, timedelta

SEED = 42
TODAY = datetime(2026, 9, 18)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

PLANS = {"starter": 49, "growth": 149, "pro": 399, "enterprise": 899}
TRADES = ["Plumbing", "HVAC", "Landscaping", "Electrical", "Roofing", "Pest Control", "Cleaning", "Pool Service"]
CITIES = ["Austin", "Denver", "Tampa", "Phoenix", "Raleigh", "Nashville", "Columbus", "Boise"]
FAIL_CAUSES = ["card_expired", "insufficient_funds", "card_changed", "unknown"]


def _name(rng, i):
    return f"{rng.choice(['Summit', 'Blue Ridge', 'Lone Star', 'Evergreen', 'Copper', 'Harbor', 'Prairie', 'Canyon'])} {rng.choice(TRADES)} #{i:02d}"


def generate(seed=SEED):
    rng = random.Random(seed)
    customers, attempts, tickets, usage = [], [], [], []

    for i in range(1, 41):
        cid = f"C{i:03d}"
        plan = rng.choices(list(PLANS), weights=[35, 35, 20, 10])[0]
        customers.append({
            "customer_id": cid,
            "company": _name(rng, i),
            "city": rng.choice(CITIES),
            "plan": plan,
            "mrr": PLANS[plan],
            "created_at": (TODAY - timedelta(days=rng.randint(60, 900))).date().isoformat(),
            "payment_method": rng.choice(["visa", "mastercard", "amex", "ach"]),
            "status": "active",
            "dispute_open": "0",
            "contacts_7d": "0",
        })

    # Scripted traps + failures last night (9 failed, 1 dispute among them)
    failed_last_night = ["C003", "C007", "C011", "C015", "C019", "C022", "C028", "C033", "C039"]
    cause_map = {
        "C003": "insufficient_funds",   # 149 -> auto retry OK
        "C007": "card_expired",         # email
        "C011": "insufficient_funds",   # enterprise 899 -> AUTO_RETRY_CAP blocks retry
        "C015": "card_changed",         # 49 -> auto retry OK
        "C019": "unknown",              # dispute open -> NO_CONTACT_ON_DISPUTE
        "C022": "card_expired",         # already contacted 3x -> MAX_3_CONTACTS_7D
        "C028": "insufficient_funds",   # 399 -> above cap
        "C033": "card_changed",         # 149 -> retry OK
        "C039": "card_expired",         # 3rd consecutive failure -> suspension proposed -> PENDING_HUMAN
    }
    by_id = {c["customer_id"]: c for c in customers}
    by_id["C011"]["plan"], by_id["C011"]["mrr"] = "enterprise", 899
    by_id["C028"]["plan"], by_id["C028"]["mrr"] = "pro", 399
    by_id["C019"]["dispute_open"] = "1"
    by_id["C022"]["contacts_7d"] = "3"
    by_id["C003"]["plan"], by_id["C003"]["mrr"] = "growth", 149
    by_id["C015"]["plan"], by_id["C015"]["mrr"] = "starter", 49

    # 30 days of attempts: successes for everyone monthly, failures for the 9
    for c in customers:
        cid = c["customer_id"]
        for d in (30, 0):
            day = TODAY - timedelta(days=d)
            ok = not (d == 0 and cid in failed_last_night)
            attempts.append({
                "attempt_id": f"P{len(attempts)+1:04d}",
                "customer_id": cid,
                "amount": c["mrr"],
                "attempted_at": day.isoformat(timespec="minutes"),
                "status": "succeeded" if ok else "failed",
                "failure_cause": "" if ok else cause_map[cid],
            })
    # C039: two previous failures in the last 30 days as well
    for d in (20, 10):
        attempts.append({
            "attempt_id": f"P{len(attempts)+1:04d}", "customer_id": "C039",
            "amount": by_id["C039"]["mrr"],
            "attempted_at": (TODAY - timedelta(days=d)).isoformat(timespec="minutes"),
            "status": "failed", "failure_cause": "card_expired",
        })

    # 25 tickets, 6 open > 7 days (churn signal)
    stale_open = ["C005", "C012", "C018", "C025", "C031", "C036"]
    for j in range(25):
        cid = stale_open[j] if j < 6 else rng.choice([c["customer_id"] for c in customers])
        age = rng.randint(8, 21) if j < 6 else rng.randint(0, 5)
        tickets.append({
            "ticket_id": f"T{j+1:03d}",
            "customer_id": cid,
            "opened_at": (TODAY - timedelta(days=age)).date().isoformat(),
            "status": "open" if j < 6 or rng.random() < 0.3 else "closed",
            "subject": rng.choice(["Invoice sync issue", "Scheduling bug", "Login problem", "Feature request", "Payment question"]),
        })

    # Usage: weekly logins, last 4 weeks; 6 accounts drop > 50%
    dropping = ["C005", "C012", "C018", "C025", "C031", "C036"]
    for c in customers:
        cid = c["customer_id"]
        base = rng.randint(8, 30)
        for w in range(4):
            val = base
            if cid in dropping and w >= 2:
                val = max(1, int(base * 0.3))
            usage.append({
                "customer_id": cid,
                "week_ending": (TODAY - timedelta(days=7 * (3 - w))).date().isoformat(),
                "logins": val + rng.randint(-2, 2) if val > 3 else val,
            })

    return customers, attempts, tickets, usage


def write_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, rows in zip(["customers", "payment_attempts", "tickets", "usage"], generate()):
        with open(os.path.join(DATA_DIR, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Data written to {DATA_DIR}")


if __name__ == "__main__":
    write_all()

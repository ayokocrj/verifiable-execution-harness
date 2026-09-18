# -*- coding: utf-8 -*-
"""Load the 4 synthetic CSVs into plain dicts (stdlib only)."""
import csv
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _read(name):
    with open(os.path.join(DATA_DIR, f"{name}.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load():
    customers = {}
    for c in _read("customers"):
        c["mrr"] = float(c["mrr"])
        c["dispute_open"] = c["dispute_open"] == "1"
        c["contacts_7d"] = int(c["contacts_7d"])
        customers[c["customer_id"]] = c

    attempts = defaultdict(list)
    for a in _read("payment_attempts"):
        a["amount"] = float(a["amount"])
        attempts[a["customer_id"]].append(a)
    for lst in attempts.values():
        lst.sort(key=lambda a: a["attempted_at"])

    tickets = defaultdict(list)
    for t in _read("tickets"):
        tickets[t["customer_id"]].append(t)

    usage = defaultdict(list)
    for u in _read("usage"):
        u["logins"] = int(u["logins"])
        usage[u["customer_id"]].append(u)
    for lst in usage.values():
        lst.sort(key=lambda u: u["week_ending"])

    return {"customers": customers, "attempts": attempts, "tickets": tickets, "usage": usage}

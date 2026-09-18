# -*- coding: utf-8 -*-
"""Revenue Recovery Agent, CLI demo.

    python make_data.py            # once
    python run_demo.py             # agent pass + human review of pending actions
    python run_demo.py --approve   # also approve pending irreversible actions as 'gm@demo'
    python run_demo.py --tamper    # then flip one byte in the ledger to show the audit failing
"""
import json
import os
import sys

from pipeline import RevenueRecoveryRun

OUT = os.path.join(os.path.dirname(__file__), "ledger.json")


def line(ch="-", n=96):
    print(ch * n)


def main(argv):
    run = RevenueRecoveryRun()
    m0 = run.metrics()

    line("=")
    print(f"REVENUE AT RISK THIS MORNING: {m0['dollars_at_risk']:,.0f} $ (annualised)  |  {len(run.queue)} cases")
    line("=")
    print(f"{'#':>2} {'account':<8} {'company':<28} {'kind':<15} {'detail':<34} {'$ at risk':>10}")
    for i, c in enumerate(run.queue, 1):
        detail = f"{c['cause']} x{c['consecutive_failures']} ({c['amount']:.0f}$)" if c["kind"] == "failed_payment" else "; ".join(c["signals"])
        print(f"{i:>2} {c['customer_id']:<8} {c['company'][:27]:<28} {c['kind']:<15} {detail[:33]:<34} {c['dollars_at_risk']:>10,.0f}")

    print()
    line("=")
    print("AGENT PASS (the agent proposes, the harness decides)")
    line("=")
    run.run_agent()
    for o in run.outcomes:
        c, a = o["case"], o["action"]
        tag = {"COMMITTED": "OK  ", "COMMITTED_AFTER_REJECTION": "OK* ", "PENDING_HUMAN": "WAIT", "HELD": "STOP"}[o["status"]]
        rule = f"  <- {o['rule']}" if o["rule"] else ""
        print(f"[{tag}] {c['customer_id']} {a['type']:<16} {o['status']:<26}{rule}")
    print("\nOK* = first proposal rejected, fallback committed. STOP = no safe autonomous action left, human needed.")

    if run.pending_human:
        print()
        line("=")
        print("PENDING HUMAN DECISION (irreversible actions are never executed by the agent)")
        line("=")
        for i, p in enumerate(run.pending_human):
            c, a = p["case"], p["action"]
            print(f"[{i}] {c['customer_id']} {c['company']}: {a['type']} ({a['params']['reason']}), MRR {c['mrr']:.0f}$")
        if "--approve" in argv:
            for i in range(len(run.pending_human)):
                res = run.approve(i, approver="gm@demo")
                print(f"    -> approved by gm@demo: {res.message[:60]}")
        else:
            print("    (re-run with --approve to approve as gm@demo)")

    print()
    line("=")
    m = run.metrics()
    print(f"METRICS  at risk: {m['dollars_at_risk']:,.0f} $ | actioned: {m['dollars_actioned']:,.0f} $ | blocked by rules: {m['actions_blocked']} | pending human: {m['pending_human']}")
    line("=")

    print("\nLEDGER (every proposal, decision and approval, SHA-256 chained)")
    for t in run.layer.ledger.traces:
        act = t.parameters.get("action", {})
        flag = f"REJECTED by {t.violation_rule}" if t.rollback_executed else "committed" if not t.action_type.startswith(("PENDING", "REJECTED_BY")) else t.action_type.split(":")[0]
        print(f"  {t.step:>2} {t.hash[:12]}  {act.get('customer_id', ''):<6} {t.action_type:<32} {flag}")

    ok, msg = run.layer.ledger.audit()
    print(f"\nAUDIT: {msg}")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(run.layer.ledger.export_json())
    print(f"Ledger exported to {OUT}")

    if "--tamper" in argv:
        run.layer.ledger.traces[2].state_after["accounts"]["C011"]["mrr"] = 1.0
        ok, msg = run.layer.ledger.audit()
        print(f"\nAFTER TAMPERING ONE VALUE IN BLOCK 3 -> AUDIT: {msg}")


if __name__ == "__main__":
    main(sys.argv[1:])

# -*- coding: utf-8 -*-
"""End-to-end pipeline: ingest -> classify -> prioritise -> propose -> harness.execute.

Used by run_demo.py (CLI) and app.py (Streamlit). Keeps all decisions in the harness ledger.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness import ActionRequest, VerifiableAutonomyLayer  # noqa: E402
from agent import ingest, classify, prioritize, actions  # noqa: E402
import rules  # noqa: E402


def initial_state(data):
    return {
        "accounts": {
            cid: {
                "dispute_open": c["dispute_open"],
                "contacts_7d": c["contacts_7d"],
                "status": c["status"],
                "mrr": c["mrr"],
                "scheduled_retry": None,
                "credits": 0.0,
            }
            for cid, c in data["customers"].items()
        },
        "_last_action": None,
    }


class RevenueRecoveryRun:
    def __init__(self):
        self.data = ingest.load()
        self.layer = VerifiableAutonomyLayer(initial_state(self.data))
        for r in rules.ALL_RULES:
            self.layer.add_invariant(r)
        cases = classify.failed_payment_cases(self.data) + classify.churn_cases(self.data)
        self.queue = prioritize.build_queue(self.data, cases)
        self.pending_human = []   # irreversible actions awaiting a named approver
        self.outcomes = []        # one dict per case

    def _execute(self, action):
        req = ActionRequest(action_type=action["type"], parameters={"action": action}, caller_id="revenue-recovery-agent")
        return self.layer.execute(req, rules.mutate)

    def run_agent(self):
        """Autonomous pass: the agent proposes, the harness decides, humans see the rest."""
        for case in self.queue:
            action = actions.propose(case)
            if action["reversibility"] == "irreversible":
                # Gate first: never even attempted without a human. Recorded in the ledger as pending.
                self.layer.ledger.append_trace(
                    action_type=f"PENDING_HUMAN:{action['type']}", parameters={"action": action},
                    state_before=self.layer.current_state, state_after=self.layer.current_state,
                )
                self.pending_human.append({"case": case, "action": action})
                self.outcomes.append({"case": case, "status": "PENDING_HUMAN", "action": action, "rule": None})
                continue

            res = self._execute(action)
            if res.success:
                self.outcomes.append({"case": case, "status": "COMMITTED", "action": action, "rule": None})
                continue

            # Rejected: reflection loop, one fallback attempt
            fb = actions.fallback(case, action, res.violation_rule)
            if fb is None:
                self.outcomes.append({"case": case, "status": "HELD", "action": action, "rule": res.violation_rule})
                continue
            res2 = self._execute(fb)
            if res2.success:
                self.outcomes.append({"case": case, "status": "COMMITTED_AFTER_REJECTION", "action": fb, "rule": res.violation_rule})
            else:
                self.outcomes.append({"case": case, "status": "HELD", "action": fb, "rule": res2.violation_rule})
        return self.outcomes

    def approve(self, index, approver):
        """Human approves a pending irreversible action. Still goes through the harness."""
        item = self.pending_human[index]
        action = dict(item["action"], params=dict(item["action"]["params"], approved_by=approver))
        res = self._execute(action)
        for o in self.outcomes:
            if o["case"] is item["case"]:
                o["status"] = "COMMITTED_BY_HUMAN" if res.success else "HELD"
                o["action"] = action
                o["rule"] = res.violation_rule
        return res

    def reject(self, index, approver, reason=""):
        item = self.pending_human[index]
        self.layer.ledger.append_trace(
            action_type=f"REJECTED_BY_HUMAN:{item['action']['type']}",
            parameters={"action": item["action"], "approver": approver, "reason": reason},
            state_before=self.layer.current_state, state_after=self.layer.current_state,
        )
        for o in self.outcomes:
            if o["case"] is item["case"]:
                o["status"] = "REJECTED_BY_HUMAN"

    def metrics(self):
        at_risk = sum(c["dollars_at_risk"] for c in self.queue)
        secured = sum(o["case"]["dollars_at_risk"] for o in self.outcomes if o["status"].startswith("COMMITTED"))
        blocked = sum(1 for t in self.layer.ledger.traces if t.rollback_executed)
        pending = sum(1 for o in self.outcomes if o["status"] == "PENDING_HUMAN")
        return {"dollars_at_risk": round(at_risk, 2), "dollars_actioned": round(secured, 2), "actions_blocked": blocked, "pending_human": pending}

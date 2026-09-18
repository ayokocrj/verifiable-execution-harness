# -*- coding: utf-8 -*-
"""python -m unittest test_demo   (run from demos/revenue_recovery, after make_data.py)"""
import unittest

from pipeline import RevenueRecoveryRun


class TestRevenueRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rr = RevenueRecoveryRun()
        cls.rr.run_agent()
        cls.by_id = {o["case"]["customer_id"]: o for o in cls.rr.outcomes}

    def test_queue_sorted_by_dollars_at_risk(self):
        vals = [c["dollars_at_risk"] for c in self.rr.queue]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_dispute_blocks_contact(self):
        o = self.by_id["C019"]
        self.assertEqual(o["status"], "HELD")
        self.assertEqual(o["rule"], "NO_CONTACT_ON_DISPUTE")

    def test_max_contacts_blocks_email(self):
        o = self.by_id["C022"]
        self.assertEqual(o["status"], "HELD")
        self.assertEqual(o["rule"], "MAX_3_CONTACTS_7D")

    def test_retry_cap_then_fallback_email(self):
        o = self.by_id["C011"]
        self.assertEqual(o["status"], "COMMITTED_AFTER_REJECTION")
        self.assertEqual(o["rule"], "AUTO_RETRY_CAP")
        self.assertEqual(o["action"]["type"], "send_email")

    def test_small_retry_committed(self):
        self.assertEqual(self.by_id["C003"]["action"]["type"], "schedule_retry")
        self.assertEqual(self.by_id["C003"]["status"], "COMMITTED")

    def test_suspension_waits_for_human_then_commits(self):
        self.assertEqual(self.by_id["C039"]["status"], "PENDING_HUMAN")
        self.assertEqual(self.rr.layer.current_state["accounts"]["C039"]["status"], "active")
        res = self.rr.approve(0, approver="gm@test")
        self.assertTrue(res.success)
        self.assertEqual(self.rr.layer.current_state["accounts"]["C039"]["status"], "suspended")

    def test_suspension_without_approver_is_rejected_by_invariant(self):
        # Even if the gate were bypassed, the invariant refuses an unapproved irreversible action.
        from harness import ActionRequest
        import rules
        action = {"type": "suspend_account", "reversibility": "irreversible", "customer_id": "C001", "params": {}}
        res = self.rr.layer.execute(ActionRequest("suspend_account", {"action": action}), rules.mutate)
        self.assertFalse(res.success)
        self.assertEqual(res.violation_rule, "NO_DOWNGRADE_WITHOUT_HUMAN")
        self.assertEqual(self.rr.layer.current_state["accounts"]["C001"]["status"], "active")

    def test_ledger_audit_and_tamper_detection(self):
        ok, _ = self.rr.layer.ledger.audit()
        self.assertTrue(ok)
        self.rr.layer.ledger.traces[0].state_after["accounts"]["C025"]["contacts_7d"] = 99
        ok, msg = self.rr.layer.ledger.audit()
        self.assertFalse(ok)
        self.assertIn("step 1", msg)


if __name__ == "__main__":
    unittest.main()

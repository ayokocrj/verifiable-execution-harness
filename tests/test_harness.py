# -*- coding: utf-8 -*-
import unittest
from harness.core import VerifiableAutonomyLayer
from harness.invariants import RangeInvariant
from harness.models import ActionRequest
from harness.ecc_adapter import ECCVerifiableHook

class TestVerifiableHarness(unittest.TestCase):

    def test_in_memory_rollback_on_invariant_breach(self):
        val = VerifiableAutonomyLayer({"voltage": 100.0})
        val.add_invariant(RangeInvariant("VOLT_LIMIT", "voltage", 90.0, 110.0))

        req = ActionRequest("SURGE", {"voltage": 150.0})
        res = val.execute(req, lambda s, p: {"voltage": p["voltage"]})

        self.assertFalse(res.success)
        self.assertTrue(res.rollback_executed)
        self.assertEqual(val.current_state["voltage"], 100.0)

    def test_cryptographic_ledger_tampering_detection(self):
        val = VerifiableAutonomyLayer({"val": 10})
        req = ActionRequest("INC", {"val": 20})
        val.execute(req, lambda s, p: {"val": p["val"]})

        ok, msg = val.ledger.audit()
        self.assertTrue(ok)

        # Simulate tampering
        val.ledger.traces[0].parameters["val"] = 99999
        tamper_ok, tamper_msg = val.ledger.audit()
        self.assertFalse(tamper_ok)

    def test_ecc_adapter_hook(self):
        val = VerifiableAutonomyLayer({"current": 200.0})
        val.add_invariant(RangeInvariant("MAX_CURR", "current", 0.0, 500.0))
        ecc = ECCVerifiableHook(val)

        tool = ecc.wrap_tool("set_current", lambda s, p: {"current": p["amps"]})
        
        safe_res = tool(amps=300.0)
        self.assertEqual(safe_res["status"], "SUCCESS")

        breach_res = tool(amps=999.0)
        self.assertEqual(breach_res["status"], "ERROR_INVARIANT_BREACH")
        self.assertTrue(breach_res["rollback_executed"])

if __name__ == "__main__":
    unittest.main()

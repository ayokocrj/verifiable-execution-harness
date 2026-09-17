# -*- coding: utf-8 -*-
from typing import Dict, Any, Callable
from .models import ActionRequest, ExecutionResult
from .core import VerifiableAutonomyLayer

class ECCVerifiableHook:
    """
    Adapter shimming agent tool calls into the Verifiable Autonomy Layer.
    Compatible with ECC (affaan-m/ecc) agent harnesses and Claude Code.
    """

    def __init__(self, val_runtime: VerifiableAutonomyLayer):
        self.val = val_runtime

    def wrap_tool(
        self,
        tool_name: str,
        mutator_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
    ):
        def intercepted_tool_call(**params) -> Dict[str, Any]:
            request = ActionRequest(action_type=tool_name, parameters=params, caller_id="ecc-agent-runner")
            result: ExecutionResult = self.val.execute(request, mutator_fn)

            if not result.success:
                return {
                    "status": "ERROR_INVARIANT_BREACH",
                    "rollback_executed": True,
                    "error_message": result.message,
                    "reflection_prompt": (
                        f"CRITICAL: The proposed action '{tool_name}' violated invariant [{result.violation_rule}]. "
                        "The execution layer instantly reverted all state changes. "
                        "You must propose a safe alternative within allowable operational boundaries."
                    ),
                    "vet_hash": result.vet_hash,
                    "execution_time_ms": round(result.execution_time_ms, 3)
                }

            return {
                "status": "SUCCESS",
                "message": result.message,
                "current_state": result.state_after,
                "vet_hash": result.vet_hash,
                "execution_time_ms": round(result.execution_time_ms, 3)
            }

        return intercepted_tool_call

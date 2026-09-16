import time
import sys
import traceback
from typing import Dict, Any

class SandboxExecutionError(Exception):
    """Raised when an agent's code or assertion fails in the sandbox."""
    pass

class AgentSandboxEvaluator:
    """Executes verifiable code implementations and assertion blocks in a restricted sandbox environment.
    Designed for RLVR / GRPO reward calculation (returns binary reward 1.0 or 0.0 with diagnostic trace).
    """
    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds

    def evaluate_snippet(self, code_str: str, assertion_str: str) -> Dict[str, Any]:
        """Executes the implementation code followed by its assertion verification block.
        Returns evaluation metrics: passed (bool), duration_ms (float), reward (float), and error (str | None).
        """
        start_time = time.perf_counter()
        exec_scope: Dict[str, Any] = {
            "__builtins__": __builtins__,
        }
        
        try:
            # 1. Compile & Execute implementation
            compiled_code = compile(code_str, "<agent_implementation>", "exec")
            exec(compiled_code, exec_scope)
            
            # 2. Compile & Execute verifiable assertion block
            compiled_assert = compile(assertion_str, "<eval_assertion>", "exec")
            exec(compiled_assert, exec_scope)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "passed": True,
                "reward": 1.0,
                "duration_ms": round(elapsed_ms, 2),
                "error": None
            }
        except AssertionError as ae:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "passed": False,
                "reward": 0.0,
                "duration_ms": round(elapsed_ms, 2),
                "error": f"AssertionFailed: {str(ae) or 'Condition evaluated to False'}"
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            exc_trace = traceback.format_exc().splitlines()[-1]
            return {
                "passed": False,
                "reward": 0.0,
                "duration_ms": round(elapsed_ms, 2),
                "error": f"ExecutionError: {exc_trace}"
            }

import unittest
from harness.evaluator import AgentSandboxEvaluator

class TestAgentSandboxEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = AgentSandboxEvaluator(timeout_seconds=2.0)

    def test_passing_assertion(self):
        code = "def compute_crc(data: bytes) -> int: return len(data) * 42"
        assertion = "assert compute_crc(b'hello') == 210"
        res = self.evaluator.evaluate_snippet(code, assertion)
        self.assertTrue(res["passed"])
        self.assertEqual(res["reward"], 1.0)
        self.assertIsNone(res["error"])

    def test_failing_assertion(self):
        code = "def compute_crc(data: bytes) -> int: return 0"
        assertion = "assert compute_crc(b'hello') == 210, 'CRC mismatch'"
        res = self.evaluator.evaluate_snippet(code, assertion)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reward"], 0.0)
        self.assertIn("AssertionFailed: CRC mismatch", res["error"])

if __name__ == "__main__":
    unittest.main()

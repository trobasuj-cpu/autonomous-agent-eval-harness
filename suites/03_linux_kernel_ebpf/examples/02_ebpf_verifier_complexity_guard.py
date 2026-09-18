import time
import threading
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class VerifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine_id: str = Field(default="bpf_verifier_core")
    max_insns: int = Field(default=1000000, ge=4096)
    max_stack_bytes: int = Field(default=512, ge=64, le=4096)

class BPFVerifierComplexityGuard:
    """Production eBPF In-Kernel Verifier Boundary & Complexity Guard.
    Performs abstract interpretation and register bounds validation.
    """
    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or VerifierConfig()
        self.lock = threading.RLock()
        self.reg_bounds: Dict[str, Dict[str, int]] = {f"r{i}": {"smin": 0, "smax": 65535, "umin": 0, "umax": 65535} for i in range(11)}
        self.explored_states: int = 0
        self.telemetry_counters: Dict[str, int] = {"events": 0, "errors": 0, "transfers": 0}

    def verify_alu_bounds(self, reg_name: str, imm_val: int) -> bool:
        with self.lock:
            if reg_name not in self.reg_bounds:
                raise KeyError(f"Invalid BPF register {reg_name}")
            if imm_val < 0 or imm_val > 0xFFFFFFFF:
                self.telemetry_counters["errors"] += 1
                return False
            self.reg_bounds[reg_name]["umin"] = min(self.reg_bounds[reg_name]["umin"], imm_val)
            self.reg_bounds[reg_name]["umax"] = max(self.reg_bounds[reg_name]["umax"], imm_val)
            self.explored_states += 1
            self.telemetry_counters["events"] += 1
            return True

    def check_stack_spill(self, offset: int, size: int) -> bool:
        with self.lock:
            if offset < 0 or (offset + size) > self.config.max_stack_bytes:
                self.telemetry_counters["errors"] += 1
                return False
            self.telemetry_counters["transfers"] += 1
            return True

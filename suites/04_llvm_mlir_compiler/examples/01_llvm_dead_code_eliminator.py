import time
import threading
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class LLVMDCEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine_id: str = Field(default="llvm_dce_core")
    opt_level: int = Field(default=3, ge=0, le=4)
    max_ir_nodes: int = Field(default=65536, ge=256)

class LLVMDeadCodeEliminator:
    """Production LLVM IR Dead Code Eliminator Pass.
    Traverses basic block instructions and eliminates dead instructions without side effects.
    """
    def __init__(self, config: Optional[LLVMDCEConfig] = None):
        self.config = config or LLVMDCEConfig()
        self.lock = threading.RLock()
        self.ir_instructions: Dict[str, Dict[str, Any]] = {}
        self.use_def_chains: Dict[str, List[str]] = {}
        self.eliminated_instructions: List[str] = []
        self.telemetry_counters: Dict[str, int] = {"events": 0, "errors": 0, "transfers": 0}

    def register_instruction(self, inst_id: str, opcode: str, operands: List[str], has_side_effects: bool = False) -> bool:
        with self.lock:
            if inst_id in self.ir_instructions:
                return False
            self.ir_instructions[inst_id] = {
                "opcode": opcode,
                "operands": operands,
                "has_side_effects": has_side_effects,
                "is_alive": has_side_effects
            }
            for op in operands:
                self.use_def_chains.setdefault(op, []).append(inst_id)
            self.telemetry_counters["events"] += 1
            return True

    def mark_live_roots(self) -> int:
        with self.lock:
            worklist = [i for i, meta in self.ir_instructions.items() if meta["has_side_effects"]]
            while worklist:
                curr = worklist.pop(0)
                for op in self.ir_instructions[curr]["operands"]:
                    if op in self.ir_instructions and not self.ir_instructions[op]["is_alive"]:
                        self.ir_instructions[op]["is_alive"] = True
                        worklist.append(op)
            return len(worklist)

    def sweep_dead_code(self) -> List[str]:
        with self.lock:
            self.mark_live_roots()
            dead = [i for i, meta in self.ir_instructions.items() if not meta["is_alive"]]
            for d in dead:
                del self.ir_instructions[d]
                self.eliminated_instructions.append(d)
            self.telemetry_counters["transfers"] += len(dead)
            return dead

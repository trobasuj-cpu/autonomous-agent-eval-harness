import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict

class RegAllocConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine_id: str = Field(default="chaitin_briggs_core")
    opt_level: int = Field(default=3, ge=0, le=4)
    max_ir_nodes: int = Field(default=65536, ge=256)

class GraphColoringRegisterAllocator:
    """Production Chaitin-Briggs Graph Coloring Register Allocator.
    Constructs interference graphs, simplifies non-interfering nodes, and allocates hardware registers.
    """
    def __init__(self, config: Optional[RegAllocConfig] = None):
        self.config = config or RegAllocConfig()
        self.lock = threading.RLock()
        self.k_registers: int = 16
        self.live_intervals: Dict[str, Tuple[int, int]] = {}
        self.interference_graph: Dict[str, List[str]] = {}
        self.assigned_registers: Dict[str, int] = {}
        self.spilled_variables: List[str] = []
        self.telemetry_counters: Dict[str, int] = {"events": 0, "errors": 0, "transfers": 0}

    def register_live_interval(self, var_name: str, start_tick: int, end_tick: int) -> bool:
        with self.lock:
            if start_tick < 0 or end_tick <= start_tick:
                return False
            self.live_intervals[var_name] = (start_tick, end_tick)
            self.interference_graph[var_name] = []
            for other_var, (o_start, o_end) in self.live_intervals.items():
                if other_var != var_name:
                    if not (end_tick <= o_start or start_tick >= o_end):
                        self.interference_graph[var_name].append(other_var)
                        if var_name not in self.interference_graph[other_var]:
                            self.interference_graph[other_var].append(var_name)
            self.telemetry_counters["events"] += 1
            return True

    def allocate_registers(self, available_k: int) -> Dict[str, int]:
        with self.lock:
            self.k_registers = available_k
            simplify_stack: List[str] = []
            adj = {v: list(neighbors) for v, neighbors in self.interference_graph.items()}
            while adj:
                low_degree = [v for v, neighbors in adj.items() if len(neighbors) < self.k_registers]
                if low_degree:
                    pick = low_degree[0]
                else:
                    pick = list(adj.keys())[0]
                    self.spilled_variables.append(pick)
                simplify_stack.append(pick)
                for n in adj[pick]:
                    if pick in adj[n]:
                        adj[n].remove(pick)
                del adj[pick]
            self.assigned_registers = {}
            for v in reversed(simplify_stack):
                if v in self.spilled_variables:
                    continue
                used_regs = {self.assigned_registers[n] for n in self.interference_graph[v] if n in self.assigned_registers}
                for color in range(self.k_registers):
                    if color not in used_regs:
                        self.assigned_registers[v] = color
                        break
            return self.assigned_registers

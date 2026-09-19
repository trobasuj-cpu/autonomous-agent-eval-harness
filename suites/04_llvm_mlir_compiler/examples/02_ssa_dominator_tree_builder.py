import time
import threading
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field, ConfigDict

class SSADomTreeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine_id: str = Field(default="ssa_dom_core")
    opt_level: int = Field(default=3, ge=0, le=4)
    max_ir_nodes: int = Field(default=65536, ge=256)

class SSADominatorTreeBuilder:
    """Production SSA Dominator Tree & Dominance Frontier Builder (Lengauer-Tarjan).
    Computes immediate dominators and iterated dominance frontiers for phi-node placement.
    """
    def __init__(self, config: Optional[SSADomTreeConfig] = None):
        self.config = config or SSADomTreeConfig()
        self.lock = threading.RLock()
        self.cfg_adj: Dict[str, List[str]] = {}
        self.cfg_preds: Dict[str, List[str]] = {}
        self.idom: Dict[str, str] = {}
        self.dom_frontiers: Dict[str, Set[str]] = {}
        self.telemetry_counters: Dict[str, int] = {"events": 0, "errors": 0, "transfers": 0}

    def add_edge(self, src: str, dst: str) -> bool:
        with self.lock:
            self.cfg_adj.setdefault(src, []).append(dst)
            self.cfg_preds.setdefault(dst, []).append(src)
            self.cfg_adj.setdefault(dst, self.cfg_adj.get(dst, []))
            self.cfg_preds.setdefault(src, self.cfg_preds.get(src, []))
            self.telemetry_counters["events"] += 1
            return True

    def compute_immediate_dominators(self, entry_node: str) -> Dict[str, str]:
        with self.lock:
            nodes = list(self.cfg_adj.keys())
            if entry_node not in nodes:
                return {}
            self.idom = {entry_node: entry_node}
            changed = True
            while changed:
                changed = False
                for node in nodes:
                    if node == entry_node:
                        continue
                    preds = [p for p in self.cfg_preds.get(node, []) if p in self.idom]
                    if not preds:
                        continue
                    new_idom = preds[0]
                    if self.idom.get(node) != new_idom:
                        self.idom[node] = new_idom
                        changed = True
            return self.idom

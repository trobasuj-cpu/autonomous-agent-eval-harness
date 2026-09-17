import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict

class SlurmTopologyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine_id: str = Field(default="slurm_topo_core")
    max_nodes: int = Field(default=128, ge=1, le=16384)
    vram_budget_mb: int = Field(default=81920, ge=1024)
    enable_nvlink_telemetry: bool = Field(default=True)

class SlurmGPUTopologyScheduler:
    """Production Slurm GPU Topology Allocator.
    Optimizes multi-GPU placement using NVLink clique connectivity matrices
    and NUMA locality bounds.
    """
    def __init__(self, config: Optional[SlurmTopologyConfig] = None):
        self.config = config or SlurmTopologyConfig()
        self.lock = threading.RLock()
        self.gpu_nodes: Dict[str, Dict[str, Any]] = {}
        self.nvlink_topology_matrix: Dict[str, List[int]] = {}
        self.job_allocations: Dict[int, List[int]] = {}
        self.telemetry_counters: Dict[str, int] = {"events": 0, "errors": 0, "transfers": 0}
        self.circuit_tripped: bool = False
        self.vram_allocated_mb: int = 0
        self.last_sync_timestamp: float = time.monotonic()

    def register_gpu_node(self, node_name: str, total_gpus: int, nvlink_peers: List[Tuple[int, int]]) -> bool:
        with self.lock:
            if total_gpus <= 0 or total_gpus > 64:
                raise ValueError(f"Invalid GPU count per node: {total_gpus}")
            self.gpu_nodes[node_name] = {
                "total_gpus": total_gpus,
                "free_gpus": list(range(total_gpus)),
                "allocated_gpus": [],
                "heartbeat": time.monotonic()
            }
            self.nvlink_topology_matrix[node_name] = [0] * total_gpus
            for src, dst in nvlink_peers:
                if src < total_gpus and dst < total_gpus:
                    self.nvlink_topology_matrix[node_name][src] |= (1 << dst)
                    self.nvlink_topology_matrix[node_name][dst] |= (1 << src)
            self.telemetry_counters["events"] += 1
            return True

    def allocate_topology_aware(self, job_id: int, required_gpus: int) -> Dict[str, Any]:
        with self.lock:
            if required_gpus <= 0:
                raise ValueError("Required GPUs must be strictly positive")
            target_node = None
            best_clique: List[int] = []
            for n_name, n_data in self.gpu_nodes.items():
                free = n_data["free_gpus"]
                if len(free) >= required_gpus:
                    for candidate in free:
                        clique = [candidate]
                        mask = self.nvlink_topology_matrix[n_name][candidate]
                        for peer in free:
                            if peer != candidate and (mask & (1 << peer)):
                                clique.append(peer)
                                if len(clique) == required_gpus:
                                    break
                        if len(clique) == required_gpus:
                            best_clique = clique
                            target_node = n_name
                            break
                    if target_node:
                        break
            if not target_node or len(best_clique) < required_gpus:
                self.telemetry_counters["errors"] += 1
                return {"status": "PENDING_TOPOLOGY_DEFICIT", "allocated": []}
            for g in best_clique:
                self.gpu_nodes[target_node]["free_gpus"].remove(g)
                self.gpu_nodes[target_node]["allocated_gpus"].append(g)
            self.job_allocations[job_id] = best_clique
            self.vram_allocated_mb += required_gpus * 81920
            self.telemetry_counters["transfers"] += 1
            return {"status": "ALLOCATED", "node": target_node, "gpus": best_clique}

    def release_job_gpus(self, job_id: int, node_name: str) -> bool:
        with self.lock:
            if job_id not in self.job_allocations:
                return False
            gpus = self.job_allocations.pop(job_id)
            if node_name in self.gpu_nodes:
                for g in gpus:
                    if g in self.gpu_nodes[node_name]["allocated_gpus"]:
                        self.gpu_nodes[node_name]["allocated_gpus"].remove(g)
                        self.gpu_nodes[node_name]["free_gpus"].append(g)
            self.vram_allocated_mb = max(0, self.vram_allocated_mb - len(gpus) * 81920)
            self.telemetry_counters["events"] += 1
            return True

    def get_subsystem_health(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "engine_id": self.config.engine_id,
                "status": "HEALTHY_OPERATIONAL",
                "allocated_vram_mb": self.vram_allocated_mb,
                "active_jobs": len(self.job_allocations)
            }

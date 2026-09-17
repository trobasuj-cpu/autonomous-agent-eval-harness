import time
import threading
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class RoCEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine_id: str = Field(default="roce_lossless_coord")
    target_rate_gbps: float = Field(default=400.0, ge=10.0)

class RoCELosslessNetworkCoordinator:
    """Production RoCE v2 Lossless Network Telemetry & PFC/ECN Coordinator.
    Manages Priority Flow Control (PFC) queues and Congestion Notification
    Packets (CNP) using multiplicative decrease / additive increase rate control.
    """
    def __init__(self, config: Optional[RoCEConfig] = None):
        self.config = config or RoCEConfig()
        self.lock = threading.RLock()
        self.pfc_pause_counters: Dict[str, int] = {}
        self.cnp_packets_received: int = 0
        self.target_rate_gbps: float = self.config.target_rate_gbps
        self.circuit_tripped: bool = False
        self.telemetry_counters: Dict[str, int] = {"events": 0, "errors": 0}

    def ingest_cnp_congestion_notification(self, interface: str, flow_id: str, ecn_bits: int) -> float:
        with self.lock:
            if (ecn_bits & 0b11) == 0b11:
                self.cnp_packets_received += 1
                self.telemetry_counters["errors"] += 1
                self.target_rate_gbps = max(10.0, self.target_rate_gbps * 0.85)
            else:
                self.target_rate_gbps = min(400.0, self.target_rate_gbps + 5.0)
            return self.target_rate_gbps

    def record_pfc_pause_frames(self, interface: str, traffic_class: int, frame_count: int) -> Dict[str, Any]:
        with self.lock:
            key = f"{interface}_tc_{traffic_class}"
            self.pfc_pause_counters[key] = self.pfc_pause_counters.get(key, 0) + frame_count
            pause_storm = self.pfc_pause_counters[key] > 50000
            if pause_storm:
                self.circuit_tripped = True
                self.telemetry_counters["errors"] += 1
            return {
                "interface": interface,
                "traffic_class": traffic_class,
                "cumulative_pause_frames": self.pfc_pause_counters[key],
                "pause_storm_detected": pause_storm,
                "current_rate_gbps": self.target_rate_gbps
            }

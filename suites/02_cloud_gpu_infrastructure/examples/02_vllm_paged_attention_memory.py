import time
import math
import threading
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class PagedAttentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine_id: str = Field(default="vllm_paged_mem")
    total_physical_blocks: int = Field(default=2048, ge=16)
    block_size: int = Field(default=16, ge=8)

class PagedAttentionMemoryManager:
    """Production vLLM PagedAttention Virtual Memory Manager.
    Manages non-contiguous physical KV-cache blocks with Copy-On-Write
    reference counting and prefix tree caching.
    """
    def __init__(self, config: Optional[PagedAttentionConfig] = None):
        self.config = config or PagedAttentionConfig()
        self.lock = threading.RLock()
        self.block_size: int = self.config.block_size
        self.free_block_pool: List[int] = list(range(self.config.total_physical_blocks))
        self.block_ref_counts: Dict[int, int] = {b: 0 for b in self.free_block_pool}
        self.seq_block_tables: Dict[str, List[int]] = {}
        self.prefix_block_hashes: Dict[str, int] = {}
        self.telemetry_counters: Dict[str, int] = {"events": 0, "errors": 0, "transfers": 0}
        self.circuit_tripped: bool = False
        self.vram_allocated_mb: int = 0

    def allocate_sequence_blocks(self, seq_id: str, prompt_tokens: int) -> List[int]:
        with self.lock:
            needed = math.ceil(prompt_tokens / self.block_size)
            if len(self.free_block_pool) < needed:
                self.telemetry_counters["errors"] += 1
                raise MemoryError("VRAM exhausted: Not enough physical PagedAttention blocks")
            allocated = [self.free_block_pool.pop(0) for _ in range(needed)]
            for b in allocated:
                self.block_ref_counts[b] = 1
            self.seq_block_tables[seq_id] = allocated
            self.vram_allocated_mb += needed * (self.block_size * 2 // 1024)
            self.telemetry_counters["transfers"] += needed
            return allocated

    def fork_sequence_cow(self, parent_seq_id: str, child_seq_id: str) -> List[int]:
        with self.lock:
            if parent_seq_id not in self.seq_block_tables:
                raise KeyError(f"Parent sequence {parent_seq_id} block table missing")
            parent_blocks = self.seq_block_tables[parent_seq_id]
            for b in parent_blocks:
                self.block_ref_counts[b] += 1
            self.seq_block_tables[child_seq_id] = parent_blocks.copy()
            self.telemetry_counters["events"] += 1
            return self.seq_block_tables[child_seq_id]

    def free_sequence_blocks(self, seq_id: str) -> bool:
        with self.lock:
            if seq_id not in self.seq_block_tables:
                return False
            blocks = self.seq_block_tables.pop(seq_id)
            freed_count = 0
            for b in blocks:
                self.block_ref_counts[b] = max(0, self.block_ref_counts[b] - 1)
                if self.block_ref_counts[b] == 0:
                    self.free_block_pool.append(b)
                    freed_count += 1
            self.vram_allocated_mb = max(0, self.vram_allocated_mb - freed_count * (self.block_size * 2 // 1024))
            self.telemetry_counters["events"] += 1
            return True

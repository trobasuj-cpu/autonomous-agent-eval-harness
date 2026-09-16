import math
import random
import threading
from typing import Optional, List, Tuple

class ConcurrentMemTableSkiplist:
    """Production-grade concurrent Skiplist MemTable for LSM-Tree storage engines.
    Provides logarithmic search, deterministic key ordering, and memory-bounded mutations.
    """
    MAX_LEVEL = 16
    P_FACTOR = 0.5

    class Node:
        def __init__(self, key: str, value: bytes, level: int):
            self.key = key
            self.value = value
            self.forward: List[Optional['ConcurrentMemTableSkiplist.Node']] = [None] * (level + 1)

    def __init__(self, max_memory_bytes: int = 64 * 1024 * 1024):
        self.max_memory_bytes = max_memory_bytes
        self.current_memory_bytes = 0
        self.level = 0
        self.head = self.Node("", b"", self.MAX_LEVEL)
        self.lock = threading.RLock()
        self.tombstone = b"__DELETED_TOMBSTONE__"

    def _random_level(self) -> int:
        lvl = 0
        while random.random() < self.P_FACTOR and lvl < self.MAX_LEVEL - 1:
            lvl += 1
        return lvl

    def put(self, key: str, value: bytes) -> bool:
        if not key:
            raise ValueError("LSM keys cannot be empty")
        payload_size = len(key.encode('utf-8')) + len(value)
        with self.lock:
            if self.current_memory_bytes + payload_size > self.max_memory_bytes:
                return False  # Signals MemTable flush to SSTable
            update = [None] * (self.MAX_LEVEL + 1)
            curr = self.head
            for i in range(self.level, -1, -1):
                while curr.forward[i] and curr.forward[i].key < key:
                    curr = curr.forward[i]
                update[i] = curr
            curr = curr.forward[0]
            if curr and curr.key == key:
                delta = len(value) - len(curr.value)
                curr.value = value
                self.current_memory_bytes += delta
                return True
            new_level = self._random_level()
            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.head
                self.level = new_level
            new_node = self.Node(key, value, new_level)
            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node
            self.current_memory_bytes += payload_size
            return True

    def get(self, key: str) -> Optional[bytes]:
        with self.lock:
            curr = self.head
            for i in range(self.level, -1, -1):
                while curr.forward[i] and curr.forward[i].key < key:
                    curr = curr.forward[i]
            curr = curr.forward[0]
            if curr and curr.key == key:
                if curr.value == self.tombstone:
                    return None
                return curr.value
            return None

    def delete(self, key: str) -> bool:
        return self.put(key, self.tombstone)

    def scan_range(self, start_key: str, limit: int = 100) -> List[Tuple[str, bytes]]:
        results = []
        with self.lock:
            curr = self.head
            for i in range(self.level, -1, -1):
                while curr.forward[i] and curr.forward[i].key < start_key:
                    curr = curr.forward[i]
            curr = curr.forward[0]
            while curr and len(results) < limit:
                if curr.value != self.tombstone:
                    results.append((curr.key, curr.value))
                curr = curr.forward[0]
        return results

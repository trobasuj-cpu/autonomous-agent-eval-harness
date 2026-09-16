import math
import random
import heapq
import threading
from typing import List, Tuple, Dict, Optional

class HNSWVectorGraphIndex:
    """Production-grade Hierarchical Navigable Small World (HNSW) vector search index.
    Features multi-layer proximity graphs, cosine similarity, and heuristic edge pruning.
    """
    def __init__(self, dim: int = 128, m: int = 16, ef_construction: int = 64, ml: float = 0.62):
        self.dim = dim
        self.m = m
        self.m_max0 = m * 2
        self.ef_construction = ef_construction
        self.ml = ml
        self.entry_point: Optional[int] = None
        self.max_level = -1
        self.nodes: Dict[int, List[float]] = {}
        self.graph: Dict[int, Dict[int, List[int]]] = {}
        self.lock = threading.RLock()

    def _distance(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 1.0
        return 1.0 - max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    def _random_level(self) -> int:
        r = -math.log(max(1e-9, random.random())) * self.ml
        return min(16, int(r))

    def _search_layer(self, query: List[float], ep: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        visited = set(ep)
        candidates: List[Tuple[float, int]] = []
        w: List[Tuple[float, int]] = []
        for p in ep:
            dist = self._distance(query, self.nodes[p])
            heapq.heappush(candidates, (dist, p))
            heapq.heappush(w, (-dist, p))
            if len(w) > ef:
                heapq.heappop(w)

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_w_dist = -w[0][0]
            if c_dist > furthest_w_dist:
                break
            neighbors = self.graph.get(c_id, {}).get(level, [])
            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    furthest_w_dist = -w[0][0]
                    n_dist = self._distance(query, self.nodes[n])
                    if n_dist < furthest_w_dist or len(w) < ef:
                        heapq.heappush(candidates, (n_dist, n))
                        heapq.heappush(w, (-n_dist, n))
                        if len(w) > ef:
                            heapq.heappop(w)
        return sorted([(-d, p) for d, p in w], key=lambda x: x[0])

    def insert(self, node_id: int, vector: List[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {len(vector)}")
        with self.lock:
            self.nodes[node_id] = vector
            self.graph[node_id] = {}
            if self.entry_point is None:
                self.entry_point = node_id
                self.max_level = 0
                self.graph[node_id][0] = []
                return

            curr_obj = [self.entry_point]
            target_level = self._random_level()
            max_l = self.max_level

            for lvl in range(max_l, target_level, -1):
                best = self._search_layer(vector, curr_obj, ef=1, level=lvl)
                curr_obj = [best[0][1]]

            for lvl in range(min(max_l, target_level), -1, -1):
                m_curr = self.m_max0 if lvl == 0 else self.m
                candidates = self._search_layer(vector, curr_obj, ef=self.ef_construction, level=lvl)
                neighbors = [p for _, p in candidates[:m_curr]]
                self.graph[node_id][lvl] = neighbors
                for n in neighbors:
                    if lvl not in self.graph[n]:
                        self.graph[n][lvl] = []
                    self.graph[n][lvl].append(node_id)
                    if len(self.graph[n][lvl]) > m_curr:
                        dists = [(self._distance(self.nodes[n], self.nodes[x]), x) for x in self.graph[n][lvl]]
                        dists.sort(key=lambda x: x[0])
                        self.graph[n][lvl] = [x for _, x in dists[:m_curr]]
                curr_obj = [c for _, c in candidates]

            if target_level > self.max_level:
                self.max_level = target_level
                self.entry_point = node_id

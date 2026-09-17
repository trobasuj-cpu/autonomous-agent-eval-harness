import time
import importlib.util

print("=== BEATSPROM AGENT VERIFICATION HARNESS: SUITE 02 (CLOUD GPU INFRASTRUCTURE) ===")
start_t = time.perf_counter()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Test 1: Slurm GPU Topology Allocator
mod1 = load_module("ex01", "suites/02_cloud_gpu_infrastructure/examples/01_slurm_gpu_topology_allocator.py")
sched = mod1.SlurmGPUTopologyScheduler()
assert sched.register_gpu_node("node-h100-01", total_gpus=8, nvlink_peers=[(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)]) is True
alloc = sched.allocate_topology_aware(job_id=101, required_gpus=2)
assert alloc["status"] == "ALLOCATED"
assert len(alloc["gpus"]) == 2
assert sched.release_job_gpus(job_id=101, node_name="node-h100-01") is True
print("[PASS] Domain 01: Slurm GPU Topology Allocator (NVLink Clique Matrix)")

# Test 2: vLLM PagedAttention Memory Manager
mod2 = load_module("ex02", "suites/02_cloud_gpu_infrastructure/examples/02_vllm_paged_attention_memory.py")
vllm = mod2.PagedAttentionMemoryManager()
blocks = vllm.allocate_sequence_blocks("seq_req_01", prompt_tokens=64)
assert len(blocks) == 4
forked = vllm.fork_sequence_cow("seq_req_01", "seq_req_01_beam_1")
assert forked == blocks
assert vllm.block_ref_counts[blocks[0]] == 2
assert vllm.free_sequence_blocks("seq_req_01_beam_1") is True
assert vllm.block_ref_counts[blocks[0]] == 1
assert vllm.free_sequence_blocks("seq_req_01") is True
assert vllm.block_ref_counts[blocks[0]] == 0
print("[PASS] Domain 03: vLLM PagedAttention Virtual Memory Manager (CoW Block Tables)")

# Test 3: NCCL Ring AllReduce Coordinator
mod3 = load_module("ex03", "suites/02_cloud_gpu_infrastructure/examples/03_nccl_ring_allreduce_coordinator.py")
nccl = mod3.NCCLAllReduceRingCoordinator()
ring = nccl.construct_optimal_ring(world_size=4, rank_numa_nodes={0: 0, 1: 0, 2: 1, 3: 1})
assert len(ring) == 4
plan = nccl.slice_tensor_for_ring_allreduce(tensor_bytes=4096000, rank=0)
assert plan["world_size"] == 4
assert plan["chunk_size"] == 1024000
print("[PASS] Domain 06: NCCL Distributed AllReduce Ring Coordinator (NUMA Locality)")

# Test 4: RoCE v2 Lossless Telemetry Coordinator
mod4 = load_module("ex04", "suites/02_cloud_gpu_infrastructure/examples/04_roce_v2_lossless_telemetry.py")
roce = mod4.RoCELosslessNetworkCoordinator()
rate = roce.ingest_cnp_congestion_notification("mlx5_0", "flow_0_1", ecn_bits=0b11)
assert rate < 400.0
pfc = roce.record_pfc_pause_frames("mlx5_0", traffic_class=3, frame_count=100)
assert pfc["cumulative_pause_frames"] == 100
assert pfc["pause_storm_detected"] is False
print("[PASS] Domain 08: RoCE v2 Lossless Network Telemetry & PFC/ECN Coordinator")

elapsed = time.perf_counter() - start_t
print("-" * 75)
print(f"Ran 4 core infrastructure suites in {elapsed:.3f}s - ALL VERIFIERS PASSED [REWARD = 1.0]")

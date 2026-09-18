import time
import importlib.util

print("=== BEATSPROM AGENT VERIFICATION HARNESS: SUITE 03 (LINUX KERNEL, eBPF & XDP DATAPLANE) ===")
start_t = time.perf_counter()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Test 1: AF_XDP Zero-Copy Packet Driver
mod1 = load_module("ex01", "suites/03_linux_kernel_ebpf/examples/01_af_xdp_packet_ring_driver.py")
driver = mod1.AFXDPZeroCopyDriver()
assert driver.register_umem_frame(0, b"\x45\x00\x00\x28" * 16) is True
assert driver.submit_rx_packet(0) is True
batch = driver.drain_rx_batch(16)
assert batch == [0]
print("[PASS] Domain 01: AF_XDP Zero-Copy Packet Ring Driver & UMEM Frame Manager")

# Test 2: eBPF Verifier Complexity Guard
mod2 = load_module("ex02", "suites/03_linux_kernel_ebpf/examples/02_ebpf_verifier_complexity_guard.py")
verifier = mod2.BPFVerifierComplexityGuard()
assert verifier.verify_alu_bounds("r1", 4096) is True
assert verifier.check_stack_spill(offset=128, size=8) is True
assert verifier.check_stack_spill(offset=600, size=8) is False
print("[PASS] Domain 02: eBPF In-Kernel Verifier Boundary & Register Bounds Guard")

# Test 3: Sockops TCP Acceleration Engine
mod3 = load_module("ex03", "suites/03_linux_kernel_ebpf/examples/03_sockops_tcp_acceleration_engine.py")
sockops = mod3.SockopsTCPAccelerationEngine()
assert sockops.register_socket_connection("127.0.0.1:8080->127.0.0.1:52100", 12) is True
assert sockops.register_socket_connection("127.0.0.1:52100->127.0.0.1:8080", 14) is True
assert sockops.redirect_sk_msg("127.0.0.1:8080->127.0.0.1:52100", "127.0.0.1:52100->127.0.0.1:8080", b"HTTP/1.1 200 OK\r\n\r\n") is True
print("[PASS] Domain 06: Sockops TCP Congestion Acceleration Engine (Sockmap Bypass)")

# Test 4: XDP SYN-Flood DDoS Defense Shield
mod4 = load_module("ex04", "suites/03_linux_kernel_ebpf/examples/04_xdp_syn_flood_ddos_shield.py")
shield = mod4.XDPSYNFloodShield()
cookie = shield.generate_syn_cookie("10.0.0.5", "10.0.0.1", 52140, 80)
assert cookie > 0
assert shield.verify_ack_cookie("10.0.0.5", "10.0.0.1", 52140, 80, cookie) is True
shield.blacklist_malicious_subnet("192.0.2.")
assert shield.generate_syn_cookie("192.0.2.45", "10.0.0.1", 4000, 80) == 0
print("[PASS] Domain 10: XDP SYN-Flood Line-Rate DDoS Defense Shield (Stateless Cookies)")

elapsed = time.perf_counter() - start_t
print("-" * 75)
print(f"Ran 4 kernel & eBPF suites in {elapsed:.3f}s - ALL VERIFIERS PASSED [REWARD = 1.0]")

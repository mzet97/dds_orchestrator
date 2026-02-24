#!/usr/bin/env python3
"""
Benchmark 100% DDS — Multi-Client Performance Testing

Mede latência end-to-end do fluxo:
  Client DDS → Orchestrator DDS → Agent DDS → llama-server DDS

Modos de execução:
  - sync:     Clientes executam sequencialmente (1 por vez)
  - async:    Clientes concorrentes via asyncio (single thread)
  - parallel: Clientes em threads paralelas (ThreadPoolExecutor)

Uso:
  python benchmark_orchestrator_dds.py --mode all --clients 1 5 10 25 50 100
  python benchmark_orchestrator_dds.py --mode parallel --clients 1 5 --requests 65
  python benchmark_orchestrator_dds.py --mode sync --clients 1 --requests 1 --verbose
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ===========================================================================
# BenchmarkDDSClient
# ===========================================================================

class BenchmarkDDSClient:
    """
    Isolated DDS client for benchmarking.
    Each instance creates its own DomainParticipant for full isolation.
    """

    def __init__(self, client_id: str, domain_id: int = 0, quiet: bool = False):
        self.client_id = client_id
        self.domain_id = domain_id
        self.quiet = quiet
        self.dds_available = False
        self._init_dds()

    def _init_dds(self):
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.topic import Topic
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader
        from cyclonedds.core import Policy
        from cyclonedds.qos import Qos
        from cyclonedds.util import duration

        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        from orchestrator import ClientRequest, ClientResponse

        self._ClientRequest = ClientRequest
        self._ClientResponse = ClientResponse

        self.participant = DomainParticipant(self.domain_id)

        self.topic_request = Topic(self.participant, "client/request", ClientRequest)
        self.topic_response = Topic(self.participant, "client/response", ClientResponse)

        qos = Qos(
            Policy.Reliability.Reliable(duration(seconds=10)),
            Policy.Durability.Volatile,
            Policy.History.KeepLast(1),
        )

        self.writer = DataWriter(self.participant, self.topic_request, qos)
        self.reader = DataReader(self.participant, self.topic_response, qos)
        self.dds_available = True

        if not self.quiet:
            print(f"  [{self.client_id}] DDS initialized on domain {self.domain_id}")

    def send_and_measure(self, messages: List[Dict], timeout_s: float = 120.0) -> Dict[str, Any]:
        """
        Send request via DDS, poll response, return timing metrics.
        Uses time.perf_counter() for high-resolution measurement.
        """
        if not self.dds_available:
            return {"rtt_ms": -1, "success": False, "error_message": "DDS not available"}

        request_id = str(uuid.uuid4())
        messages_json = json.dumps(messages)

        req = self._ClientRequest(
            request_id=request_id,
            client_id=self.client_id,
            task_type="chat",
            messages_json=messages_json,
            priority=2,
            timeout_ms=int(timeout_s * 1000),
            requires_context=False,
        )

        # --- HIGH-RESOLUTION TIMING ---
        t0 = time.perf_counter()
        self.writer.write(req)

        # Poll for response matching our request_id
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed > timeout_s:
                return {
                    "rtt_ms": -1,
                    "success": False,
                    "error_message": "Timeout",
                    "request_id": request_id,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "processing_time_ms": 0,
                }
            try:
                samples = self.reader.take()
                for sample in samples:
                    if sample and getattr(sample, "request_id", None) == request_id:
                        t1 = time.perf_counter()
                        rtt_ms = (t1 - t0) * 1000.0
                        return {
                            "rtt_ms": round(rtt_ms, 3),
                            "success": bool(getattr(sample, "success", False)),
                            "prompt_tokens": getattr(sample, "prompt_tokens", 0),
                            "completion_tokens": getattr(sample, "completion_tokens", 0),
                            "processing_time_ms": getattr(sample, "processing_time_ms", 0),
                            "error_message": getattr(sample, "error_message", ""),
                            "request_id": request_id,
                        }
            except Exception:
                pass
            time.sleep(0.005)  # 5ms poll interval

    def warmup(self, messages: List[Dict], count: int = 3, timeout_s: float = 120.0):
        """Send throwaway requests to prime DDS discovery + GPU cache."""
        results = []
        for i in range(count):
            result = self.send_and_measure(messages, timeout_s)
            rtt = result.get("rtt_ms", -1)
            results.append(rtt)
            if not self.quiet:
                status = f"{rtt:.0f}ms" if rtt > 0 else "FAIL"
                print(f"  [{self.client_id}] Warmup {i+1}/{count}: {status}")
        valid = [r for r in results if r > 0]
        return valid

    def close(self):
        if hasattr(self, "participant"):
            del self.reader
            del self.writer
            del self.topic_response
            del self.topic_request
            del self.participant
            self.dds_available = False


# ===========================================================================
# CSV Writer
# ===========================================================================

RAW_COLUMNS = [
    "scenario", "mode", "num_clients", "client_id",
    "request_num", "rtt_ms", "success",
    "prompt_tokens", "completion_tokens",
    "processing_time_ms", "timestamp",
]

SUMMARY_COLUMNS = [
    "scenario", "mode", "num_clients", "total_requests",
    "successful", "failed",
    "mean_ms", "stddev_ms", "min_ms", "max_ms",
    "p50_ms", "p90_ms", "p95_ms", "p99_ms",
    "cv", "throughput_rps", "wall_time_s",
]


def write_raw_row(filepath: str, row: Dict):
    """Append single row to per-client CSV (real-time, survives crashes)."""
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def compute_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute statistical summary from a list of RTT values (ms)."""
    if not latencies:
        return {k: 0.0 for k in [
            "mean_ms", "stddev_ms", "min_ms", "max_ms",
            "p50_ms", "p90_ms", "p95_ms", "p99_ms", "cv"
        ]}

    s = sorted(latencies)
    n = len(s)

    mean = statistics.mean(s)
    stddev = statistics.stdev(s) if n > 1 else 0.0

    return {
        "mean_ms": round(mean, 3),
        "stddev_ms": round(stddev, 3),
        "min_ms": round(s[0], 3),
        "max_ms": round(s[-1], 3),
        "p50_ms": round(s[int(n * 0.50)], 3),
        "p90_ms": round(s[min(int(n * 0.90), n - 1)], 3),
        "p95_ms": round(s[min(int(n * 0.95), n - 1)], 3),
        "p99_ms": round(s[min(int(n * 0.99), n - 1)], 3),
        "cv": round(stddev / mean, 4) if mean > 0 else 0.0,
    }


def write_summary_row(filepath: str, row: Dict):
    """Append one scenario row to summary.csv."""
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ===========================================================================
# Execution Modes
# ===========================================================================

def run_sync(
    num_clients: int,
    requests_per_client: int,
    messages: List[Dict],
    domain_id: int,
    timeout_s: float,
    warmup_count: int,
    output_dir: str,
    verbose: bool,
) -> Dict:
    """
    Synchronous mode: clients execute one after another.
    Each client: create → warmup → N requests → close.
    """
    scenario = f"sync_c{num_clients}"
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    all_latencies = []
    total_success = 0
    total_fail = 0
    wall_start = time.perf_counter()

    for ci in range(num_clients):
        client_id = f"client_{ci:03d}"
        client = BenchmarkDDSClient(client_id, domain_id, quiet=not verbose)

        # Warmup
        print(f"[Warmup] {client_id}: {warmup_count} requests...")
        client.warmup(messages, warmup_count, timeout_s)

        # Discovery delay after warmup
        time.sleep(1.0)

        csv_path = os.path.join(raw_dir, f"{scenario}_{client_id}.csv")
        client_latencies = []

        for ri in range(requests_per_client):
            result = client.send_and_measure(messages, timeout_s)
            rtt = result["rtt_ms"]
            success = result["success"] and rtt > 0

            row = {
                "scenario": scenario,
                "mode": "sync",
                "num_clients": num_clients,
                "client_id": client_id,
                "request_num": ri + 1,
                "rtt_ms": rtt,
                "success": success,
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "processing_time_ms": result.get("processing_time_ms", 0),
                "timestamp": time.time(),
            }
            write_raw_row(csv_path, row)

            if success:
                all_latencies.append(rtt)
                client_latencies.append(rtt)
                total_success += 1
            else:
                total_fail += 1

            if verbose or (ri + 1) % 10 == 0 or ri == 0:
                avg = statistics.mean(client_latencies) if client_latencies else 0
                print(
                    f"[Bench]  {client_id}: {ri+1}/{requests_per_client} "
                    f"| last={rtt:.0f}ms avg={avg:.0f}ms"
                )

        client.close()

    wall_time = time.perf_counter() - wall_start
    stats = compute_stats(all_latencies)
    stats.update({
        "scenario": scenario,
        "mode": "sync",
        "num_clients": num_clients,
        "total_requests": num_clients * requests_per_client,
        "successful": total_success,
        "failed": total_fail,
        "throughput_rps": round(total_success / wall_time, 4) if wall_time > 0 else 0,
        "wall_time_s": round(wall_time, 2),
    })

    print(
        f"[Done]   {scenario}: {total_success + total_fail} reqs in {wall_time:.1f}s "
        f"| mean={stats['mean_ms']:.0f}ms p50={stats['p50_ms']:.0f}ms "
        f"p95={stats['p95_ms']:.0f}ms"
    )
    return stats


def _parallel_worker(
    client_id: str,
    num_clients: int,
    requests_per_client: int,
    messages: List[Dict],
    domain_id: int,
    timeout_s: float,
    warmup_count: int,
    raw_dir: str,
    scenario: str,
    verbose: bool,
    progress_lock: threading.Lock,
) -> List[float]:
    """Worker function for parallel and async modes."""
    client = BenchmarkDDSClient(client_id, domain_id, quiet=True)

    # Warmup
    client.warmup(messages, warmup_count, timeout_s)
    time.sleep(1.0)

    csv_path = os.path.join(raw_dir, f"{scenario}_{client_id}.csv")
    latencies = []

    for ri in range(requests_per_client):
        result = client.send_and_measure(messages, timeout_s)
        rtt = result["rtt_ms"]
        success = result["success"] and rtt > 0

        row = {
            "scenario": scenario,
            "mode": scenario.split("_")[0],  # extract mode from scenario name
            "num_clients": num_clients,
            "client_id": client_id,
            "request_num": ri + 1,
            "rtt_ms": rtt,
            "success": success,
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "processing_time_ms": result.get("processing_time_ms", 0),
            "timestamp": time.time(),
        }
        write_raw_row(csv_path, row)

        if success:
            latencies.append(rtt)

        if verbose or (ri + 1) % 10 == 0:
            with progress_lock:
                avg = statistics.mean(latencies) if latencies else 0
                print(
                    f"[Bench]  {client_id}: {ri+1}/{requests_per_client} "
                    f"| last={rtt:.0f}ms avg={avg:.0f}ms"
                )

    client.close()
    return latencies


def run_parallel(
    num_clients: int,
    requests_per_client: int,
    messages: List[Dict],
    domain_id: int,
    timeout_s: float,
    warmup_count: int,
    output_dir: str,
    verbose: bool,
) -> Dict:
    """
    Parallel mode: all clients run simultaneously in threads.
    Uses ThreadPoolExecutor(max_workers=num_clients).
    """
    scenario = f"parallel_c{num_clients}"
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    progress_lock = threading.Lock()
    wall_start = time.perf_counter()

    print(f"[Start]  Launching {num_clients} parallel clients...")

    all_latencies = []
    with ThreadPoolExecutor(max_workers=num_clients) as executor:
        futures = {}
        for ci in range(num_clients):
            client_id = f"client_{ci:03d}"
            future = executor.submit(
                _parallel_worker,
                client_id, num_clients, requests_per_client,
                messages, domain_id, timeout_s, warmup_count,
                raw_dir, scenario, verbose, progress_lock,
            )
            futures[future] = client_id

        for future in as_completed(futures):
            cid = futures[future]
            try:
                latencies = future.result()
                all_latencies.extend(latencies)
                print(f"[Finish] {cid}: {len(latencies)} successful requests")
            except Exception as e:
                print(f"[Error]  {cid}: {e}")

    wall_time = time.perf_counter() - wall_start
    total = num_clients * requests_per_client
    total_success = len(all_latencies)
    total_fail = total - total_success

    stats = compute_stats(all_latencies)
    stats.update({
        "scenario": scenario,
        "mode": "parallel",
        "num_clients": num_clients,
        "total_requests": total,
        "successful": total_success,
        "failed": total_fail,
        "throughput_rps": round(total_success / wall_time, 4) if wall_time > 0 else 0,
        "wall_time_s": round(wall_time, 2),
    })

    print(
        f"[Done]   {scenario}: {total} reqs in {wall_time:.1f}s "
        f"| mean={stats['mean_ms']:.0f}ms p50={stats['p50_ms']:.0f}ms "
        f"p95={stats['p95_ms']:.0f}ms tput={stats['throughput_rps']:.2f}rps"
    )
    return stats


def run_async(
    num_clients: int,
    requests_per_client: int,
    messages: List[Dict],
    domain_id: int,
    timeout_s: float,
    warmup_count: int,
    output_dir: str,
    verbose: bool,
) -> Dict:
    """
    Async mode: all clients run concurrently via asyncio (single thread).
    Uses run_in_executor to avoid blocking the event loop.
    """
    scenario = f"async_c{num_clients}"
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    async def _run():
        loop = asyncio.get_event_loop()

        # Create all clients
        clients = []
        for ci in range(num_clients):
            client_id = f"client_{ci:03d}"
            client = BenchmarkDDSClient(client_id, domain_id, quiet=True)
            clients.append((client_id, client))

        # Warmup all clients
        print(f"[Warmup] Warming up {num_clients} async clients...")
        for client_id, client in clients:
            await loop.run_in_executor(
                None, client.warmup, messages, warmup_count, timeout_s
            )
        await asyncio.sleep(1.0)

        # Define per-client coroutine
        async def client_worker(client_id: str, client: BenchmarkDDSClient) -> List[float]:
            csv_path = os.path.join(raw_dir, f"{scenario}_{client_id}.csv")
            latencies = []

            for ri in range(requests_per_client):
                result = await loop.run_in_executor(
                    None, client.send_and_measure, messages, timeout_s
                )
                rtt = result["rtt_ms"]
                success = result["success"] and rtt > 0

                row = {
                    "scenario": scenario,
                    "mode": "async",
                    "num_clients": num_clients,
                    "client_id": client_id,
                    "request_num": ri + 1,
                    "rtt_ms": rtt,
                    "success": success,
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "completion_tokens": result.get("completion_tokens", 0),
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "timestamp": time.time(),
                }
                write_raw_row(csv_path, row)

                if success:
                    latencies.append(rtt)

                if verbose or (ri + 1) % 10 == 0:
                    avg = statistics.mean(latencies) if latencies else 0
                    print(
                        f"[Bench]  {client_id}: {ri+1}/{requests_per_client} "
                        f"| last={rtt:.0f}ms avg={avg:.0f}ms"
                    )

            return latencies

        # Run all clients concurrently
        print(f"[Start]  Launching {num_clients} async clients...")
        wall_start = time.perf_counter()

        tasks = [
            client_worker(client_id, client)
            for client_id, client in clients
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        wall_time = time.perf_counter() - wall_start

        # Collect results
        all_latencies = []
        for i, res in enumerate(results):
            cid = clients[i][0]
            if isinstance(res, Exception):
                print(f"[Error]  {cid}: {res}")
            else:
                all_latencies.extend(res)
                print(f"[Finish] {cid}: {len(res)} successful requests")

        # Cleanup
        for _, client in clients:
            client.close()

        return all_latencies, wall_time

    all_latencies, wall_time = asyncio.run(_run())

    total = num_clients * requests_per_client
    total_success = len(all_latencies)
    total_fail = total - total_success

    stats = compute_stats(all_latencies)
    stats.update({
        "scenario": scenario,
        "mode": "async",
        "num_clients": num_clients,
        "total_requests": total,
        "successful": total_success,
        "failed": total_fail,
        "throughput_rps": round(total_success / wall_time, 4) if wall_time > 0 else 0,
        "wall_time_s": round(wall_time, 2),
    })

    print(
        f"[Done]   {scenario}: {total} reqs in {wall_time:.1f}s "
        f"| mean={stats['mean_ms']:.0f}ms p50={stats['p50_ms']:.0f}ms "
        f"p95={stats['p95_ms']:.0f}ms tput={stats['throughput_rps']:.2f}rps"
    )
    return stats


# ===========================================================================
# Main
# ===========================================================================

def print_header(mode: str, num_clients: int, requests_per_client: int):
    print("=" * 65)
    print(f" DDS Orchestrator Benchmark")
    print(f" Mode: {mode} | Clients: {num_clients} | Requests/client: {requests_per_client}")
    print("=" * 65)


def print_env_check():
    """Print environment requirements."""
    print("Environment check:")
    cyclone_uri = os.environ.get("CYCLONEDDS_URI", "NOT SET")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "NOT SET")
    print(f"  CYCLONEDDS_URI: {cyclone_uri}")
    print(f"  LD_LIBRARY_PATH: {'SET' if ld_path != 'NOT SET' else 'NOT SET'}")

    # Check CycloneDDS import
    try:
        import cyclonedds
        print(f"  CycloneDDS: OK")
    except ImportError:
        print("  CycloneDDS: NOT FOUND - install cyclonedds-python")
        sys.exit(1)

    # Check orchestrator types
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from orchestrator import ClientRequest, ClientResponse
        print("  IDL types: OK (ClientRequest, ClientResponse)")
    except ImportError as e:
        print(f"  IDL types: FAILED ({e})")
        sys.exit(1)

    print()


def estimate_time(modes: List[str], client_counts: List[int], requests: int, gpu_time_s: float = 2.0):
    """Print estimated execution time."""
    total_s = 0
    for mode in modes:
        for nc in client_counts:
            total_req = nc * requests
            if mode == "sync":
                est = total_req * gpu_time_s
            else:
                # async/parallel: limited by GPU concurrency (~4 slots)
                est = (total_req / min(nc, 4)) * gpu_time_s
            total_s += est

    hours = total_s / 3600
    print(f"Estimated total time: ~{hours:.1f}h (assuming ~{gpu_time_s}s/request on GPU)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 100%% DDS Orchestrator — Multi-Client Performance Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (1 client, 1 request)
  python benchmark_orchestrator_dds.py --mode sync --clients 1 --requests 1 --verbose

  # Full suite
  python benchmark_orchestrator_dds.py --mode all --clients 1 5 10 25 50 100

  # Quick parallel test
  python benchmark_orchestrator_dds.py --mode parallel --clients 1 5 10 --requests 10
        """,
    )
    parser.add_argument(
        "--mode", choices=["sync", "async", "parallel", "all"],
        required=True, help="Execution mode",
    )
    parser.add_argument(
        "--clients", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100],
        help="Client counts to test (default: 1 5 10 25 50 100)",
    )
    parser.add_argument("--requests", type=int, default=65, help="Requests per client (default: 65)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup requests per client (default: 3)")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain ID (default: 0)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per request in seconds (default: 120)")
    parser.add_argument("--output", type=str, default="results", help="Output directory (default: results/)")
    parser.add_argument("--prompt", type=str, default="What is 2+2?", help="Prompt text (default: 'What is 2+2?')")
    parser.add_argument("--max-tokens", type=int, default=10, help="Max tokens (default: 10)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenarios with existing results")
    parser.add_argument("--verbose", action="store_true", help="Print per-request details")
    args = parser.parse_args()

    # Setup
    modes = ["sync", "async", "parallel"] if args.mode == "all" else [args.mode]
    messages = [{"role": "user", "content": args.prompt}]
    output_dir = args.output
    summary_path = os.path.join(output_dir, "summary.csv")

    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "agg"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Header
    print()
    print("=" * 65)
    print(" DDS Orchestrator Benchmark — 100% DDS Flow")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()
    print_env_check()

    print(f"Configuration:")
    print(f"  Modes:    {modes}")
    print(f"  Clients:  {args.clients}")
    print(f"  Requests: {args.requests} per client")
    print(f"  Warmup:   {args.warmup} per client")
    print(f"  Prompt:   \"{args.prompt}\"")
    print(f"  Timeout:  {args.timeout}s")
    print(f"  Output:   {output_dir}/")
    print()
    estimate_time(modes, args.clients, args.requests)

    # Build scenario list
    scenarios = []
    for mode in modes:
        for nc in args.clients:
            scenarios.append((mode, nc))

    print(f"Total scenarios: {len(scenarios)}")
    print(f"Total requests:  {sum(nc * args.requests for _, nc in scenarios)}")
    print()

    # Execute
    all_stats = []
    for i, (mode, nc) in enumerate(scenarios):
        scenario_name = f"{mode}_c{nc}"

        # Skip if exists
        if args.skip_existing:
            raw_dir = os.path.join(output_dir, "raw")
            existing = [f for f in os.listdir(raw_dir) if f.startswith(scenario_name)]
            if len(existing) >= nc:
                print(f"[Skip]   {scenario_name} (results exist, use --no-skip to override)")
                continue

        print()
        print(f"--- Scenario {i+1}/{len(scenarios)}: {scenario_name} ---")
        print_header(mode, nc, args.requests)

        runner_kwargs = dict(
            num_clients=nc,
            requests_per_client=args.requests,
            messages=messages,
            domain_id=args.domain,
            timeout_s=args.timeout,
            warmup_count=args.warmup,
            output_dir=output_dir,
            verbose=args.verbose,
        )

        if mode == "sync":
            stats = run_sync(**runner_kwargs)
        elif mode == "async":
            stats = run_async(**runner_kwargs)
        elif mode == "parallel":
            stats = run_parallel(**runner_kwargs)

        all_stats.append(stats)
        write_summary_row(summary_path, stats)

    # Final summary
    print()
    print("=" * 65)
    print(" FINAL SUMMARY")
    print("=" * 65)
    print()
    print(f"{'Scenario':<20} {'Reqs':>6} {'OK':>5} {'Fail':>5} "
          f"{'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Tput':>8} {'Time':>8}")
    print("-" * 100)
    for s in all_stats:
        print(
            f"{s['scenario']:<20} {s['total_requests']:>6} {s['successful']:>5} {s['failed']:>5} "
            f"{s['mean_ms']:>7.0f}ms {s['p50_ms']:>7.0f}ms {s['p95_ms']:>7.0f}ms "
            f"{s['p99_ms']:>7.0f}ms {s['throughput_rps']:>7.2f} {s['wall_time_s']:>7.1f}s"
        )
    print()
    print(f"Results saved to: {output_dir}/")
    print(f"Summary CSV:      {summary_path}")
    print(f"Raw CSVs:         {output_dir}/raw/")
    print()
    print("Run analysis: python analyze_benchmark.py --input results/")


if __name__ == "__main__":
    main()

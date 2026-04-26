"""
E4: Multi-client scalability — full DDS native (Bug 4 fix).

Cada cliente é spawned em **subprocess separado** (1 DomainParticipant
por processo) para isolar o crash do CycloneDDS Python wrapper que
segfault-a com >2 participants no mesmo processo.

CLIENT_COUNTS: [1, 2, 4, 8] como antes.
Cada execução produz CSV consolidado por contagem de clientes e summary
único agregando todas as combinações.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

CLIENT_COUNTS = [1, 2, 4, 8]
WORKER = Path(__file__).parent / "_e4_dds_one_client.py"


def run_subfase(num_clientes: int, n_per_client: int) -> dict | None:
    """Spawn `num_clientes` subprocess de _e4_dds_one_client.py em paralelo,
    aguarda todos, agrega resultados."""
    print(f"\n--- {num_clientes} client(s) ---")
    procs = []
    wall_start = time.perf_counter()
    for cid in range(num_clientes):
        p = subprocess.Popen(
            [sys.executable, str(WORKER),
             "--client-id", str(cid),
             "--n", str(n_per_client),
             "--num-clientes", str(num_clientes)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        procs.append(p)

    # Wait for all
    worker_outputs = []
    for p in procs:
        out, err = p.communicate(timeout=3600)
        worker_outputs.append((p.returncode, out, err))

    wall_dur = time.perf_counter() - wall_start

    # Aggregate by reading each worker's CSV
    all_rows = []
    n_workers_ok = 0
    for cid, (rc, out, err) in enumerate(worker_outputs):
        if rc != 0:
            print(f"  client {cid}: worker FAIL rc={rc} stderr={err.strip()[:300]}")
            continue
        n_workers_ok += 1
        csv_path = Path("results") / f"E4_DDS_NATIVE_one_client_{num_clientes}_c{cid}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append({
                    "client": int(r["client"]),
                    "latency_ms": float(r["latency_ms"]),
                    "success": int(r["success"]) == 1,
                })

    successes = [r for r in all_rows if r["success"]]
    latencies = sorted(r["latency_ms"] for r in successes)
    n_succ = len(latencies)
    if n_succ == 0:
        print("  ALL FAILED")
        return None

    p_pct = lambda q: latencies[min(int(len(latencies) * q), len(latencies) - 1)]
    summary = {
        "protocol": "DDS_NATIVE",
        "phase": "A",
        "num_agentes": 1,
        "num_clientes": num_clientes,
        "workers_spawned": num_clientes,
        "workers_ok": n_workers_ok,
        "total_requests": len(all_rows),
        "successful_requests": n_succ,
        "throughput_req_s": round(n_succ / wall_dur, 3),
        "wall_dur_s": round(wall_dur, 2),
        "latency_p50_ms": round(p_pct(0.50), 2),
        "latency_p95_ms": round(p_pct(0.95), 2),
        "latency_p99_ms": round(p_pct(0.99), 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_stdev_ms": round(statistics.stdev(latencies), 2) if n_succ > 1 else 0,
    }
    print(f"  workers ok: {n_workers_ok}/{num_clientes}  throughput: {summary['throughput_req_s']:.2f} req/s")
    print(f"  p50: {summary['latency_p50_ms']:.1f}ms p95: {summary['latency_p95_ms']:.1f}ms p99: {summary['latency_p99_ms']:.1f}ms")

    # CSV consolidado da subfase
    consolidated = Path("results") / f"E4_DDS_NATIVE_faseA_1ag_{num_clientes}cl.csv"
    with open(consolidated, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client", "latency_ms", "success"])
        for r in all_rows:
            w.writerow([r["client"], r["latency_ms"], 1 if r["success"] else 0])
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="N requests per client (v3: N=1000)")
    args = p.parse_args()

    print("E4: Escalabilidade - DDS NATIVE (subprocess-isolated clients)")
    print(f"Client counts: {CLIENT_COUNTS}, N per client: {args.n}")
    print("=" * 60)

    Path("results").mkdir(exist_ok=True)
    all_summaries = []
    for num_clientes in CLIENT_COUNTS:
        s = run_subfase(num_clientes, args.n)
        if s:
            all_summaries.append(s)

    json_file = Path("results") / "E4_DDS_NATIVE_faseA_1ag_summary.json"
    with open(json_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nJSON: {json_file}")


if __name__ == "__main__":
    main()

"""
Worker subprocess para 1 cliente DDS no E4 (Bug 4 fix).

Crash-isolation: cada cliente roda em processo Python separado para
evitar SEGFAULT do CycloneDDS quando >2 DomainParticipants coexistem
no mesmo processo. Cada worker grava seu próprio CSV em
results/E4_DDS_NATIVE_one_client_<num_clientes>_c<id>.csv e termina
com exit code 0 em sucesso.

Uso (não invocado direto pelo usuário, é spawned pelo E4 master):
    python _e4_dds_one_client.py --client-id 3 --n 1000
"""
import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dds-automation" / "bench"))
from bench_dds_native_client import DDSClient


async def run(client_id: int, n_per_client: int, num_clientes: int):
    c = DDSClient(domain_id=0, client_id=f"e4-c{client_id}")
    c.bind_loop(asyncio.get_running_loop())
    await c.wait_for_discovery(timeout_s=5)
    rows = []
    for _ in range(n_per_client):
        try:
            wall, resp = await c.request(
                [{"role": "user", "content": "Hi"}],
                max_tokens=1, temperature=0.0, timeout_s=120,
            )
            success = resp is not None and getattr(resp, "success", True)
            rows.append((client_id, wall, 1 if success else 0))
        except Exception as e:
            rows.append((client_id, 0, 0))

    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"E4_DDS_NATIVE_one_client_{num_clientes}_c{client_id}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client", "latency_ms", "success"])
        w.writerows(rows)
    # Per-worker summary on stdout for the master to parse
    succ = [r[1] for r in rows if r[2]]
    print(json.dumps({
        "client_id": client_id, "n": n_per_client,
        "csv": str(csv_path), "n_success": len(succ),
        "median_ms": (sorted(succ)[len(succ)//2] if succ else None),
    }), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--num-clientes", type=int, required=True,
                   help="Número total de clientes nesta sub-fase (para nome do CSV)")
    args = p.parse_args()
    asyncio.run(run(args.client_id, args.n, args.num_clientes))


if __name__ == "__main__":
    main()

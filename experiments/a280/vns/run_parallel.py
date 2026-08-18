from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_vns import SIZES, INSTANCES, SEEDS, TIME_BUDGET  # noqa: E402


def build_shards(n_shards, sizes, seeds, instances):
    n_inst = len(instances)
    units = [(n, s, TIME_BUDGET[n] * n_inst) for n in sizes for s in seeds]
    units.sort(key=lambda u: -u[2])                 # longest unit first
    bins = [defaultdict(list) for _ in range(n_shards)]
    load = [0.0] * n_shards
    for n, s, cost in units:
        k = min(range(n_shards), key=lambda i: load[i])
        bins[k][n].append(s)
        load[k] += cost
    shards = [{n: sorted(v) for n, v in b.items()} for b in bins]
    return shards, load


def merge_csvs(shard_paths, out_path):
    header_written = False
    n_rows = 0
    with open(out_path, "w", newline="") as out:
        writer = None
        for p in shard_paths:
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                continue
            with open(p, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    continue
                if not header_written:
                    writer = csv.writer(out)
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    n_rows += 1
    return n_rows


def main():
    ap = argparse.ArgumentParser(description="Parallel VNS sweep launcher.")
    ap.add_argument("--shards", type=int, default=4,
                    help="number of parallel processes = number of vCPUs "
                         "(default 4)")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--instances", type=int, default=len(INSTANCES))
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", default=os.path.join(HERE, "results_vns.csv"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the shard plan and exit (launch nothing)")
    cli = ap.parse_args()

    sizes = [int(s) for s in cli.sizes.split(",")]
    seeds = list(range(cli.seeds))
    instances = list(range(1, cli.instances + 1))

    shards, load = build_shards(cli.shards, sizes, seeds, instances)
    total_h = sum(load) / 3600.0

    print("=" * 72)
    print(f"PARALLEL VNS SWEEP  shards={cli.shards}  "
          f"total={total_h:.1f} core-h  wall~={max(load)/3600:.1f} h")
    print("=" * 72)
    for k, sh in enumerate(shards):
        parts = "  ".join(f"n{n}:{sh[n]}" for n in sorted(sh))
        print(f"shard {k}: {load[k]/3600:5.2f} h | {parts}")
    print("=" * 72)

    if cli.dry_run:
        print("dry-run: nothing launched.")
        return

    procs = []
    shard_csvs = []
    for k, sh in enumerate(shards):
        csv_path = os.path.join(HERE, f"results_vns_shard{k}.csv")
        log_path = os.path.join(HERE, f"run_vns_shard{k}.log")
        shard_csvs.append(csv_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)        # fresh start for this shard

        cmds = []
        for n in sorted(sh):
            seed_list = ",".join(str(s) for s in sh[n])
            cmds.append(
                f'{sys.executable} {os.path.join(HERE, "run_vns.py")} '
                f'--sizes {n} --instances {cli.instances} '
                f'--seed-list {seed_list} '
                f'--out {csv_path} --logfile {log_path}'
            )
        shell_cmd = " && ".join(cmds)
        print(f"launching shard {k} (pid will write {os.path.basename(csv_path)})")
        procs.append(subprocess.Popen(shell_cmd, shell=True))

    rc = 0
    for k, p in enumerate(procs):
        ret = p.wait()
        print(f"shard {k} exited with code {ret}")
        rc = rc or ret

    n_rows = merge_csvs(shard_csvs, os.path.abspath(cli.out))
    print("=" * 72)
    print(f"MERGED {n_rows} rows -> {os.path.abspath(cli.out)}")
    print("=" * 72)
    sys.exit(rc)


if __name__ == "__main__":
    main()

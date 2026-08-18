
from __future__ import annotations
import argparse
import csv
import os
import re
import subprocess
import sys


_ROOT = os.path.dirname(os.path.abspath(__file__))
while (_ROOT != os.path.dirname(_ROOT)
       and not os.path.isdir(os.path.join(_ROOT, "ttpd"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttpd import _paths  # noqa: E402
_paths.use("lisa/ttd300")

from ttpd.hub import ensure_local, weights_dir as hub_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

from ttd300_io import FULL_BUDGET, reference_sources          # noqa: E402

PY = sys.executable
SOLVE = os.path.join(HERE, "hybrid_solve.py")
OUT = os.path.join(HERE, "result.csv")

# this bundle's sweep
SIZES = [10, 20, 30, 40, 50, 75, 100]
LAYOUTS = [1]
ED_FRACS = [0.25, 0.50, 0.75, 1.00]
FRACS = [("5%", 0.05), ("10%", 0.10), ("20%", 0.20),
         ("33.33%", 1.0 / 3.0), ("50%", 0.50)]

def weights_for(n: int) -> str | None:
    p = os.path.join(hub_weights("ttd300", "lisa", "behaviour_cloning"),
                     f"n{n}.pt")
    return p if os.path.exists(p) else None
LOG_LINE = re.compile(
    r"w\d+\s+OK\s+(SA|VNS)\s+n=(\d+)\s+L(\d+)\s+(f\d+)\s+seed=\d+\s+"
    r"obj=\s*(-?[\d.]+)")

def load_reference():
    sa_best: dict[str, float] = {}
    all_best: dict[str, float] = {}

    def feed(method, label, obj):
        if method == "SA":
            sa_best[label] = max(sa_best.get(label, obj), obj)
        all_best[label] = max(all_best.get(label, obj), obj)

    csv_paths, log_paths = reference_sources()
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status", "ok") != "ok":
                    continue
                try:
                    feed(row["method"], row["instance"], float(row["objective"]))
                except (KeyError, ValueError):
                    pass

    for log_path in log_paths:
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            for line in f:
                m = LOG_LINE.search(line)
                if m:
                    method, n, L, ftag, obj = m.groups()
                    feed(method, f"ttd300_n{n}_L{L}_{ftag}", float(obj))

    print(f"[ref] {len(sa_best)} instances with an SA reference, "
          f"{len(all_best)} with any reference", flush=True)
    return sa_best, all_best

def done_cells(path):
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                seen.add((int(row["n"]), row["frac_label"]))
            except (KeyError, ValueError):
                pass
    return seen
LINE = re.compile(
    r"^(ttd300_n\d+_L\d+_f\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+"
    r"(-?[\d.]+)\s+([+-][\d.]+)\s+(\w+)")

def run_cell(n, frac_label, frac, weights, beam, n_aug, jobs):
    budget = FULL_BUDGET[n] * frac
    cmd = [PY, SOLVE, "--weights", weights, "--sizes", str(n),
           "--layouts", *map(str, LAYOUTS),
           "--ed-fracs", *map(str, ED_FRACS),
           "--budget", f"{budget:.4f}", "--decode", "beam",
           "--beam-width", str(beam), "--n-aug", str(n_aug),
           "--jobs", str(jobs), "--device", "cpu"]
    print(f"\n{'='*78}\n[cell] n={n} frac={frac_label} budget={budget:.2f}s "
          f"(jobs={jobs})\n  $ {' '.join(cmd)}\n{'='*78}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr)
        print(f"[cell] n={n} {frac_label} FAILED rc={proc.returncode}", flush=True)
    rows = []
    for line in proc.stdout.splitlines():
        m = LINE.match(line.strip())
        if m:
            inst, gat, sa, warm, portf, _imp, winner = m.groups()
            rows.append((inst, float(gat), float(sa), float(warm),
                         float(portf), winner))
    return budget, rows

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--fracs", type=str, nargs="+",
                    default=[lbl for lbl, _ in FRACS])
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--n-aug", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=2,
                    help="parallel SA solves (c3-standard-4 = 2 workers)")
    ap.add_argument("--out", type=str, default=OUT)
    args = ap.parse_args()

    fracs = [(lbl, f) for lbl, f in FRACS if lbl in set(args.fracs)]
    if not fracs:
        raise SystemExit(f"no valid --fracs among {[l for l, _ in FRACS]}")

    sa_ref, best_ref = load_reference()
    already = done_cells(args.out)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "n", "instance", "frac_label", "frac", "lisa_budget_s",
                "sa_full_budget_s", "GAT_beam", "SA_scratch_frac",
                "SA_warm_frac", "PORTFOLIO_frac", "winner",
                "SA_full_best", "BEST_REF",
                "LISA_gap_vs_BESTREF_%", "SA_full_gap_vs_BESTREF_%",
                "portfolio_vs_SAfull_%"])
            f.flush()
        for n in args.sizes:
            weights = weights_for(n)
            if weights is None:
                print(f"[skip] n={n}: no behaviour-cloning checkpoint n{n}.pt", flush=True)
                continue
            for frac_label, frac in fracs:
                if (n, frac_label) in already:
                    print(f"[skip] n={n} frac={frac_label}: already in "
                          f"{os.path.basename(args.out)}", flush=True)
                    continue
                budget, rows = run_cell(n, frac_label, frac, weights,
                                        args.beam_width, args.n_aug, args.jobs)
                for inst, gat, sa, warm, portf, winner in rows:
                    sa_full = sa_ref.get(inst, float("nan"))
                    bks = best_ref.get(inst, float("nan"))

                    def gap(v):
                        return (100.0 * (bks - v) / abs(bks)
                                if bks == bks and bks != 0 else float("nan"))
                    pf_vs_safull = (100.0 * (portf - sa_full) / abs(sa_full)
                                    if sa_full == sa_full and sa_full != 0
                                    else float("nan"))
                    w.writerow([
                        n, inst, frac_label, f"{frac:.4f}", f"{budget:.2f}",
                        f"{FULL_BUDGET[n]:.1f}", f"{gat:.1f}", f"{sa:.1f}",
                        f"{warm:.1f}", f"{portf:.1f}", winner,
                        f"{sa_full:.1f}", f"{bks:.1f}",
                        f"{gap(portf):.3f}", f"{gap(sa_full):.3f}",
                        f"{pf_vs_safull:+.3f}"])
                    f.flush()
    print(f"\n[done] wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()

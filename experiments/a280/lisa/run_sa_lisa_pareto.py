from __future__ import annotations
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

_REPO = HERE
while (_REPO != os.path.dirname(_REPO)
       and not os.path.isdir(os.path.join(_REPO, "ttpd"))):
    _REPO = os.path.dirname(_REPO)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from ttpd.hub import ensure_local, weights_dir as hub_weights  # noqa: E402

# Run the solver with whatever interpreter is running this script; set
# $TTPD_PYTHON to point at a different one (e.g. a venv with the LISA deps).
PY = os.environ.get("TTPD_PYTHON", sys.executable)
SOLVE = os.path.join(_REPO, "ttpd", "lisa", "a280", "hybrid_solve.py")
RESULTS = os.path.join(_REPO, "artifacts", "data", "results", "a280",
                       "Results.csv")
OUT = os.path.join(HERE, "sa_lisa.csv")

# full per-size SA budgets (seconds), matching Results.csv / z.csv
FULL_BUDGET = {5: 10.0, 10: 30.0, 15: 120.0, 20: 300.0, 30: 450.0, 40: 600.0, 50: 750.0}
# weight file per size (n=5 and n=10 share the jointly-trained checkpoint)
WEIGHTS = {
    5: "n5_n10.pt", 10: "n5_n10.pt", 15: "n15.pt", 20: "n20.pt",
    30: "n30.pt", 40: "n40.pt", 50: "n50.pt",
}
WDIR = hub_weights("a280", "lisa", "behaviour_cloning")
FRACS = [("5%", 0.05), ("10%", 0.10), ("20%", 0.20), ("25%", 0.25),
         ("33.33%", 1.0 / 3.0), ("50%", 0.50)]
INSTANCES = [1, 2, 3, 4]
SIZES = [5, 10, 15, 20, 30, 40, 50]


def load_sa_full_reference():
    """Map (n, 'bench_n_i') -> (SA_full_best, BKS) from Results.csv."""
    ref = {}
    with open(RESULTS) as f:
        for row in csv.DictReader(f):
            inst = row.get("instance", "")
            if not inst.startswith("bench_"):
                continue
            try:
                ref[(int(row["n"]), inst)] = (
                    float(row["SA_best"]), float(row["BKS"]))
            except (ValueError, KeyError):
                pass
    return ref


LINE = re.compile(
    r"^(bench_\d+_\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([+-][\d.]+)\s+(\w+)")


def run_cell(n, frac_label, frac):
    budget = FULL_BUDGET[n] * frac
    weights = ensure_local(os.path.join(WDIR, WEIGHTS[n]))
    cmd = [PY, SOLVE, "--weights", weights, "--sizes", str(n),
           "--instances", *map(str, INSTANCES),
           "--budget", f"{budget:.4f}", "--decode", "beam",
           "--beam-width", "128", "--n-aug", "8", "--device", "cpu"]
    print(f"\n{'='*78}\n[cell] n={n} frac={frac_label} budget={budget:.2f}s\n"
          f"  $ {' '.join(cmd)}\n{'='*78}", flush=True)
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
    ref = load_sa_full_reference()
    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "n", "instance", "frac_label", "frac", "lisa_budget_s",
                "sa_full_budget_s", "GAT_beam", "SA_scratch_frac",
                "SA_warm_frac", "PORTFOLIO_frac", "winner",
                "SA_full_best", "BKS",
                "LISA_gap_vs_BKS_%", "SA_full_gap_vs_BKS_%",
                "portfolio_vs_SAfull_%"])
        for n in SIZES:
            for frac_label, frac in FRACS:
                budget, rows = run_cell(n, frac_label, frac)
                for inst, gat, sa, warm, portf, winner in rows:
                    sa_full, bks = ref.get((n, inst), (float("nan"), float("nan")))
                    def gap(v):
                        return 100.0 * (bks - v) / abs(bks) if bks == bks and bks != 0 else float("nan")
                    lisa_gap = gap(portf)
                    safull_gap = gap(sa_full)
                    pf_vs_safull = (100.0 * (portf - sa_full) / abs(sa_full)
                                    if sa_full == sa_full and sa_full != 0 else float("nan"))
                    w.writerow([
                        n, inst, frac_label, f"{frac:.4f}", f"{budget:.2f}",
                        f"{FULL_BUDGET[n]:.1f}", f"{gat:.1f}", f"{sa:.1f}",
                        f"{warm:.1f}", f"{portf:.1f}", winner,
                        f"{sa_full:.1f}", f"{bks:.1f}",
                        f"{lisa_gap:.3f}", f"{safull_gap:.3f}", f"{pf_vs_safull:+.3f}"])
                    f.flush()
    print(f"\n[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

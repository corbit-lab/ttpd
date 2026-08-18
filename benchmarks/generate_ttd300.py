import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "artifacts", "bench", "instances", "ttd300")
BOX = 300
N_VALUES = [10, 20, 30, 40, 50, 75, 100]
LAYOUTS = [1, 2, 3, 4, 5]
FRACS = [0.25, 0.50, 0.75, 1.00]       
ITEMS_PER_NODE = 5
W_LO, W_HI = 1000, 1009
P_LO, P_HI = 1, 1000
R = 50.0
V_MIN, V_MAX = 0.1, 1
NL = "\r\n"
CAP_PER_NODE = 637010 / 280

def _fmt_speed(v):
    return str(int(v)) if float(v).is_integer() else ("%g" % v)


def make_instance(n, layout):
    rng = random.Random(1000 * n + layout)
    n_nodes = n + 1
    coords = [(rng.randint(0, BOX), rng.randint(0, BOX)) for _ in range(n_nodes)]
    items = []
    for node in range(2, n_nodes + 1):
        for _ in range(ITEMS_PER_NODE):
            items.append((rng.randint(P_LO, P_HI), rng.randint(W_LO, W_HI), node))
    return coords, items


def d_max_ceil2d(coords):
    m = 0
    for i in range(len(coords)):
        xi, yi = coords[i]
        for j in range(i + 1, len(coords)):
            xj, yj = coords[j]
            d = math.ceil(math.hypot(xi - xj, yi - yj))
            if d > m:
                m = d
    return m


def build_text(name, n, coords, items, ED, W):
    lines = [
        f"PROBLEM NAME: \t{name}",
        "KNAPSACK DATA TYPE: uncorrelated, similar weights",
        f"DIMENSION:\t{n + 1}",
        f"NUMBER OF ITEMS: \t{len(items)}",
        f"CAPACITY OF KNAPSACK: \t{W}",
        f"MIN SPEED: \t{_fmt_speed(V_MIN)}",
        f"MAX SPEED: \t{_fmt_speed(V_MAX)}",
        f"RENTING RATIO: \t{R:.2f}",
        "EDGE_WEIGHT_TYPE:\tCEIL_2D",
    ]
    if ED is not None:
        lines.append(f"DRONE ENDURANCE: \t{ED}")
    lines.append("NODE_COORD_SECTION\t(INDEX, X, Y): ")
    for i, (x, y) in enumerate(coords, start=1):
        lines.append(f"{i}\t{x}\t{y}")
    lines.append("ITEMS SECTION\t(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER): ")
    for idx, (profit, weight, node) in enumerate(items, start=1):
        lines.append(f"{idx}\t{profit}\t{weight}\t{node}")
    return NL.join(lines) + NL


def main():
    os.makedirs(OUT, exist_ok=True)
    # clear stale (old absolute-ED) files so the dataset is unambiguous
    for fn in os.listdir(OUT):
        if fn.startswith("ttd300_") and fn.endswith(".txt"):
            os.remove(os.path.join(OUT, fn))

    n_files = 0
    for n in N_VALUES:
        W = int(round(CAP_PER_NODE * n))   
        for L in LAYOUTS:
            coords, items = make_instance(n, L)
            dmax = d_max_ceil2d(coords)
            base = f"ttd300_n{n}_L{L}"
            with open(os.path.join(OUT, base + ".txt"), "w", newline="") as fh:
                fh.write(build_text(f"{base}-TTP", n, coords, items, ED=None, W=W))
            n_files += 1
            for frac in FRACS:
                ED = int(round(frac * dmax))
                tag = f"f{int(round(frac * 100)):03d}"     
                name = f"{base}_{tag}"
                with open(os.path.join(OUT, name + ".txt"), "w", newline="") as fh:
                    fh.write(build_text(f"{name}-TTP", n, coords, items, ED=ED, W=W))
                n_files += 1
        print(f"  n={n:<4} d_max(L1..L5)="
              f"{[d_max_ceil2d(make_instance(n, L)[0]) for L in LAYOUTS]}  "
              f"W={W}")

    with open(os.path.join(OUT, "README.txt"), "w", newline="") as fh:
        fh.write(NL.join([
            "ttd300 -- TTP-D drone-endurance benchmark (relative endurance)",
            f"sizes (customers) : {N_VALUES}",
            f"layouts per size  : {LAYOUTS}",
            f"endurance         : ED = round(frac * d_max), frac in {FRACS}",
            "                    d_max = instance's own max CEIL_2D pairwise distance",
            f"box               : uniform integer coords on [0,{BOX}]^2, CEIL_2D",
            f"items             : {ITEMS_PER_NODE}/customer, w~U[{W_LO},{W_HI}], p~U[{P_LO},{P_HI}]",
            f"capacity          : fixed per size W = round({CAP_PER_NODE:.4f} * n) "
            f"(a280 per-node ratio 637010/280); same W shared by L1..L5",
            f"speeds            : v_min={V_MIN}, v_max={V_MAX} (drone = 2x v_max downstream)",
            f"renting ratio R   : {R}",
            "filename _f025/_f050/_f075/_f100 = frac; absolute ED is in the header.",
            "the ...L<L>.txt file (no header) is the unbounded reference.",
        ]) + NL)

    print(f"done. {n_files} instance files -> {OUT}")


if __name__ == "__main__":
    main()

import math
import os
import random
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.abspath(__file__))
while (_ROOT != os.path.dirname(_ROOT)
       and not os.path.isdir(os.path.join(_ROOT, "ttpd"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttpd.hub import instance as hub_instance

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SOURCE = hub_instance("a280", "a280_benchmark.txt")
OUT_DIR = os.path.join(REPO, "artifacts", "bench", "instances", "a280")

N_VALUES = [5, 10, 15, 20, 30, 40, 50, 100]
SEEDS = [1, 2, 3, 4, 5]
N_CITIES_FULL = 280
DEPOT_ID = 1

NL = "\r\n" 


def parse_source(filepath):
    params = {}
    coords = {}
    node_items = defaultdict(list) 
    coord_section = items_section = False

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("PROBLEM NAME"):
                params["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("KNAPSACK DATA TYPE"):
                params["ktype"] = line.split(":", 1)[1].strip()
            elif line.startswith("DIMENSION"):
                params["n_nodes"] = int(line.split(":")[1].strip().split()[0])
            elif line.startswith("CAPACITY OF KNAPSACK"):
                params["W"] = float(line.split(":")[1].strip())
            elif line.startswith("MIN SPEED"):
                params["v_min"] = float(line.split(":")[1].strip())
            elif line.startswith("MAX SPEED"):
                params["v_max"] = float(line.split(":")[1].strip())
            elif line.startswith("RENTING RATIO"):
                params["R"] = float(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                params["edge_weight_type"] = line.split(":", 1)[1].strip()
            elif "NODE_COORD_SECTION" in line:
                coord_section, items_section = True, False
            elif "ITEMS SECTION" in line:
                items_section, coord_section = True, False
            elif coord_section:
                p = line.split()
                if len(p) == 3:
                    coords[int(p[0])] = (int(float(p[1])), int(float(p[2])))
            elif items_section:
                p = line.split()
                if len(p) == 4:
                    node_items[int(p[3])].append((int(p[1]), int(p[2])))

    return params, coords, node_items


def select_cities(n, candidate_nodes, seed):
    rng = random.Random(seed)
    return sorted(rng.sample(candidate_nodes, n))


def build_instance_text(name, params, coords, node_items, chosen_nodes):
    n = len(chosen_nodes)
    new_order = [DEPOT_ID] + list(chosen_nodes)
    old_to_new = {old: new for new, old in enumerate(new_order, start=1)}

    items = []
    for old in chosen_nodes:
        for (profit, weight) in node_items.get(old, []):
            items.append((profit, weight, old_to_new[old]))

    W_scaled = params["W"] * n / N_CITIES_FULL

    lines = []
    lines.append(f"PROBLEM NAME: \t{name}")
    lines.append(f"KNAPSACK DATA TYPE: {params['ktype']}")
    lines.append(f"DIMENSION:\t{n + 1}")
    lines.append(f"NUMBER OF ITEMS: \t{len(items)}")
    lines.append(f"CAPACITY OF KNAPSACK: \t{int(round(W_scaled))}")
    lines.append(f"MIN SPEED: \t{_fmt_speed(params['v_min'])}")
    lines.append(f"MAX SPEED: \t{_fmt_speed(params['v_max'])}")
    lines.append(f"RENTING RATIO: \t{params['R']:.2f}")
    lines.append(f"EDGE_WEIGHT_TYPE:\t{params['edge_weight_type']}")
    lines.append("NODE_COORD_SECTION\t(INDEX, X, Y): ")
    for new_id, old in enumerate(new_order, start=1):
        x, y = coords[old]
        lines.append(f"{new_id}\t{x}\t{y}")
    lines.append("ITEMS SECTION\t(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER): ")
    for idx, (profit, weight, node) in enumerate(items, start=1):
        lines.append(f"{idx}\t{profit}\t{weight}\t{node}")

    return NL.join(lines) + NL


def _fmt_speed(v):
    if float(v).is_integer():
        return str(int(v))
    return ("%g" % v)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate TTP-D benchmark instances.")
    ap.add_argument("--n", type=int, nargs="+", default=None,
                    help="Which n values to generate (default: all).")
    cli = ap.parse_args()
    n_values = cli.n if cli.n else N_VALUES

    params, coords, node_items = parse_source(SOURCE)
    candidate_nodes = [nid for nid in coords if nid != DEPOT_ID]
    print(f"source: {SOURCE}")
    print(f"  nodes={len(coords)}  candidate(non-depot)={len(candidate_nodes)}  "
          f"W_full={params['W']}  R={params['R']}")

    os.makedirs(HERE, exist_ok=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    for n in n_values:
        consolidated = []
        for seed in SEEDS:
            chosen = select_cities(n, candidate_nodes, seed)
            name = f"bench_{n}_{seed}-TTP"
            text = build_instance_text(name, params, coords, node_items, chosen)

            indiv_path = os.path.join(OUT_DIR, f"bench_{n}_{seed}.txt")
            with open(indiv_path, "w", newline="") as fh:
                fh.write(text)
            consolidated.append(text)

        cons_path = os.path.join(OUT_DIR, f"bench_{n}.txt")
        with open(cons_path, "w", newline="") as fh:
            fh.write((NL).join(consolidated))

        print(f"  n={n:<4} -> 5 individual files + bench_{n}.txt "
              f"(W_scaled={int(round(params['W']*n/N_CITIES_FULL))})")

    print("done.")


if __name__ == "__main__":
    main()

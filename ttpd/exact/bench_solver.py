import math
import time
import os
import sys
from datetime import datetime
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def parse_instance(filepath):
    lines = open(filepath).readlines()
    params, coords = {}, {}
    node_items = defaultdict(list)
    coord_section = items_section = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("DIMENSION"):
            params["n_nodes"] = int(line.split(":")[1].strip().split()[0])
        elif line.startswith("CAPACITY OF KNAPSACK"):
            params["W"] = float(line.split(":")[1].strip())
        elif line.startswith("MIN SPEED"):
            params["v_min"] = float(line.split(":")[1].strip())
        elif line.startswith("MAX SPEED"):
            params["v_max"] = float(line.split(":")[1].strip())
        elif line.startswith("RENTING RATIO"):
            params["R"] = float(line.split(":")[1].strip())
        elif "NODE_COORD_SECTION" in line:
            coord_section, items_section = True, False
        elif "ITEMS SECTION" in line:
            items_section, coord_section = True, False
        elif coord_section:
            p = line.split()
            if len(p) == 3:
                coords[int(p[0])] = (float(p[1]), float(p[2]))
        elif items_section:
            p = line.split()
            if len(p) == 4:
                node_items[int(p[3])].append((int(p[1]), int(p[2])))

    best_item = {nid: max(items, key=lambda x: x[0])
                 for nid, items in node_items.items()}
    return params, coords, best_item


def euclidean_ceil(x1, y1, x2, y2):
    return math.ceil(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def gurobi_status_str(status):
    mapping = {
        GRB.OPTIMAL:     "Optimal",
        GRB.INFEASIBLE:  "Infeasible",
        GRB.UNBOUNDED:   "Unbounded",
        GRB.TIME_LIMIT:  "TimeLimit",
        GRB.SUBOPTIMAL:  "Suboptimal",
        GRB.INF_OR_UNBD: "InfOrUnbd",
        GRB.INTERRUPTED: "Interrupted",
    }
    return mapping.get(status, f"Status({status})")


def solve_instance(filepath, K=10, time_limit=86400,
                   drone_speed_factor=2.0, mip_gap=0.0001,
                   warmstart_sol=None, sol_out=None):
    t_start = time.perf_counter()

    params, coords, best_item = parse_instance(filepath)
    W     = params["W"]            # already scaled in the file
    v_min = params["v_min"]
    v_max = params["v_max"]
    R     = params["R"]
    dv    = v_max - v_min
    depot_id = 1

    customers = sorted(nid for nid in coords if nid != depot_id)
    n = len(customers)

    log(f"file={os.path.basename(filepath)}  n={n}  W={W:.1f}  R={R}  "
        f"v_min={v_min}  v_max={v_max}  time_limit={time_limit}s  "
        f"mip_gap={mip_gap}")

    node_ids = [depot_id] + customers + [depot_id]
    profit   = [0.0] + [float(best_item[c][0]) for c in customers] + [0.0]
    weight   = [0.0] + [float(best_item[c][1]) for c in customers] + [0.0]

    V   = list(range(n + 2))
    N   = list(range(1, n + 1))
    src, snk = 0, n + 1

    def dist(i, j):
        xi, yi = coords[node_ids[i]]
        xj, yj = coords[node_ids[j]]
        return euclidean_ceil(xi, yi, xj, yj)

    d = {(i, j): dist(i, j) for i in V for j in V if i != j}

    v_D = v_max * drone_speed_factor
    W_D = max(weight)

    W_actual = sum(weight)
    w_bp  = [k * W_actual / K for k in range(K + 1)]
    v_bp  = [v_max - w_bp[k] * dv / W for k in range(K + 1)]
    inv_v = [1.0 / max(vk, 1e-9) for vk in v_bp]

    def nn_tour(nodes_list):
        unvisited = list(nodes_list)
        cur, route, total = src, [src], 0
        while unvisited:
            nxt = min(unvisited, key=lambda j: d[cur, j])
            route.append(nxt)
            total += d[cur, nxt]
            cur = nxt
            unvisited.remove(nxt)
        route.append(snk)
        total += d[cur, snk]
        return total, route

    nn_dist, nn_route = nn_tour(N)
    T_horizon = nn_dist / v_min
    M_W       = W_actual

    m = gp.Model(f"TTPD_n{n}")
    m.Params.OutputFlag    = 1
    m.Params.TimeLimit     = float(time_limit)
    m.Params.MIPGap        = float(mip_gap)
    m.Params.Threads       = int(os.environ.get("GRB_THREADS", "0"))
    m.Params.MIPFocus      = 3
    m.Params.Heuristics    = 0.2
    m.Params.Cuts          = 2
    m.Params.Presolve      = 2
    m.Params.IntFeasTol    = 1e-7
    m.ModelSense           = GRB.MAXIMIZE

    xT = m.addVars([(i, j) for i in V for j in V if i != j],
                   vtype=GRB.BINARY, name="xT")
    xD = m.addVars([(i, j) for i in V for j in V if i != j],
                   vtype=GRB.BINARY, name="xD")
    yT = m.addVars(N, vtype=GRB.BINARY, name="yT")
    yD = m.addVars(N, vtype=GRB.BINARY, name="yD")
    yC = m.addVars(N, vtype=GRB.BINARY, name="yC")
    z  = m.addVars(N, vtype=GRB.BINARY, name="z")

    v_mc = m.addVars([(k, j) for k in N for j in V if j != k],
                     lb=0.0, ub=1.0, name="v_mc")

    W_var = m.addVars(V, lb=0.0, ub=W_actual, name="W")
    a_var = m.addVars(V, lb=0.0, name="a")

    dep_nodes = [v for v in V if v != snk]
    beta = m.addVars([(i, k) for i in dep_nodes for k in range(K + 1)],
                     lb=0.0, ub=1.0, name="beta")
    phi  = m.addVars(dep_nodes, lb=inv_v[0], ub=inv_v[-1], name="phi")

    m.update()

    m.setObjective(
        gp.quicksum(profit[i] * z[i] for i in N) - R * a_var[snk],
        GRB.MAXIMIZE,
    )

    m.addConstrs((yT[i] + yD[i] + yC[i] == 1 for i in N), name="mode")

    m.addConstr(gp.quicksum(xT[src, j] for j in V if j != src) == 1, "truck_src_out")
    m.addConstr(gp.quicksum(xT[i, snk] for i in V if i != snk) == 1, "truck_snk_in")
    m.addConstrs((xT[j, src] == 0 for j in V if j != src), "truck_no_return_src")
    m.addConstrs((xT[snk, i] == 0 for i in V if i != snk), "truck_no_leave_snk")
    m.addConstrs(
        (gp.quicksum(xT[i, j] for j in V if j != i) ==
         gp.quicksum(xT[j, i] for j in V if j != i)
         for i in N),
        name="truck_flow",
    )
    m.addConstrs(
        (gp.quicksum(xT[i, j] for j in V if j != i) == yT[i] + yC[i]
         for i in N),
        name="truck_degree",
    )

    m.addConstr(gp.quicksum(xD[src, j] for j in V if j != src) == 1, "drone_src_out")
    m.addConstr(gp.quicksum(xD[i, snk] for i in V if i != snk) == 1, "drone_snk_in")
    m.addConstrs((xD[j, src] == 0 for j in V if j != src), "drone_no_return_src")
    m.addConstrs((xD[snk, i] == 0 for i in V if i != snk), "drone_no_leave_snk")
    m.addConstrs(
        (gp.quicksum(xD[j, i] for j in V if j != i) == yD[i] + yC[i]
         for i in N),
        name="drone_in_degree",
    )
    m.addConstrs(
        (gp.quicksum(xD[i, j] for j in V if j != i) == yD[i] + yC[i]
         for i in N),
        name="drone_out_degree",
    )
    m.addConstrs(
        (xD[i, j] <= yC[i] + yC[j]
         for i in N for j in N if i != j),
        name="drone_anchor",
    )

    m.addConstrs(
        (weight[i] * z[i] <= W_D + M_W * (1 - yD[i]) for i in N),
        name="drone_payload",
    )

    m.addConstrs(
        (v_mc[k, j] >= z[k] + xD[k, j] - 1
         for k in N for j in V if j != k),
        name="mc1",
    )
    m.addConstrs(
        (v_mc[k, j] <= z[k]
         for k in N for j in V if j != k),
        name="mc2",
    )
    m.addConstrs(
        (v_mc[k, j] <= xD[k, j]
         for k in N for j in V if j != k),
        name="mc3",
    )

    m.addConstr(W_var[src] == 0, "W_src")
    for i in V:
        if i == snk:
            continue
        for j in V:
            if j == i or j == src:
                continue
            drone_wt = gp.quicksum(v_mc[k, j] * weight[k] for k in N if k != j)
            truck_wt = weight[j] * z[j] if j in N else 0.0
            rhs_expr = W_var[i] + truck_wt + drone_wt
            m.addGenConstrIndicator(
                xT[i, j], True,
                W_var[j] - rhs_expr, GRB.EQUAL, 0.0,
                name=f"W_prop_{i}_{j}",
            )

    for i in dep_nodes:
        m.addConstr(
            W_var[i] == gp.quicksum(beta[i, k] * w_bp[k] for k in range(K + 1)),
            name=f"sos2_W_{i}",
        )
        m.addConstr(
            phi[i] == gp.quicksum(beta[i, k] * inv_v[k] for k in range(K + 1)),
            name=f"sos2_phi_{i}",
        )
        m.addConstr(
            gp.quicksum(beta[i, k] for k in range(K + 1)) == 1,
            name=f"sos2_sum_{i}",
        )
        m.addSOS(GRB.SOS_TYPE2, [beta[i, k] for k in range(K + 1)])

    m.addConstr(a_var[src] == 0, "a_src")
    for i in V:
        if i == snk:
            continue
        for j in V:
            if j == i or j == src:
                continue
            m.addGenConstrIndicator(
                xT[i, j], True,
                a_var[j] - a_var[i] - d[i, j] * phi[i], GRB.GREATER_EQUAL, 0.0,
                name=f"truck_time_{i}_{j}",
            )

    M_D = T_horizon
    for i in V:
        for j in V:
            if j == i:
                continue
            m.addConstr(
                a_var[i] + d[i, j] / v_D <=
                a_var[j] + M_D * (1 - xD[i, j] + xT[i, j]),
                name=f"drone_time_{i}_{j}",
            )

    m.addConstr(
        a_var[snk] >= inv_v[0] * gp.quicksum(
            d[i, j] * xT[i, j] for i in V for j in V if i != j
        ),
        name="arrival_lb",
    )

    m.addConstr(
        a_var[snk] >= (1.0 / v_D) * gp.quicksum(
            d[i, j] * xD[i, j] for i in V for j in V if i != j
        ),
        name="arrival_lb_drone",
    )

    # Mode-aware round-trip arrival LB (validated valid inequality, see gurobi15.py).
    for i in N:
        _rt = d[src, i] + d[i, snk]
        m.addConstr(
            a_var[snk] >= _rt * ((yT[i] + yC[i]) / v_max + yD[i] / v_D),
            name=f"roundtrip_mode_{i}",
        )

    # warm start (nearest-neighbour truck tour, greedy knapsack)
    for key in xT:
        xT[key].Start = 0.0
    for key in xD:
        xD[key].Start = 0.0
    tour_arc_set = set(zip(nn_route[:-1], nn_route[1:]))
    for (i, j) in tour_arc_set:
        xT[i, j].Start = 1.0
    xD[src, snk].Start = 1.0
    for i in N:
        yT[i].Start = 1.0
        yD[i].Start = 0.0
        yC[i].Start = 0.0

    cumW = 0.0
    z_hint = {}
    for nd in nn_route[1:-1]:
        if cumW + weight[nd] <= W:
            z_hint[nd] = 1.0
            cumW += weight[nd]
        else:
            z_hint[nd] = 0.0
    for i in N:
        z[i].Start = z_hint.get(i, 0.0)

    cumW = 0.0
    W_var[src].Start = 0.0
    for nd in nn_route[1:]:
        if nd in N:
            cumW += weight[nd] * z_hint.get(nd, 0.0)
        W_var[nd].Start = cumW

    def phi_at(W_val):
        for seg in range(len(w_bp) - 1):
            if w_bp[seg] <= W_val <= w_bp[seg + 1]:
                span = w_bp[seg + 1] - w_bp[seg]
                frac = (W_val - w_bp[seg]) / span if span > 1e-12 else 0.0
                return inv_v[seg] + frac * (inv_v[seg + 1] - inv_v[seg])
        return inv_v[-1]

    a_cur = 0.0
    a_var[src].Start = 0.0
    dep_node_set = set(dep_nodes)
    for seg_idx in range(len(nn_route) - 1):
        u, v_ = nn_route[seg_idx], nn_route[seg_idx + 1]
        W_u   = W_var[u].Start
        phi_v = phi_at(W_u)
        if u in dep_node_set:
            phi[u].Start = phi_v
            for bp in range(len(w_bp) - 1):
                if w_bp[bp] <= W_u <= w_bp[bp + 1]:
                    span = w_bp[bp + 1] - w_bp[bp]
                    frac = (W_u - w_bp[bp]) / span if span > 1e-12 else 0.0
                    for kk in range(K + 1):
                        beta[u, kk].Start = 0.0
                    beta[u, bp].Start     = 1.0 - frac
                    beta[u, bp + 1].Start = frac
                    break
        a_cur += d[u, v_] * phi_v
        a_var[v_].Start = a_cur

    if warmstart_sol and os.path.exists(warmstart_sol) and \
            os.path.getsize(warmstart_sol) > 0:
        try:
            m.update()
            m.read(warmstart_sol)     # .sol -> sets .Start by var name
            log(f"  warm-started from incumbent {os.path.basename(warmstart_sol)}")
        except Exception as e:
            log(f"  [warm-start read failed, using NN start] {e}")

    def build_solution_dict(getter, obj_val, runtime, t_build, gap, node_cnt):
        zX, yTX, yDX, yCX = getter(z), getter(yT), getter(yD), getter(yC)
        xTX, xDX, aX, WX  = getter(xT), getter(xD), getter(a_var), getter(W_var)

        all_col = [i for i in N if zX[i] > 0.5]
        t_items = [i for i in N if zX[i] > 0.5 and yTX[i] > 0.5]
        d_items = [i for i in N if zX[i] > 0.5 and yDX[i] > 0.5]
        r_items = [i for i in N if zX[i] > 0.5 and yCX[i] > 0.5]
        truck_arcs = [(i, j) for (i, j) in d if xTX[i, j] > 0.5]
        drone_arcs = [(i, j) for (i, j) in d if xDX[i, j] > 0.5]

        route, visited, cur = [src], set(), src
        while True:
            nxt = next((j for (i, j) in truck_arcs
                        if i == cur and j not in visited), None)
            if nxt is None or nxt == snk:
                route.append(snk)
                break
            route.append(nxt)
            visited.add(nxt)
            cur = nxt

        total_profit = sum(profit[i] for i in all_col)
        arrival = aX[snk]
        return {
            "obj": obj_val, "gap": gap, "runtime": runtime,
            "profit": total_profit, "rental": R * arrival, "arrival": arrival,
            "all_col": all_col, "t_items": t_items, "d_items": d_items,
            "r_items": r_items, "route": route, "drone_arcs": drone_arcs,
            "W_snk": WX[snk], "W_seq": [(node_ids[nd], round(WX[nd], 1))
                                        for nd in route],
            "node_cnt": node_cnt,
        }

    best_seen = {"obj": -float("inf")}

    all_vars = m.getVars() if sol_out else []

    def _write_sol_atomic(path, names_vals, obj_val):
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(f"# Solution for model TTPD_n{n}\n")
            f.write(f"# Objective value = {obj_val:.10g}\n")
            for nm, val in names_vals:
                f.write(f"{nm} {val:.10g}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)     # atomic on POSIX

    def incumbent_cb(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            obj_bnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            if obj_val <= best_seen["obj"] + 1e-6:
                return
            best_seen["obj"] = obj_val
            gap = abs(obj_bnd - obj_val) / max(abs(obj_val), 1e-9) * 100
            log(f"  NEW BEST obj={obj_val:.2f} bound={obj_bnd:.2f} "
                f"gap={gap:.2f}% t={runtime:.0f}s")
            # Persist the incumbent so a restart-with-more-threads can resume it.
            if sol_out:
                vals = model.cbGetSolution(all_vars)
                names_vals = [(v.VarName, x) for v, x in zip(all_vars, vals)]
                _write_sol_atomic(sol_out, names_vals, obj_val)
        except Exception as e:
            log(f"[incumbent callback error, ignored] {e}")

    t_build = time.perf_counter()
    log(f"  model built in {t_build - t_start:.2f}s "
        f"({m.NumVars} vars, {m.NumConstrs} constrs, "
        f"{m.NumGenConstrs} indicators); solving...")
    m.optimize(incumbent_cb)
    t_solve = time.perf_counter()

    status = m.Status
    res = {
        "n": n, "status": gurobi_status_str(status),
        "build_s": round(t_build - t_start, 3),
        "solve_s": round(t_solve - t_build, 3),
        "total_s": round(t_solve - t_start, 3),
        "objective": None, "profit": None, "rental": None, "arrival": None,
        "n_items": None, "n_truck": None, "n_drone": None, "n_rendv": None,
        "W_final": None, "mip_gap": None, "n_nodes": None,
        "W_capacity": W, "W_actual": W_actual,
        "collected_node_ids": [], "truck_node_ids": [], "drone_node_ids": [],
        "rendezvous_node_ids": [], "truck_route_node_ids": [],
        "drone_arc_node_ids": [], "W_sequence": [],
    }

    if m.SolCount > 0:
        obj_val = m.ObjVal
        gap = m.MIPGap * 100 if hasattr(m, "MIPGap") else 0.0
        s = build_solution_dict(lambda v: {k: v[k].X for k in v},
                                obj_val, res["solve_s"], t_build, gap,
                                int(m.NodeCount))
        res.update({
            "objective": round(obj_val, 2), "profit": round(s["profit"], 2),
            "rental": round(s["rental"], 2), "arrival": round(s["arrival"], 4),
            "n_items": len(s["all_col"]), "n_truck": len(s["t_items"]),
            "n_drone": len(s["d_items"]), "n_rendv": len(s["r_items"]),
            "W_final": round(s["W_snk"], 1), "mip_gap": round(gap, 2),
            "n_nodes": int(m.NodeCount),
            "collected_node_ids":   [node_ids[i] for i in s["all_col"]],
            "truck_node_ids":       [node_ids[i] for i in s["t_items"]],
            "drone_node_ids":       [node_ids[i] for i in s["d_items"]],
            "rendezvous_node_ids":  [node_ids[i] for i in s["r_items"]],
            "truck_route_node_ids": [node_ids[nd] for nd in s["route"]],
            "drone_arc_node_ids":   [(node_ids[u], node_ids[v_])
                                     for u, v_ in s["drone_arcs"]],
            "W_sequence":           s["W_seq"],
        })

    return res

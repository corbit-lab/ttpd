V_MAX = 1.0
V_MIN = 0.1
DELTA_V = V_MAX - V_MIN
DRONE_SPEED_FACTOR = 2.0
V_DRONE_DEFAULT = V_MAX * DRONE_SPEED_FACTOR

BOX = 300                      # uniform integer coords on [0, BOX]^2, CEIL_2D
ITEMS_PER_NODE = 5             # best (max-profit) item per node is kept downstream
W_ITEM_RANGE = (1000, 1009)    # item weight ~ U{lo..hi}
P_ITEM_RANGE = (1, 1000)       # item profit ~ U{lo..hi}
R_DEFAULT = 50.0
R_LOGU_RANGE = (1.0, 300.0)    # only used with sample_R=True (off for ttd300)
CAP_PER_NODE = 637_010.0 / 280.0

ED_FRACS_BENCH = (0.25, 0.50, 0.75, 1.00)
ED_FRAC_RANGE_DEFAULT = (0.25, 1.00)


def capacity_for_n(n: int) -> float:
    return float(round(CAP_PER_NODE * n))


def scaled_capacity(W_full: float, n: int, n_pool: int) -> float:
    """Subset capacity preserving the pool file's capacity-per-node ratio.
    For a full-pool load (n == n_pool) this is the file's W verbatim."""
    return W_full * n / max(1, n_pool)


def tour_time_ub(d_max: float, n: int) -> float:
    return d_max * (n + 1) / V_MAX

# constants from the dataset 
V_MAX = 1.0
V_MIN = 0.1
DELTA_V = V_MAX - V_MIN
W_TRUCK_DEFAULT = 637_010.0
N_CITIES_FULL = 280  # a280: the capacity W_TRUCK_DEFAULT is calibrated for all 280 cities
DRONE_SPEED_FACTOR = 2.0


def scaled_capacity(W_full: float, n: int) -> float:
    return W_full * n / N_CITIES_FULL
V_DRONE_DEFAULT = V_MAX * DRONE_SPEED_FACTOR
R_DEFAULT = 72.70
R_LOGU_RANGE = (1.0, 300.0)

def tour_time_ub(d_max: float, n: int) -> float:
    return d_max * (n + 1) / V_MAX

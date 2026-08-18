#a
import numpy as np
from instance import TTPDInstance

NO_LAUNCH = -1
_EPS = 1e-9


def dummy_slot(inst: TTPDInstance) -> int:
    return inst.n + 2

#Customers the truck may still drive to / the drone may be launched at: neither already served nor currently committed to the in-flight drone 
def available_mask(V_t: np.ndarray, committed_D: int, inst: TTPDInstance) -> np.ndarray:
    avail = ~V_t.copy()
    avail[inst.source] = False
    avail[inst.sink] = False
    if committed_D is not None and committed_D >= 0:
        avail[committed_D] = False
    return avail

# stay_in_flight, land_now. At the sink an in-flight drone must land (terminal)
def mask_rejoin(in_flight: bool, c_t: int, inst: TTPDInstance) -> np.ndarray:
    if not in_flight:
        return np.array([True, False], dtype=bool)
    mask = np.array([True, True], dtype=bool)
    if c_t == inst.sink:
        mask[0] = False
    return mask

# Collect the drone-delivered item? checked at rejoin against the live truck load, since the item's weight transfers to the truck when the drone lands
def mask_z_drone(W_after_truck_pickup: float, drone_D: int, inst: TTPDInstance) -> np.ndarray:
    mask = np.array([True, False], dtype=bool)
    if drone_D is None or drone_D < 0 or drone_D == inst.source or drone_D == inst.sink:
        return mask
    if W_after_truck_pickup + inst.weights[drone_D] <= inst.W + _EPS:
        mask[1] = True
    return mask

# Collect the item at the current node (truck pickup), capacity permitting
def mask_z_curr(c_t: int, W_t: float, inst: TTPDInstance) -> np.ndarray:
    mask = np.array([True, False], dtype=bool)
    if c_t == inst.source or c_t == inst.sink:
        return mask
    if W_t + inst.weights[c_t] <= inst.W + _EPS:
        mask[1] = True
    return mask


def mask_launch_k(
    c_t: int,
    available: np.ndarray,
    in_flight_after_rejoin: bool,
    inst: TTPDInstance,
) -> np.ndarray:
    n_nodes = inst.n + 2
    n_slots = n_nodes + 1
    mask = np.zeros(n_slots, dtype=bool)
    mask[dummy_slot(inst)] = True
    if in_flight_after_rejoin:
        return mask
    mask[:n_nodes] = available
    return mask

# Truck destinations: any unserved customer; the sink only once none remain
def mask_j(
    c_t: int,
    available: np.ndarray,
    in_flight: bool,
    inst: TTPDInstance,
) -> np.ndarray:
    mask = available.copy()
    if int(available.sum()) == 0:
        mask[inst.sink] = True
    return mask

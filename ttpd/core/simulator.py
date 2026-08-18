#a
from typing import Any
import numpy as np
import masking
from constants import tour_time_ub
from instance import TTPDInstance, load_a280, sample_instance
from masking import NO_LAUNCH


class TTPDEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        a280_path: str = "a280_benchmark.txt",
        n: int = 10,
        sample_R: bool = False,
        fixed_R: float | None = None,
        scale_capacity: bool = True,
        a280_data: dict | None = None,
    ):
        # a280_data lets batched rollouts spawn many envs without re-parsing the file
        self._a280 = a280_data if a280_data is not None else load_a280(a280_path)
        self._n = n
        self._sample_R = sample_R
        self._fixed_R = fixed_R
        self._scale_capacity = scale_capacity

        self._rng: np.random.Generator | None = None
        self.inst: TTPDInstance | None = None

        self.c_t: int = 0
        self.tau_t: float = 0.0
        self.W_t: float = 0.0
        self.V_t: np.ndarray = np.zeros(0, dtype=bool)

        # drone state
        self.in_flight: bool = False
        self.drone_L: int = -1        
        self.drone_D: int = -1         
        self.drone_t_launch: float = 0.0  

        self.t: int = 0
        self._collected_profit: float = 0.0

    # reset
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()

        options = options or {}
        if "instance" in options:
            self.inst = options["instance"]
        else:
            self.inst = sample_instance(
                self._a280,
                n=options.get("n", self._n),
                rng=self._rng,
                R=self._fixed_R,
                sample_R=self._sample_R,
                scale_capacity=self._scale_capacity,
            )

        self.c_t = self.inst.source
        self.tau_t = 0.0
        self.W_t = 0.0
        self.V_t = np.zeros(self.inst.n + 2, dtype=bool)

        self.in_flight = False
        self.drone_L = -1
        self.drone_D = -1
        self.drone_t_launch = 0.0

        self.t = 0
        self._collected_profit = 0.0

        return self._obs(), self._info(extra={})

    # views
    def _available(self) -> np.ndarray:
        # customers neither served nor committed to the in-flight drone
        committed = self.drone_D if self.in_flight else -1
        return masking.available_mask(self.V_t, committed, self.inst)

    def _velocity(self, W_prime: float) -> float:
        inst = self.inst
        denom = inst.v_max - W_prime * (inst.v_max - inst.v_min) / inst.W
        if denom < inst.v_min - 1e-9:
            raise RuntimeError(
                f"Truck velocity {denom} < v_min={inst.v_min}; capacity-mask bug."
            )
        return denom

    # step
    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        if self.inst is None:
            raise RuntimeError("Call reset() before step().")
        inst = self.inst
        tau_start = self.tau_t
        profit_gain = 0.0

        j = int(action.get("j", inst.sink))
        z_curr = int(action.get("z_curr", 0))
        rejoin = int(action.get("rejoin", 0))
        z_drone = int(action.get("z_drone", 0))
        k = int(action.get("k", NO_LAUNCH))

        # this step only lands the drone there and ends the episode.
        if self.c_t == inst.sink:
            if self.in_flight:
                m_r = masking.mask_rejoin(self.in_flight, self.c_t, inst)
                if not m_r[1]:
                    raise ValueError("Drone must land at sink but rejoin is masked off.")
                profit_gain += self._land_drone(z_drone)
            r = profit_gain - inst.R * (self.tau_t - tau_start)
            self.t += 1
            terminated = bool(self.V_t[1 : inst.n + 1].all()) and not self.in_flight
            truncated = not terminated
            return self._obs(), float(r), terminated, truncated, self._info(extra={"j": inst.sink})

        # 1. rejoin (land the drone at c_t)
        m_r = masking.mask_rejoin(self.in_flight, self.c_t, inst)
        if not m_r[rejoin]:
            raise ValueError(f"Infeasible rejoin={rejoin} (mask_rejoin).")
        if self.in_flight and rejoin == 1:
            profit_gain += self._land_drone(z_drone)

        # 2. truck pickup at c_t
        m_zc = masking.mask_z_curr(self.c_t, self.W_t, inst)
        if not m_zc[z_curr]:
            raise ValueError(f"Infeasible z_curr={z_curr} (mask_z_curr).")
        W_prime = self.W_t + inst.weights[self.c_t] * z_curr
        if z_curr == 1:
            self._collect(self.c_t)
            profit_gain += inst.profits[self.c_t]

        #  3. launch the drone (idle only)
        m_k = masking.mask_launch_k(self.c_t, self._available(), self.in_flight, inst)
        dummy = masking.dummy_slot(inst)
        k_slot = dummy if k == NO_LAUNCH else k
        if k_slot < 0 or k_slot >= m_k.shape[0] or not m_k[k_slot]:
            raise ValueError(f"Infeasible launch k={k} (mask_launch_k).")
        if k != NO_LAUNCH:
            self.in_flight = True
            self.drone_L = self.c_t
            self.drone_D = k
            self.drone_t_launch = self.tau_t   # launches at the (post-rejoin) current time

        #  4. truck move c_t -> j 
        m_j = masking.mask_j(self.c_t, self._available(), self.in_flight, inst)
        if not m_j[j]:
            raise ValueError(f"Infeasible j={j} (mask_j).")
        dtau_T = inst.dist[self.c_t, j] / self._velocity(W_prime)
        self.tau_t = self.tau_t + dtau_T
        self.W_t = W_prime
        if j != inst.source and j != inst.sink:
            self.V_t[j] = True              # truck serves j
        self.c_t = j
        self.t += 1

        r = profit_gain - inst.R * (self.tau_t - tau_start)

        all_served = bool(self.V_t[1 : inst.n + 1].all()) if inst.n > 0 else True
        terminated = (self.c_t == inst.sink) and all_served and not self.in_flight
        # If we drove to the sink with the drone still out, we do NOT terminate yet; the
        # next step lands the drone at the sink (handled by the terminal block above).
        truncated = (self.t >= 2 * (inst.n + 2)) and not terminated

        return self._obs(), float(r), bool(terminated), bool(truncated), self._info(
            extra={"j": j, "z_curr": z_curr, "rejoin": rejoin, "k": k, "z_drone": z_drone,
                   "dtau_T": float(dtau_T)}
        )

    # drone landing: rendezvous at c_t, optional collection, weight transfer to truck
    def _land_drone(self, z_drone: int) -> float:
        inst = self.inst
        D = self.drone_D
        drone_arr = self.drone_t_launch + (
            inst.dist[self.drone_L, D] + inst.dist[D, self.c_t]
        ) / inst.v_D
        self.tau_t = max(self.tau_t, drone_arr)      # the early arriver waits
        m_zd = masking.mask_z_drone(self.W_t, D, inst)
        if not m_zd[z_drone]:
            raise ValueError(f"Infeasible z_drone={z_drone} at rejoin (capacity).")
        profit_gain = 0.0
        if z_drone == 1:
            self.W_t = self.W_t + inst.weights[D]    # delivered item now rides the truck
            self._collect(D)
            profit_gain = inst.profits[D]
        self.V_t[D] = True                            # D is served (drone delivered)
        self.in_flight = False
        self.drone_L = -1
        self.drone_D = -1
        return profit_gain

    def _collect(self, node: int) -> None:
        self._collected_profit += self.inst.profits[node]

    # observation
    def _obs(self) -> dict:
        inst = self.inst
        p_total = float(inst.profits.sum())

        tau_ub = tour_time_ub(inst.d_max, inst.n)
        avail = self._available()
        n_avail = int(avail.sum())

        # Drone timing keeps the in-flight state Markovian: drone_wait is the
        # rental-incurring wait if the truck lands the drone at c_t now, drone_elapsed
        # is how long the drone has been out.
        if self.in_flight:
            drone_arr = self.drone_t_launch + (
                inst.dist[self.drone_L, self.drone_D] + inst.dist[self.drone_D, self.c_t]
            ) / inst.v_D
            drone_wait = max(0.0, drone_arr - self.tau_t)
            drone_elapsed = max(0.0, self.tau_t - self.drone_t_launch)
        else:
            drone_wait = 0.0
            drone_elapsed = 0.0

        return {
            "c_t": int(self.c_t),
            "tau_t": float(self.tau_t),
            "W_t": float(self.W_t),
            "U_mask": avail,
            "n": int(inst.n),
            "in_flight": bool(self.in_flight),
            "drone_D": int(self.drone_D),
            "W_norm": float(self.W_t / inst.W),
            "tau_norm": float(self.tau_t / tau_ub) if tau_ub > 0 else 0.0,
            "U_frac": float(n_avail / inst.n) if inst.n > 0 else 0.0,
            "drone_norm": 1.0 if self.in_flight else 0.0,
            "R_norm": float(self._collected_profit / p_total) if p_total > 0 else 0.0,
            "drone_wait_norm": float(drone_wait / tau_ub) if tau_ub > 0 else 0.0,
            "drone_elapsed_norm": float(drone_elapsed / tau_ub) if tau_ub > 0 else 0.0,
        }

    def _info(self, extra: dict) -> dict:
        info = {"instance": self.inst, "action_masks": self.current_masks()}
        info.update(extra)
        return info

    # progressive masks for the policy (decisions in order: rejoin -> z_curr -> launch k -> j)
    def current_masks(self) -> dict:
        inst = self.inst
        avail = self._available()
        return {
            "rejoin": masking.mask_rejoin(self.in_flight, self.c_t, inst),
            "z_curr": masking.mask_z_curr(self.c_t, self.W_t, inst),
            # pre-launch view of the truck destinations (valid when no launch is chosen);
            # the policy uses mask_j_after_launch once k is decided.
            "j": masking.mask_j(self.c_t, avail, self.in_flight, inst),
        }

    def mask_z_drone_now(self) -> np.ndarray:
        # z_drone feasibility for landing the current in-flight drone at c_t (live load)
        return masking.mask_z_drone(self.W_t, self.drone_D, self.inst)

    def masks_for_launch(self, rejoined: bool) -> dict:
        inst = self.inst
        in_flight_after = self.in_flight and not rejoined
        return {
            "k_ext": masking.mask_launch_k(self.c_t, self._available(), in_flight_after, inst),
            "dummy_slot": int(masking.dummy_slot(inst)),
        }

    def mask_j_after_launch(self, k: int, rejoined: bool) -> np.ndarray:
        inst = self.inst
        in_flight_after = (self.in_flight and not rejoined) or k != NO_LAUNCH
        avail = masking.available_mask(self.V_t, self.drone_D if self.in_flight else -1, inst)
        if k != NO_LAUNCH:
            avail[k] = False
        return masking.mask_j(self.c_t, avail, in_flight_after, inst)

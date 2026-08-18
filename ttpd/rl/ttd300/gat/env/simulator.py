#a
from typing import Any
import numpy as np
from env import masking
from env.constants import tour_time_ub
from env.instance import TTPDInstance, load_ttd300, sample_instance
from env.masking import NO_LAUNCH


class TTPDEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        a280_path: str | None = None,
        n: int = 10,
        sample_R: bool = False,
        fixed_R: float | None = None,
        scale_capacity: bool = True,
        a280_data: dict | None = None,
        ED: float | None = None,
        ED_frac: float | None = None,
        ED_frac_range: tuple[float, float] | None = None,
        ED_frac_power: float = 1.0,
    ):
        self._a280 = a280_data if a280_data is not None else (
            load_ttd300(a280_path) if a280_path else None)
        self._n = n
        self._sample_R = sample_R
        self._fixed_R = fixed_R
        self._scale_capacity = scale_capacity
        self._ED = ED
        self._ED_frac = ED_frac
        self._ED_frac_range = ED_frac_range
        self._ED_frac_power = float(ED_frac_power)

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
            # a caller-supplied instance carries its own ED (respected as-is)
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
            self.inst.ED = self._resolve_ED(self.inst)

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

    def _resolve_ED(self, inst) -> float | None:
        # Per-episode endurance budget for a freshly sampled instance.
        if self._ED_frac_range is not None:
            lo, hi = self._ED_frac_range
            u = float(self._rng.uniform(0.0, 1.0)) ** self._ED_frac_power
            return (lo + (hi - lo) * u) * inst.d_max
        if self._ED_frac is not None:
            return float(self._ED_frac) * inst.d_max
        return self._ED

    # views
    def _available(self) -> np.ndarray:
        # customers neither served nor committed to the in-flight drone
        committed = self.drone_D if self.in_flight else -1
        return masking.available_mask(self.V_t, committed, self.inst)

    def _endurance_future_feasible(self) -> bool:
        inst = self.inst
        if inst.ED is None:
            return True
        base = inst.dist[self.drone_L, self.drone_D]
        ED = inst.ED + masking._EPS
        if base + inst.dist[self.drone_D, inst.sink] <= ED:
            return True
        idx = np.nonzero(self._available())[0]
        return bool(idx.size and np.any(base + inst.dist[self.drone_D, idx] <= ED))

    def _rejoin_mask(self) -> np.ndarray:
        inst = self.inst
        base = masking.mask_rejoin(self.in_flight, self.c_t, inst)
        if not self.in_flight or inst.ED is None:
            return base
        reach_here = inst.dist[self.drone_L, self.drone_D] + inst.dist[self.drone_D, self.c_t]
        can_land = reach_here <= inst.ED + masking._EPS
        can_stay = self._endurance_future_feasible()
        mask = base.copy()
        mask[1] = base[1] and can_land
        mask[0] = base[0] and can_stay
        if not mask[0] and not mask[1]:
            mask[1] = True   # forced landing (guaranteed feasible by the launch mask)
        return mask

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
                m_r = self._rejoin_mask()
                if not m_r[1]:
                    raise ValueError("Drone must land at sink but rejoin is masked off.")
                profit_gain += self._land_drone(z_drone)
            r = profit_gain - inst.R * (self.tau_t - tau_start)
            self.t += 1
            terminated = bool(self.V_t[1 : inst.n + 1].all()) and not self.in_flight
            truncated = not terminated
            return self._obs(), float(r), terminated, truncated, self._info(extra={"j": inst.sink})

        # 1. rejoin (land the drone at c_t)
        m_r = self._rejoin_mask()
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

        if self.in_flight:
            drone_arr = self.drone_t_launch + (
                inst.dist[self.drone_L, self.drone_D] + inst.dist[self.drone_D, self.c_t]
            ) / inst.v_D
            drone_wait = max(0.0, drone_arr - self.tau_t)
            drone_elapsed = max(0.0, self.tau_t - self.drone_t_launch)
        else:
            drone_wait = 0.0
            drone_elapsed = 0.0

        d_ref = 2.0 * inst.d_max if inst.d_max > 0 else 1.0
        if inst.ED is None:
            ed_norm = 1.0
            endurance_slack_norm = 1.0
        else:
            ed_norm = min(float(inst.ED / d_ref), 1.0)
            if self.in_flight:
                reach = inst.dist[self.drone_L, self.drone_D] + inst.dist[self.drone_D, self.c_t]
                endurance_slack_norm = float(np.clip((inst.ED - reach) / d_ref, -1.0, 1.0))
            else:
                endurance_slack_norm = ed_norm

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
            "ED_norm": ed_norm,
            "endurance_slack_norm": endurance_slack_norm,
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
            "rejoin": self._rejoin_mask(),
            "z_curr": masking.mask_z_curr(self.c_t, self.W_t, inst),
            # pre-launch view of the truck destinations (valid when no launch is chosen);
            # the policy uses mask_j_after_launch once k is decided.
            "j": masking.mask_j(self.c_t, avail, self.in_flight, inst),
        }

    def mask_z_drone_now(self) -> np.ndarray:
        # z_drone feasibility for landing the current in-flight drone at c_t (live load)
        return masking.mask_z_drone(self.W_t, self.drone_D, self.inst)

    def mask_z_curr_after(self, rejoin: int, z_drone: int) -> np.ndarray:
        inst = self.inst
        W_eff = self.W_t
        if self.in_flight and rejoin == 1 and z_drone == 1 and self.drone_D >= 0:
            W_eff = W_eff + inst.weights[self.drone_D]
        return masking.mask_z_curr(self.c_t, W_eff, inst)

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

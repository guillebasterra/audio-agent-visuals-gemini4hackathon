"""30 Hz crowd physics simulation (plan.md section 5)."""

from __future__ import annotations

import math

import numpy as np

from server.config import (
    AGENT_RADIUS,
    COLLISION_ITERATIONS,
    CONTAGION_KAPPA,
    CONTAGION_RADIUS_MULT,
    DENSITY_CAP,
    DENSITY_RADIUS_MULT,
    DJ_POS,
    ENERGY_A,
    ENERGY_B,
    ENERGY_C,
    ENERGY_D,
    EXHAUST_RECOVER_E,
    HAPPINESS_TAU,
    MASS_BASE,
    MASS_PUSH_SCALE,
    MAX_DIST_TO_DJ,
    MAX_SPEED,
    PHYSICS_HZ,
)
from server.state import Agent, CrowdState, X_MAX, X_MIN, Y_MAX, Y_MIN

GRID_SIZE: int = 10
PUSH_STEP: float = 0.08
YIELD_STEP: float = 0.06
REST_SEARCH_RADIUS: float = 0.25
RETREAT_Y: float = 0.95

# Precompute 10x10 density grid cell centers for fast rest/yield lookups
_CELL_COORDS = (np.arange(GRID_SIZE, dtype=np.float64) + 0.5) / float(GRID_SIZE)
_CELL_GX, _CELL_GY = np.meshgrid(_CELL_COORDS, _CELL_COORDS, indexing="ij")
_CELL_CX_FLAT = _CELL_GX.ravel()
_CELL_CY_FLAT = _CELL_GY.ravel()


def _clamp(val: float, lo: float, hi: float) -> float:
    return lo if val < lo else (hi if val > hi else val)


def _cell_index(coord: float) -> int:
    idx = int(coord * GRID_SIZE)
    if idx < 0:
        return 0
    if idx >= GRID_SIZE:
        return GRID_SIZE - 1
    return idx


def _build_density_grid(agents: list[Agent]) -> np.ndarray:
    """Count agents in each cell of a 10x10 grid over [0, 1]^2."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    for a in agents:
        grid[_cell_index(a.x), _cell_index(a.y)] += 1
    return grid


def _compute_agent_target(agent: Agent, grid: np.ndarray, grid_flat: np.ndarray) -> tuple[float, float]:
    """Compute target (tx, ty) for a single agent given its current action and 10x10 density grid."""
    x, y = agent.x, agent.y
    action = agent.action

    if action == "hold" or action == "dance":
        return (x, y)

    if action == "retreat":
        return (_clamp(x, X_MIN, X_MAX), _clamp(RETREAT_Y, Y_MIN, Y_MAX))

    if action == "push":
        dx = DJ_POS[0] - x
        dy = DJ_POS[1] - y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 0.0, -1.0
        return (
            _clamp(x + PUSH_STEP * ux, X_MIN, X_MAX),
            _clamp(y + PUSH_STEP * uy, Y_MIN, Y_MAX),
        )

    if action == "yield":
        dx = DJ_POS[0] - x
        dy = DJ_POS[1] - y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 0.0, -1.0
        # Two perpendicular candidates
        c1x = _clamp(x + YIELD_STEP * (-uy), X_MIN, X_MAX)
        c1y = _clamp(y + YIELD_STEP * ux, Y_MIN, Y_MAX)
        c2x = _clamp(x + YIELD_STEP * uy, X_MIN, X_MAX)
        c2y = _clamp(y + YIELD_STEP * (-ux), Y_MIN, Y_MAX)

        d1 = int(grid[_cell_index(c1x), _cell_index(c1y)])
        d2 = int(grid[_cell_index(c2x), _cell_index(c2y)])
        if d1 < d2:
            return (c1x, c1y)
        if d2 < d1:
            return (c2x, c2y)

        # Tie-break: avoid walls (prefer larger actual displacement) and open a lane away from center
        disp1 = math.hypot(c1x - x, c1y - y)
        disp2 = math.hypot(c2x - x, c2y - y)
        if abs(disp1 - disp2) > 1e-3:
            return (c1x, c1y) if disp1 > disp2 else (c2x, c2y)
        if abs(c1x - DJ_POS[0]) >= abs(c2x - DJ_POS[0]):
            return (c1x, c1y)
        return (c2x, c2y)

    if action == "rest":
        cdx = _CELL_CX_FLAT - x
        cdy = _CELL_CY_FLAT - y
        dists = np.hypot(cdx, cdy)
        within = np.nonzero(dists <= REST_SEARCH_RADIUS)[0]
        if within.size == 0:
            return (_clamp(x, X_MIN, X_MAX), _clamp(y, Y_MIN, Y_MAX))

        counts = grid_flat[within]
        min_count = int(counts.min())
        best_candidates = within[counts == min_count]
        if best_candidates.size == 1:
            idx = int(best_candidates[0])
        else:
            # Among equally empty/low-density cells, prefer nearby cells slightly away from the DJ
            scores = dists[best_candidates] - 0.5 * _CELL_CY_FLAT[best_candidates]
            idx = int(best_candidates[int(np.argmin(scores))])
        return (
            _clamp(float(_CELL_CX_FLAT[idx]), X_MIN, X_MAX),
            _clamp(float(_CELL_CY_FLAT[idx]), Y_MIN, Y_MAX),
        )

    # Fallback for any unrecognized action
    return (x, y)


class PhysicsEngine:
    """Vectorized 30 Hz crowd physics engine."""

    def __init__(
        self,
        target_refresh_seconds: float = 0.5,
        steer_tau: float = 0.15,
        arrive_tau: float = 0.20,
        collision_relax: float = 0.85,
    ) -> None:
        self.target_refresh_seconds = target_refresh_seconds
        self.steer_tau = steer_tau
        self.arrive_tau = arrive_tau
        self.collision_relax = collision_relax

        self._target_timer: float = target_refresh_seconds  # force initial compute on first step
        self._prev_actions: dict[int, str] = {}
        self._triu_n: int = -1
        self._triu_mask: np.ndarray = np.empty((0, 0), dtype=bool)

    def reset(self) -> None:
        """Reset internal timers and cached action history."""
        self._target_timer = self.target_refresh_seconds
        self._prev_actions.clear()

    def _get_triu_mask(self, n: int) -> np.ndarray:
        if self._triu_n != n:
            self._triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            self._triu_n = n
        return self._triu_mask

    def recompute_targets(self, crowd: CrowdState, force: bool = True) -> None:
        """Recompute targets for all agents (when force=True) or those whose action changed."""
        agents = crowd.agents
        if not agents:
            return

        if force:
            need_indices = list(range(len(agents)))
        else:
            need_indices = [
                i for i, a in enumerate(agents) if self._prev_actions.get(a.id) != a.action
            ]

        if not need_indices:
            return

        grid = _build_density_grid(agents)
        grid_flat = grid.ravel()
        for i in need_indices:
            a = agents[i]
            a.target = _compute_agent_target(a, grid, grid_flat)
            self._prev_actions[a.id] = a.action

    def step(self, crowd: CrowdState, dt: float | None = None, advance_time: bool = True) -> None:
        """Advance crowd physics by one frame (default dt = 1 / PHYSICS_HZ)."""
        if dt is None:
            dt = 1.0 / float(PHYSICS_HZ)

        if advance_time:
            crowd.t += dt

        agents = crowd.agents
        n = len(agents)
        if n == 0:
            return

        # Enforce exhaustion invariant before target computation
        for a in agents:
            if a.exhausted:
                a.action = "retreat"

        # 1. Target by action (recompute when action changes or every 0.5 s)
        self._target_timer += dt
        if self._target_timer >= self.target_refresh_seconds:
            self._target_timer = 0.0
            self.recompute_targets(crowd, force=True)
        else:
            self.recompute_targets(crowd, force=False)

        # Pack state into contiguous numpy arrays for fast vectorized physics
        pos = np.empty((n, 2), dtype=np.float64)
        vel = np.empty((n, 2), dtype=np.float64)
        target = np.empty((n, 2), dtype=np.float64)
        E = np.empty(n, dtype=np.float64)
        pushiness = np.empty(n, dtype=np.float64)
        k_arr = np.empty(n, dtype=np.float64)
        s_arr = np.empty(n, dtype=np.float64)
        H = np.empty(n, dtype=np.float64)
        is_push = np.zeros(n, dtype=np.float64)
        is_dance = np.zeros(n, dtype=np.float64)
        is_rest_or_retreat = np.zeros(n, dtype=bool)
        is_hold = np.zeros(n, dtype=bool)

        for i, a in enumerate(agents):
            pos[i, 0] = a.x
            pos[i, 1] = a.y
            vel[i, 0] = a.vx
            vel[i, 1] = a.vy
            target[i, 0] = a.target[0]
            target[i, 1] = a.target[1]
            E[i] = a.E
            pushiness[i] = a.pushiness
            k_arr[i] = a.k
            s_arr[i] = a.s
            H[i] = a.H
            act = a.action
            if act == "push":
                is_push[i] = 1.0
            elif act == "dance":
                is_dance[i] = 1.0
            elif act == "rest" or act == "retreat":
                is_rest_or_retreat[i] = True
            elif act == "hold":
                is_hold[i] = True

        # 2. Integrate: steer velocity smoothly toward target, clamped to max_speed
        max_speed = MAX_SPEED * (0.4 + 0.6 * E)
        to_target = target - pos
        dist_target = np.hypot(to_target[:, 0], to_target[:, 1])

        v_des = np.zeros((n, 2), dtype=np.float64)
        moving = dist_target > 1e-6
        if np.any(moving):
            desired_speed = np.minimum(max_speed[moving], dist_target[moving] / self.arrive_tau)
            scale = desired_speed / dist_target[moving]
            v_des[moving, 0] = to_target[moving, 0] * scale
            v_des[moving, 1] = to_target[moving, 1] * scale

        steer_alpha = 1.0 - math.exp(-dt / self.steer_tau)
        vel += steer_alpha * (v_des - vel)

        speed = np.hypot(vel[:, 0], vel[:, 1])
        too_fast = speed > max_speed
        if np.any(too_fast):
            clamp_scale = max_speed[too_fast] / speed[too_fast]
            vel[too_fast, 0] *= clamp_scale
            vel[too_fast, 1] *= clamp_scale

        pos += vel * dt

        # 3. Collide: 4 iterations, brute-force all pairs, inverse-mass split, wall clamp
        pressure = np.zeros(n, dtype=np.float64)
        mass = MASS_BASE + MASS_PUSH_SCALE * (pushiness * pushiness) * E
        inv_mass = 1.0 / mass
        min_dist = 2.0 * AGENT_RADIUS
        min_dist_sq = min_dist * min_dist
        triu_mask = self._get_triu_mask(n)
        relax = self.collision_relax

        for iter_idx in range(COLLISION_ITERATIONS):
            dx = pos[:, 0:1] - pos[None, :, 0]
            dy = pos[:, 1:2] - pos[None, :, 1]
            dist_sq = dx * dx + dy * dy

            ii, jj = np.nonzero((dist_sq < min_dist_sq) & triu_mask)
            if ii.size > 0:
                pdx = dx[ii, jj]
                pdy = dy[ii, jj]
                pdist = np.sqrt(dist_sq[ii, jj])
                overlap = min_dist - pdist

                zero = pdist < 1e-8
                if np.any(zero):
                    nx = np.empty_like(pdist)
                    ny = np.empty_like(pdist)
                    nz = ~zero
                    nx[nz] = pdx[nz] / pdist[nz]
                    ny[nz] = pdy[nz] / pdist[nz]
                    angles = ((ii[zero] * 31 + jj[zero] * 17) % 360) * (math.pi / 180.0)
                    nx[zero] = np.cos(angles)
                    ny[zero] = np.sin(angles)
                else:
                    nx = pdx / pdist
                    ny = pdy / pdist

                im_i = inv_mass[ii]
                im_j = inv_mass[jj]
                inv_sum = im_i + im_j
                sep = overlap * relax
                move_i = sep * (im_i / inv_sum)
                move_j = sep * (im_j / inv_sum)

                np.add.at(pos[:, 0], ii, move_i * nx)
                np.add.at(pos[:, 1], ii, move_i * ny)
                np.add.at(pos[:, 0], jj, -move_j * nx)
                np.add.at(pos[:, 1], jj, -move_j * ny)

                # Shoulder-slip & jostle on first iteration only so remaining iterations
                # resolve all overlaps cleanly (< 0.001) while heavy pushers still slip past.
                if iter_idx == 0:
                    push_pair = np.maximum(is_push[ii], is_push[jj])
                    if np.any(push_pair > 0.0):
                        tx = -ny
                        ty = nx
                        jostle_sign = np.sin(crowd.t * 3.1 + ii * 1.7 - jj * 2.3)
                        side = np.where(
                            np.abs(pdx) > 1e-4,
                            np.sign(pdx),
                            np.where(jostle_sign >= 0.0, 1.0, -1.0),
                        )
                        mass_diff = np.abs(mass[ii] - mass[jj]) / (mass[ii] + mass[jj])
                        slip = sep * push_pair * (0.72 * mass_diff * side + 0.38 * jostle_sign)
                        np.add.at(pos[:, 0], ii, slip * tx * (im_i / inv_sum))
                        np.add.at(pos[:, 1], ii, slip * ty * (im_i / inv_sum))
                        np.add.at(pos[:, 0], jj, -slip * tx * (im_j / inv_sum))
                        np.add.at(pos[:, 1], jj, -slip * ty * (im_j / inv_sum))

                        i_hb = (mass[ii] > mass[jj] * 1.35) & (is_push[ii] > 0.5) & (pos[ii, 1] > pos[jj, 1])
                        j_hb = (mass[jj] > mass[ii] * 1.35) & (is_push[jj] > 0.5) & (pos[jj, 1] > pos[ii, 1])
                        if np.any(i_hb):
                            k_idx = np.nonzero(i_hb)[0]
                            s_y = sep[k_idx] * 0.45 * mass_diff[k_idx]
                            np.add.at(pos[:, 1], ii[k_idx], -s_y)
                            np.add.at(pos[:, 1], jj[k_idx], +s_y)
                        if np.any(j_hb):
                            k_idx = np.nonzero(j_hb)[0]
                            s_y = sep[k_idx] * 0.45 * mass_diff[k_idx]
                            np.add.at(pos[:, 1], jj[k_idx], -s_y)
                            np.add.at(pos[:, 1], ii[k_idx], +s_y)

                np.add.at(pressure, ii, overlap)
                np.add.at(pressure, jj, overlap)

            # Clamp to floor walls (DJ booth y < DJ_BOOTH_Y is a wall)
            pen_x_lo = np.maximum(0.0, X_MIN - pos[:, 0])
            pen_x_hi = np.maximum(0.0, pos[:, 0] - X_MAX)
            pen_y_lo = np.maximum(0.0, Y_MIN - pos[:, 1])
            pen_y_hi = np.maximum(0.0, pos[:, 1] - Y_MAX)
            pressure += pen_x_lo + pen_x_hi + pen_y_lo + pen_y_hi

            hit_x_lo = pos[:, 0] < X_MIN
            hit_x_hi = pos[:, 0] > X_MAX
            hit_y_lo = pos[:, 1] < Y_MIN
            hit_y_hi = pos[:, 1] > Y_MAX

            pos[:, 0] = np.clip(pos[:, 0], X_MIN, X_MAX)
            pos[:, 1] = np.clip(pos[:, 1], Y_MIN, Y_MAX)

            if np.any(hit_x_lo):
                vel[hit_x_lo, 0] = np.maximum(vel[hit_x_lo, 0], 0.0)
            if np.any(hit_x_hi):
                vel[hit_x_hi, 0] = np.minimum(vel[hit_x_hi, 0], 0.0)
            if np.any(hit_y_lo):
                vel[hit_y_lo, 1] = np.maximum(vel[hit_y_lo, 1], 0.0)
            if np.any(hit_y_hi):
                vel[hit_y_hi, 1] = np.minimum(vel[hit_y_hi, 1], 0.0)

        # 4. Derive rho and v (compute post-collision pairwise squared distances once)
        dx = pos[:, 0:1] - pos[None, :, 0]
        dy = pos[:, 1:2] - pos[None, :, 1]
        dist_sq = dx * dx + dy * dy

        density_r_sq = (DENSITY_RADIUS_MULT * AGENT_RADIUS) ** 2
        rho_neighbors = dist_sq <= density_r_sq
        np.fill_diagonal(rho_neighbors, False)
        rho = np.minimum(rho_neighbors.sum(axis=1), DENSITY_CAP) / float(DENSITY_CAP)

        dist_dj = np.hypot(pos[:, 0] - DJ_POS[0], pos[:, 1] - DJ_POS[1])
        v = np.clip(1.0 - dist_dj / MAX_DIST_TO_DJ, 0.0, 1.0)

        # 5. Energy & exhaustion
        recover_mask = (is_rest_or_retreat | (is_hold & (rho < 0.2))).astype(np.float64)
        dE_dt = (
            -ENERGY_A * v * rho
            - ENERGY_B * is_push
            - ENERGY_D * is_dance
            + ENERGY_C * (1.0 - v) * (1.0 - rho) * recover_mask
        )
        E = np.clip(E + dE_dt * dt, 0.0, 1.0)

        # 6. Happiness + synchronous neighbor contagion
        if crowd.use_dsp_proxy:
            crowd.w = 0.5 + 2.5 * crowd.energy_5s

        w = float(crowd.w)
        music_energy = float(crowd.music_energy)
        denom = np.maximum(k_arr + s_arr, 1e-6)
        u = E * (k_arr * v * w + s_arr * (1.0 - rho) * music_energy) / denom
        u_norm = np.clip(u / 3.0, 0.0, 1.0)

        ema_alpha = 1.0 - math.exp(-dt / HAPPINESS_TAU)
        H = H + ema_alpha * (u_norm - H)

        contagion_r_sq = (CONTAGION_RADIUS_MULT * AGENT_RADIUS) ** 2
        cont_neighbors = dist_sq <= contagion_r_sq
        np.fill_diagonal(cont_neighbors, False)
        cont_counts = cont_neighbors.sum(axis=1)
        has_cont = cont_counts > 0
        if np.any(has_cont):
            # Synchronous neighbor mean computed from pre-contagion H
            neighbor_h_sum = cont_neighbors[has_cont].astype(np.float64) @ H
            mean_h_neighbors = neighbor_h_sum / cont_counts[has_cont]
            H[has_cont] = H[has_cont] + CONTAGION_KAPPA * (mean_h_neighbors - H[has_cont]) * dt

        H = np.clip(H, 0.0, 1.0)

        # Write back to Agent objects and apply exhaustion transitions
        for i, a in enumerate(agents):
            a.x = float(pos[i, 0])
            a.y = float(pos[i, 1])
            a.vx = float(vel[i, 0])
            a.vy = float(vel[i, 1])
            a.E = float(E[i])
            a.rho = float(rho[i])
            a.v = float(v[i])
            a.pressure = float(pressure[i])
            a.u = float(u[i])
            a.H = float(H[i])

            if a.E <= 0.0:
                a.exhausted = True
                a.action = "retreat"
            elif a.exhausted and a.E >= EXHAUST_RECOVER_E:
                a.exhausted = False

"""30 Hz crowd physics simulation (plan.md section 5 + Phase C2 & Phase E2 rebalance).

Phase E2 highlights:
- Wider DJ hot zone segment x in [0.3, 0.7], y = 0.
- Ring-attenuated `local_music = music_energy * RING_GAINS[ring - 1]` used everywhere.
- Music charge at the front (`+g * v * local_music * w`), halved front-crush drain, rare exhaustion (<= 0.05 -> >= 0.25).
- Softened reward `(0.4 + 0.6*E) * (k*v*w + s*(1-rho)*local_music + m*v*local_music) / (k+s+m)`.
- Hype (`CONTAGION_KAPPA = 0.25`) + energy (`ENERGY_CONTAGION_KAPPA = 0.10`) neighbor contagion.
- Dynamic per-agent radii (archetype * proximity * energy deflation) used in soft collisions & wall clamps.
- Smoothed house-ear section probabilities scale movement: build -> approach, drop -> tight & bouncy on kick
  (including pushers), breakdown -> loosen, drift back, sway gently.
"""

from __future__ import annotations

import math

import numpy as np

from server.config import (
    AGENT_RADIUS,
    CONTAGION_KAPPA,
    CONTAGION_RADIUS_MULT,
    DENSITY_CAP,
    DENSITY_RADIUS_MULT,
    DJ_BOOTH_Y,
    DJ_POS,
    DJ_SEG_X0,
    DJ_SEG_X1,
    ENERGY_A,
    ENERGY_B,
    ENERGY_C,
    ENERGY_CONTAGION_KAPPA,
    ENERGY_D,
    ENERGY_G,
    EXHAUST_RECOVER_E,
    EXHAUST_TRIGGER_E,
    HAPPINESS_TAU,
    MASS_BASE,
    MASS_PUSH_SCALE,
    MAX_DIST_TO_DJ,
    MAX_SPEED,
    PHYSICS_HZ,
    REWARD_M,
    RING_EDGES,
    RING_GAINS,
    U_NORM_DIVISOR,
)
from server.state import Agent, CrowdState, X_MAX, X_MIN, Y_MAX, Y_MIN, dj_closest_point

GRID_SIZE: int = 10
PUSH_STEP: float = 0.10
YIELD_STEP: float = 0.075
REST_SEARCH_RADIUS: float = 0.28
RETREAT_Y: float = 0.94

# Soft spring collision parameters (allows ~15-25% squishy overlap in front crush, resolved over frames)
SOFT_COLLISION_ITERS: int = 4
SOFT_SPRING_RATE: float = 0.16  # gentle spring compliance per sub-step
MAX_OVERLAP_FRAC: float = 0.22  # firm inner core cap at 78% of sum of radii

# Idle Ornstein-Uhlenbeck wander & anti-stall parameters
OU_TAU: float = 1.6
OU_SIGMA: float = 0.044
STALL_SPEED_THRESH: float = 0.006
STALL_TRIGGER_SEC: float = 1.25
NUDGE_DECAY_TAU: float = 0.45

_RING_EDGES_ARR = np.array(RING_EDGES, dtype=np.float64)
_RING_GAINS_ARR = np.array(RING_GAINS, dtype=np.float64)

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
    r = max(0.015, agent.radius)
    x_lo, x_hi = r, 1.0 - r
    y_lo, y_hi = DJ_BOOTH_Y + r, 1.0 - r

    if action == "hold" or action == "dance":
        return (x, y)

    if action == "retreat":
        return (_clamp(x, x_lo, x_hi), _clamp(RETREAT_Y, y_lo, y_hi))

    dj_x, dj_y = dj_closest_point(x, y)

    if action == "push":
        dx = dj_x - x
        dy = dj_y - y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 0.0, -1.0
        return (
            _clamp(x + PUSH_STEP * ux, x_lo, x_hi),
            _clamp(y + PUSH_STEP * uy, y_lo, y_hi),
        )

    if action == "yield":
        dx = dj_x - x
        dy = dj_y - y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 0.0, -1.0
        c1x = _clamp(x + YIELD_STEP * (-uy), x_lo, x_hi)
        c1y = _clamp(y + YIELD_STEP * ux, y_lo, y_hi)
        c2x = _clamp(x + YIELD_STEP * uy, x_lo, x_hi)
        c2y = _clamp(y + YIELD_STEP * (-ux), y_lo, y_hi)

        d1 = int(grid[_cell_index(c1x), _cell_index(c1y)])
        d2 = int(grid[_cell_index(c2x), _cell_index(c2y)])
        if d1 < d2:
            return (c1x, c1y)
        if d2 < d1:
            return (c2x, c2y)

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
            return (_clamp(x, x_lo, x_hi), _clamp(y, y_lo, y_hi))

        counts = grid_flat[within]
        min_count = int(counts.min())
        best_candidates = within[counts == min_count]
        if best_candidates.size == 1:
            idx = int(best_candidates[0])
        else:
            # Prefer nearby open cells slightly away from the DJ
            scores = dists[best_candidates] - 0.5 * _CELL_CY_FLAT[best_candidates]
            idx = int(best_candidates[int(np.argmin(scores))])
        return (
            _clamp(float(_CELL_CX_FLAT[idx]), x_lo, x_hi),
            _clamp(float(_CELL_CY_FLAT[idx]), y_lo, y_hi),
        )

    return (x, y)


class PhysicsEngine:
    """Vectorized 30 Hz crowd physics engine with local music, section scaling, per-agent radii & contagion."""

    def __init__(
        self,
        target_refresh_seconds: float = 0.5,
        steer_tau: float = 0.16,
        arrive_tau: float = 0.22,
        enable_wander: bool = True,
    ) -> None:
        self.target_refresh_seconds = target_refresh_seconds
        self.steer_tau = steer_tau
        self.arrive_tau = arrive_tau
        self.enable_wander = enable_wander

        self._target_timer: float = target_refresh_seconds
        self._prev_actions: dict[int, str] = {}
        self._triu_n: int = -1
        self._triu_mask: np.ndarray = np.empty((0, 0), dtype=bool)

        # Per-agent smooth OU noise, stall timers, nudges, and kick envelope
        self._state_n: int = -1
        self._state_seed: int = -1
        self._rng: np.random.Generator = np.random.default_rng(42)
        self._ou: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self._stall_time: np.ndarray = np.zeros(0, dtype=np.float64)
        self._nudge_vel: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self._kick_env: float = 0.0
        self._dance_phase: float = 0.0
        self._last_kick_t: float = -1.0

    def reset(self) -> None:
        """Reset internal timers, OU wander state, and cached action history."""
        self._target_timer = self.target_refresh_seconds
        self._prev_actions.clear()
        self._state_n = -1
        self._state_seed = -1
        self._kick_env = 0.0
        self._dance_phase = 0.0
        self._last_kick_t = -1.0

    def _ensure_agent_arrays(self, n: int, seed: int) -> None:
        if self._state_n != n or self._state_seed != seed:
            self._state_n = n
            self._state_seed = seed
            self._rng = np.random.default_rng(seed + 7919)
            self._ou = self._rng.normal(0.0, OU_SIGMA * 0.5, size=(n, 2))
            self._stall_time = np.zeros(n, dtype=np.float64)
            self._nudge_vel = np.zeros((n, 2), dtype=np.float64)

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

    def _section_drives(self, crowd: CrowdState, dt: float) -> tuple[float, float, float, float]:
        """Return smoothed (p_build, p_drop, p_groove, p_break) driving physical movement scaling."""
        if crowd.use_dsp_proxy:
            # Smooth proxy for offline/shove-before-house-ear tests
            e5 = float(crowd.energy_5s)
            trend = float(crowd.dsp.get("trend", 0.0) or 0.0)
            raw_drop = _clamp((e5 - 0.58) / 0.32, 0.0, 1.0)
            raw_build = _clamp((trend - 0.06) / 0.25, 0.0, 1.0) * _clamp(e5 / 0.45, 0.0, 1.0)
            raw_break = _clamp((0.40 - e5) / 0.30, 0.0, 1.0)
            raw_groove = max(0.05, 1.0 - max(raw_drop, raw_build, raw_break))
            alpha = 1.0 - math.exp(-dt / 4.0)
            crowd.update_section_probs(
                {
                    "drop": raw_drop,
                    "build": raw_build,
                    "groove": raw_groove,
                    "breakdown": raw_break,
                    "intro": 0.0,
                    "outro": 0.0,
                    "other": 0.0,
                },
                alpha=alpha,
            )

        sp = crowd.section_probs_smooth
        p_drop = float(sp.get("drop", 0.0))
        p_build = float(sp.get("build", 0.0))
        p_groove = float(sp.get("groove", 0.0))
        p_break = (
            float(sp.get("breakdown", 0.0))
            + 0.55 * float(sp.get("intro", 0.0))
            + 0.55 * float(sp.get("outro", 0.0))
        )

        # Blend gently with sustained audio quiet/loud energy so breakdown loosening & drop bounce
        # stay unmistakable even when house-ear splits probability across breakdown/groove/intro.
        e5 = float(crowd.energy_5s)
        quiet_env = _clamp((0.50 - e5) / 0.30, 0.0, 1.0)
        loud_env = _clamp((e5 - 0.55) / 0.35, 0.0, 1.0)
        break_eff = max(p_break, 0.80 * quiet_env)
        drop_eff = max(p_drop, 0.45 * loud_env * (1.0 - break_eff))
        return p_build, drop_eff, p_groove, break_eff

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

        self._ensure_agent_arrays(n, crowd.seed)

        # Enforce exhaustion & active yield-override invariants before target computation
        for a in agents:
            if a.exhausted:
                a.action = "retreat"
                a.pushiness = 0.05
            elif a.yield_until > crowd.t:
                a.action = "yield"
                a.pushiness = 0.05

        # 1. Target by action (recompute when action changes or every 0.5 s)
        self._target_timer += dt
        if self._target_timer >= self.target_refresh_seconds:
            self._target_timer = 0.0
            self.recompute_targets(crowd, force=True)
        else:
            self.recompute_targets(crowd, force=False)

        # Pack state into contiguous numpy arrays
        pos = np.empty((n, 2), dtype=np.float64)
        vel = np.empty((n, 2), dtype=np.float64)
        target = np.empty((n, 2), dtype=np.float64)
        E = np.empty(n, dtype=np.float64)
        pushiness = np.empty(n, dtype=np.float64)
        k_arr = np.empty(n, dtype=np.float64)
        s_arr = np.empty(n, dtype=np.float64)
        size_mult = np.empty(n, dtype=np.float64)
        H = np.empty(n, dtype=np.float64)
        ids = np.empty(n, dtype=np.float64)
        wander_scale = np.ones(n, dtype=np.float64)
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
            size_mult[i] = a.size_mult
            H[i] = a.H
            ids[i] = float(a.id)
            if a.archetype == "dancer":
                wander_scale[i] = 1.45
            elif a.archetype == "diehard":
                wander_scale[i] = 0.72
            else:
                wander_scale[i] = 1.00

            act = a.action
            if act == "push":
                is_push[i] = 1.0
            elif act == "dance":
                is_dance[i] = 1.0
            elif act == "rest" or act == "retreat":
                is_rest_or_retreat[i] = True
            elif act == "hold":
                is_hold[i] = True

        prev_pos = pos.copy()
        music_energy = float(crowd.music_energy)

        # Pre-step DJ segment distance, proximity, ring-attenuated local_music, and per-agent radius
        dj_cx_pre = np.clip(pos[:, 0], DJ_SEG_X0, DJ_SEG_X1)
        to_dj_x = dj_cx_pre - pos[:, 0]
        to_dj_y = DJ_POS[1] - pos[:, 1]
        dist_dj_pre = np.hypot(to_dj_x, to_dj_y)
        inv_dj_dist = 1.0 / np.maximum(dist_dj_pre, 1e-6)
        ux_dj = to_dj_x * inv_dj_dist
        uy_dj = to_dj_y * inv_dj_dist

        v_pre = np.clip(1.0 - dist_dj_pre / MAX_DIST_TO_DJ, 0.0, 1.0)
        ring_idx_pre = np.searchsorted(_RING_EDGES_ARR, dist_dj_pre, side="right")
        local_music_pre = music_energy * _RING_GAINS_ARR[ring_idx_pre]
        radii = AGENT_RADIUS * size_mult * (0.80 + 0.45 * v_pre) * (0.85 + 0.15 * E)

        p_build, p_drop, p_groove, p_break = self._section_drives(crowd, dt)

        # Update kick envelope & dance bob phase
        self._kick_env *= math.exp(-dt / 0.14)
        if crowd.dsp.get("kick") and (crowd.t - self._last_kick_t > 0.12):
            self._kick_env = 1.0
            self._last_kick_t = crowd.t
        self._dance_phase += dt * (2.0 * math.pi * 2.05)

        # 2. Integrate: target steering + section movement scaling + local_music*E wander/bob + wall push + nudge
        section_speed = 1.0 + 0.30 * p_build + 0.20 * p_drop - 0.18 * p_break
        max_speed = MAX_SPEED * (0.4 + 0.6 * E) * max(0.65, section_speed)
        to_target = target - pos
        dist_target = np.hypot(to_target[:, 0], to_target[:, 1])

        v_des = np.zeros((n, 2), dtype=np.float64)
        moving = dist_target > 1e-6
        if np.any(moving):
            desired_speed = np.minimum(max_speed[moving], dist_target[moving] / self.arrive_tau)
            # On breakdowns, forward pushing eases off so the crowd genuinely loosens
            if self.enable_wander and p_break > 0.05:
                push_damp = 1.0 - 0.72 * p_break * is_push[moving]
                desired_speed = desired_speed * np.clip(push_damp, 0.22, 1.0)
            scale = desired_speed / dist_target[moving]
            v_des[moving, 0] = to_target[moving, 0] * scale
            v_des[moving, 1] = to_target[moving, 1] * scale

        if self.enable_wander:
            local_drive = local_music_pre * E
            not_retreating = (~is_rest_or_retreat).astype(np.float64)

            # Song-structure physical scaling (build -> approach, drop -> tight & bouncy, breakdown -> loosen/drift back)
            approach_mag = (
                (0.058 * p_build + 0.044 * p_drop + 0.014 * p_groove * music_energy)
                * (1.0 - p_break)
                * (0.35 + 0.65 * k_arr)
                * (0.55 + 0.45 * E)
                * not_retreating
            )
            v_des[:, 0] += ux_dj * approach_mag
            v_des[:, 1] += uy_dj * approach_mag

            if p_break > 0.05:
                # Breakdown loosening: smooth backward drift away from DJ segment into open mid-floor
                loosen_mag = 0.125 * p_break * np.clip((v_pre - 0.25) / 0.60, 0.0, 1.0) * not_retreating
                v_des[:, 0] -= ux_dj * loosen_mag
                v_des[:, 1] -= uy_dj * loosen_mag
                flank_dir = np.where(pos[:, 0] >= 0.5, 1.0, -1.0)
                v_des[:, 0] += flank_dir * (0.028 * p_break * v_pre * not_retreating)

            # Ornstein-Uhlenbeck smooth per-agent wander scaled by local_music * E
            ou_decay = math.exp(-dt / OU_TAU)
            ou_noise_std = OU_SIGMA * math.sqrt(max(0.0, 1.0 - ou_decay * ou_decay))
            self._ou = self._ou * ou_decay + self._rng.normal(0.0, ou_noise_std, size=(n, 2))

            liveliness = np.maximum(0.20, 0.20 + 0.98 * local_drive)
            wander_amp = liveliness * wander_scale * (1.0 - 0.18 * p_break)
            v_des[:, 0] += self._ou[:, 0] * wander_amp
            v_des[:, 1] += self._ou[:, 1] * wander_amp

            # Rhythmic sway & kick bounce: scales with local_music * E, max on drop (including pushers!)
            phi = self._dance_phase + ids * 0.73
            drop_bounce_scale = (0.28 + 0.95 * p_drop + 0.22 * p_build + 0.18 * p_groove) * (1.0 - 0.55 * p_break)
            # Dancers sway and bounce continuously; pushers/holders bounce hard on drops & kicks
            sway_factor = np.where(is_dance > 0.5, 1.0, 0.35 + 0.55 * p_drop) * not_retreating
            bob_amp = (
                (0.014 * np.where(is_dance > 0.5, 1.0, 0.45) + 0.068 * self._kick_env * drop_bounce_scale)
                * (0.22 + 1.05 * local_drive)
                * not_retreating
            )
            v_des[:, 0] += np.cos(0.5 * phi) * (0.024 * (0.22 + 0.90 * local_drive) * sway_factor)
            v_des[:, 1] += np.sin(phi) * bob_amp

            # Soft push away from walls so agents never glue to edges or back line
            wall_margin = 1.45 * radii
            wall_strength = 0.048
            x_lo_w = radii
            x_hi_w = 1.0 - radii
            y_lo_w = DJ_BOOTH_Y + radii
            y_hi_w = 1.0 - radii

            d_left = pos[:, 0] - x_lo_w
            d_right = x_hi_w - pos[:, 0]
            d_top = pos[:, 1] - y_lo_w
            d_bot = y_hi_w - pos[:, 1]

            v_des[:, 0] += wall_strength * np.square(np.clip((wall_margin - d_left) / wall_margin, 0.0, 1.0))
            v_des[:, 0] -= wall_strength * np.square(np.clip((wall_margin - d_right) / wall_margin, 0.0, 1.0))
            v_des[:, 1] -= wall_strength * np.square(np.clip((wall_margin - d_bot) / wall_margin, 0.0, 1.0))
            # On drops let front rows stay flush against DJ rail; on breakdowns gently push away from rail
            dj_wall_repel = (1.0 - is_push * (1.0 - p_break)) * (0.55 + 0.65 * p_break) * (1.0 - 0.55 * p_drop)
            v_des[:, 1] += (wall_strength * dj_wall_repel) * np.square(
                np.clip((wall_margin - d_top) / wall_margin, 0.0, 1.0)
            )

            # Decaying anti-stall nudge velocity
            self._nudge_vel *= math.exp(-dt / NUDGE_DECAY_TAU)
            v_des += self._nudge_vel

        steer_alpha = 1.0 - math.exp(-dt / self.steer_tau)
        vel += steer_alpha * (v_des - vel)

        speed = np.hypot(vel[:, 0], vel[:, 1])
        # Allow kick-bob / drop-bounce headroom above steady max_speed
        speed_cap = max_speed * (1.0 + 0.35 * self._kick_env * (0.4 + 0.6 * p_drop) * np.maximum(is_dance, 0.65))
        too_fast = speed > speed_cap
        if np.any(too_fast):
            clamp_scale = speed_cap[too_fast] / speed[too_fast]
            vel[too_fast, 0] *= clamp_scale
            vel[too_fast, 1] *= clamp_scale

        pos += vel * dt

        # 3. Soft spring-like collisions using per-agent radii (allows ~15-25% squishy overlap)
        pressure = np.zeros(n, dtype=np.float64)
        mass = MASS_BASE + MASS_PUSH_SCALE * (pushiness * pushiness) * E
        inv_mass = 1.0 / mass
        min_dist_mat = radii[:, None] + radii[None, :]
        min_dist_sq_mat = min_dist_mat * min_dist_mat
        triu_mask = self._get_triu_mask(n)

        x_lo = radii
        x_hi = 1.0 - radii
        y_lo = DJ_BOOTH_Y + radii
        y_hi = 1.0 - radii

        for iter_idx in range(SOFT_COLLISION_ITERS):
            dx = pos[:, 0:1] - pos[None, :, 0]
            dy = pos[:, 1:2] - pos[None, :, 1]
            dist_sq = dx * dx + dy * dy

            ii, jj = np.nonzero((dist_sq < min_dist_sq_mat) & triu_mask)
            if ii.size > 0:
                pdx = dx[ii, jj]
                pdy = dy[ii, jj]
                pdist = np.sqrt(dist_sq[ii, jj])
                pair_min = min_dist_mat[ii, jj]
                overlap = pair_min - pdist
                core_dist = (1.0 - MAX_OVERLAP_FRAC) * pair_min

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

                soft_sep = overlap * SOFT_SPRING_RATE
                core_stiff = 0.96 if iter_idx >= SOFT_COLLISION_ITERS - 2 else 0.82
                core_excess = np.maximum(0.0, core_dist - pdist) * core_stiff
                sep = np.maximum(soft_sep, core_excess)

                move_i = sep * (im_i / inv_sum)
                move_j = sep * (im_j / inv_sum)

                np.add.at(pos[:, 0], ii, move_i * nx)
                np.add.at(pos[:, 1], ii, move_i * ny)
                np.add.at(pos[:, 0], jj, -move_j * nx)
                np.add.at(pos[:, 1], jj, -move_j * ny)

                # Shoulder-slip & jostle on first iteration so heavier pushers weave past lighter blockers
                if iter_idx == 0:
                    push_pair = np.maximum(is_push[ii], is_push[jj])
                    if np.any(push_pair > 0.0):
                        tx = -ny
                        ty = nx
                        jostle_sign = np.sin(crowd.t * 3.2 + ii * 1.7 - jj * 2.3)
                        side = np.where(
                            np.abs(pdx) > 1e-4,
                            np.sign(pdx),
                            np.where(jostle_sign >= 0.0, 1.0, -1.0),
                        )
                        mass_diff = np.abs(mass[ii] - mass[jj]) / (mass[ii] + mass[jj])
                        slip = sep * push_pair * (0.85 * mass_diff * side + 0.50 * jostle_sign)
                        np.add.at(pos[:, 0], ii, slip * tx * (im_i / inv_sum))
                        np.add.at(pos[:, 1], ii, slip * ty * (im_i / inv_sum))
                        np.add.at(pos[:, 0], jj, -slip * tx * (im_j / inv_sum))
                        np.add.at(pos[:, 1], jj, -slip * ty * (im_j / inv_sum))

                        i_hb = (mass[ii] > mass[jj] * 1.35) & (is_push[ii] > 0.5) & (pos[ii, 1] > pos[jj, 1])
                        j_hb = (mass[jj] > mass[ii] * 1.35) & (is_push[jj] > 0.5) & (pos[jj, 1] > pos[ii, 1])
                        if np.any(i_hb):
                            k_idx = np.nonzero(i_hb)[0]
                            s_y = sep[k_idx] * 0.55 * mass_diff[k_idx]
                            np.add.at(pos[:, 1], ii[k_idx], -s_y)
                            np.add.at(pos[:, 1], jj[k_idx], +s_y)
                        if np.any(j_hb):
                            k_idx = np.nonzero(j_hb)[0]
                            s_y = sep[k_idx] * 0.55 * mass_diff[k_idx]
                            np.add.at(pos[:, 1], jj[k_idx], -s_y)
                            np.add.at(pos[:, 1], ii[k_idx], +s_y)

                np.add.at(pressure, ii, overlap)
                np.add.at(pressure, jj, overlap)

            # Clamp to floor walls using per-agent radius (DJ booth y < DJ_BOOTH_Y is a wall)
            pen_x_lo = np.maximum(0.0, x_lo - pos[:, 0])
            pen_x_hi = np.maximum(0.0, pos[:, 0] - x_hi)
            pen_y_lo = np.maximum(0.0, y_lo - pos[:, 1])
            pen_y_hi = np.maximum(0.0, pos[:, 1] - y_hi)
            pressure += pen_x_lo + pen_x_hi + pen_y_lo + pen_y_hi

            hit_x_lo = pos[:, 0] < x_lo
            hit_x_hi = pos[:, 0] > x_hi
            hit_y_lo = pos[:, 1] < y_lo
            hit_y_hi = pos[:, 1] > y_hi

            pos[:, 0] = np.clip(pos[:, 0], x_lo, x_hi)
            pos[:, 1] = np.clip(pos[:, 1], y_lo, y_hi)

            if np.any(hit_x_lo):
                vel[hit_x_lo, 0] = np.maximum(vel[hit_x_lo, 0], 0.0)
            if np.any(hit_x_hi):
                vel[hit_x_hi, 0] = np.minimum(vel[hit_x_hi, 0], 0.0)
            if np.any(hit_y_lo):
                vel[hit_y_lo, 1] = np.maximum(vel[hit_y_lo, 1], 0.0)
            if np.any(hit_y_hi):
                vel[hit_y_hi, 1] = np.minimum(vel[hit_y_hi, 1], 0.0)

        # Anti-stall nudge: if any agent's actual post-collision speed stays near zero, nudge it
        actual_vel = (pos - prev_pos) / max(dt, 1e-6)
        actual_speed = np.hypot(actual_vel[:, 0], actual_vel[:, 1])
        if self.enable_wander:
            stalled = actual_speed < STALL_SPEED_THRESH
            self._stall_time = np.where(stalled, self._stall_time + dt, np.maximum(0.0, self._stall_time - 3.0 * dt))
            need_nudge = self._stall_time >= STALL_TRIGGER_SEC
            if np.any(need_nudge):
                n_idx = np.nonzero(need_nudge)[0]
                angles = self._rng.uniform(0.0, 2.0 * math.pi, size=n_idx.size)
                nx_n = np.cos(angles)
                ny_n = np.sin(angles)
                near_l = pos[n_idx, 0] < x_lo[n_idx] + 1.2 * radii[n_idx]
                near_r = pos[n_idx, 0] > x_hi[n_idx] - 1.2 * radii[n_idx]
                near_t = pos[n_idx, 1] < y_lo[n_idx] + 1.2 * radii[n_idx]
                near_b = pos[n_idx, 1] > y_hi[n_idx] - 1.2 * radii[n_idx]
                nx_n = np.where(near_l, np.abs(nx_n), np.where(near_r, -np.abs(nx_n), nx_n))
                ny_n = np.where(near_t, np.abs(ny_n) * 0.35, np.where(near_b, -np.abs(ny_n), ny_n))
                norm_n = np.maximum(np.hypot(nx_n, ny_n), 1e-6)
                nudge_mag = 0.055
                self._nudge_vel[n_idx, 0] = (nx_n / norm_n) * nudge_mag
                self._nudge_vel[n_idx, 1] = (ny_n / norm_n) * nudge_mag
                pos[n_idx, 0] = np.clip(pos[n_idx, 0] + self._nudge_vel[n_idx, 0] * dt * 2.5, x_lo[n_idx], x_hi[n_idx])
                pos[n_idx, 1] = np.clip(pos[n_idx, 1] + self._nudge_vel[n_idx, 1] * dt * 2.5, y_lo[n_idx], y_hi[n_idx])
                self._stall_time[n_idx] = 0.0
                actual_vel = (pos - prev_pos) / max(dt, 1e-6)

        # Keep reported velocity faithful to real on-screen movement (used for velocity stretch & trails)
        vel = 0.45 * vel + 0.55 * actual_vel

        # 4. Derive post-collision rho, v (to DJ segment), ring-attenuated local_music
        dx = pos[:, 0:1] - pos[None, :, 0]
        dy = pos[:, 1:2] - pos[None, :, 1]
        dist_sq = dx * dx + dy * dy

        density_r_mat = DENSITY_RADIUS_MULT * 0.5 * (radii[:, None] + radii[None, :])
        rho_neighbors = dist_sq <= (density_r_mat * density_r_mat)
        np.fill_diagonal(rho_neighbors, False)
        rho = np.minimum(rho_neighbors.sum(axis=1), DENSITY_CAP) / float(DENSITY_CAP)

        dj_cx = np.clip(pos[:, 0], DJ_SEG_X0, DJ_SEG_X1)
        dist_dj = np.hypot(pos[:, 0] - dj_cx, pos[:, 1] - DJ_POS[1])
        v = np.clip(1.0 - dist_dj / MAX_DIST_TO_DJ, 0.0, 1.0)

        ring_idx = np.searchsorted(_RING_EDGES_ARR, dist_dj, side="right")
        local_music = music_energy * _RING_GAINS_ARR[ring_idx]

        # Contagion neighborhood (used for both E contagion and H contagion)
        contagion_r_sq = (CONTAGION_RADIUS_MULT * AGENT_RADIUS) ** 2
        cont_neighbors = dist_sq <= contagion_r_sq
        np.fill_diagonal(cont_neighbors, False)
        cont_counts = cont_neighbors.sum(axis=1)
        has_cont = cont_counts > 0

        # 5. Energy rebalance: music charge at front feeds you, pushing/long dancing tires you + E contagion
        if crowd.use_dsp_proxy:
            crowd.w = 0.5 + 2.5 * crowd.energy_5s
        w = float(crowd.w)

        recover_mask = (is_rest_or_retreat | (is_hold & (rho < 0.25))).astype(np.float64)
        dE_dt = (
            +ENERGY_G * v * local_music * w
            - ENERGY_A * v * rho
            - ENERGY_B * is_push
            - ENERGY_D * is_dance
            + ENERGY_C * (1.0 - v) * (1.0 - rho) * recover_mask
        )
        E = np.clip(E + dE_dt * dt, 0.0, 1.0)

        if np.any(has_cont):
            neighbor_e_sum = cont_neighbors[has_cont].astype(np.float64) @ E
            mean_e_neighbors = neighbor_e_sum / cont_counts[has_cont]
            E[has_cont] = np.clip(
                E[has_cont] + ENERGY_CONTAGION_KAPPA * (mean_e_neighbors - E[has_cont]) * dt,
                0.0,
                1.0,
            )

        # 6. Reward & Happiness: fatigue dampens (doesn't kill), everyone loves loud music + H contagion
        denom = np.maximum(k_arr + s_arr + REWARD_M, 1e-6)
        u = (
            (0.4 + 0.6 * E)
            * (k_arr * v * w + s_arr * (1.0 - rho) * local_music + REWARD_M * v * local_music)
            / denom
        )
        u_norm = np.clip(u / U_NORM_DIVISOR, 0.0, 1.0)

        ema_alpha = 1.0 - math.exp(-dt / HAPPINESS_TAU)
        H = H + ema_alpha * (u_norm - H)

        if np.any(has_cont):
            neighbor_h_sum = cont_neighbors[has_cont].astype(np.float64) @ H
            mean_h_neighbors = neighbor_h_sum / cont_counts[has_cont]
            H[has_cont] = H[has_cont] + CONTAGION_KAPPA * (mean_h_neighbors - H[has_cont]) * dt

        H = np.clip(H, 0.0, 1.0)

        # Post-step updated per-agent radius
        radii_post = AGENT_RADIUS * size_mult * (0.80 + 0.45 * v) * (0.85 + 0.15 * E)

        # Write back to Agent objects and apply rare exhaustion transitions (<= 0.05 -> >= 0.25)
        for i, a in enumerate(agents):
            a.x = float(pos[i, 0])
            a.y = float(pos[i, 1])
            a.vx = float(vel[i, 0])
            a.vy = float(vel[i, 1])
            a.E = float(E[i])
            a.rho = float(rho[i])
            a.v = float(v[i])
            a.local_music = float(local_music[i])
            a.radius = float(radii_post[i])
            a.pressure = float(pressure[i])
            a.u = float(u[i])
            a.H = float(H[i])

            if a.E <= EXHAUST_TRIGGER_E:
                a.exhausted = True
                a.action = "retreat"
            elif a.exhausted and a.E >= EXHAUST_RECOVER_E:
                a.exhausted = False

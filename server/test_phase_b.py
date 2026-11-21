"""Phase B verification suite: state, shove, 30 Hz physics, contagion, exhaustion, and /ws streaming.

Run with:
    .venv/bin/python -m server.test_phase_b
"""

from __future__ import annotations

import math
import time
from collections import Counter

from fastapi.testclient import TestClient

from server.config import (
    AGENT_RADIUS,
    CONTAGION_KAPPA,
    DJ_BOOTH_Y,
    EXHAUST_RECOVER_E,
    HAPPINESS_TAU,
    MASS_BASE,
    MASS_PUSH_SCALE,
    N_AGENTS,
    N_LANES,
    PHYSICS_HZ,
)
from server.fake_agents import run_headless_shove_demo
from server.main import app, runtime
from server.physics import PhysicsEngine
from server.shove import apply_shove_actions
from server.state import (
    LANE_LABELS,
    VALID_ACTIONS,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Agent,
    CrowdState,
    dist_to_dj,
    init_crowd,
    ring_for_agent,
)


def test_state_and_seeded_init() -> None:
    print("\n=== 1. STATE & SEEDED INITIALIZATION ===")
    c1 = init_crowd(seed=42, condition="shove")
    c2 = init_crowd(seed=42, condition="shove")

    assert len(c1.agents) == N_AGENTS == 100
    assert c1.lane_labels == LANE_LABELS
    assert len(c1.yielded_to) == N_AGENTS

    # Exact determinism across same seed
    for a, b in zip(c1.agents, c2.agents):
        assert (a.id, a.archetype, a.lane, a.k, a.s, a.x, a.y) == (
            b.id,
            b.archetype,
            b.lane,
            b.k,
            b.s,
            b.x,
            b.y,
        )

    arch_counts = Counter(a.archetype for a in c1.agents)
    assert arch_counts == {"diehard": 30, "dancer": 30, "between": 40}, arch_counts

    lane_counts = Counter(a.lane for a in c1.agents)
    assert len(lane_counts) == N_LANES and all(v == 10 for v in lane_counts.values()), lane_counts

    for a in c1.agents:
        assert 0.0 <= a.k <= 1.0 and 0.0 <= a.s <= 1.0
        assert X_MIN <= a.x <= X_MAX
        assert 0.85 <= a.y <= min(0.97, Y_MAX)
        assert 0.0 <= a.v <= 1.0
        assert 0.0 <= a.rho <= 1.0

    # Ring mapping check across all 5 rings
    ring_probes = [(0.5, 0.10, 1), (0.5, 0.30, 2), (0.5, 0.50, 3), (0.5, 0.70, 4), (0.5, 0.90, 5)]
    for px, py, expected_ring in ring_probes:
        probe = Agent(id=999, x=px, y=py)
        assert ring_for_agent(probe) == expected_ring, (px, py, dist_to_dj(px, py))

    # Payload schemas
    init_msg = c1.init_payload()
    assert init_msg["type"] == "init"
    assert init_msg["n_lanes"] == 10
    assert len(init_msg["agents"]) == 100
    assert set(init_msg["agents"][0].keys()) == {"id", "archetype", "lane", "k", "s"}

    phys_msg = c1.physics_payload()
    assert phys_msg["type"] == "physics"
    assert len(phys_msg["agents"]) == 100
    assert len(phys_msg["agents"][0]) == 10
    print("PASS: deterministic init, traits/lanes, rings 1..5, init & physics schemas")


def test_collision_inverse_mass_and_walls() -> None:
    print("\n=== 2. COLLISION INVERSE-MASS SPLIT, WALLS & YIELD ===")
    # Heavy vs light overlapping pair
    heavy = Agent(id=0, x=0.49, y=0.50, E=1.0, pushiness=0.9, action="hold", target=(0.49, 0.50))
    light = Agent(id=1, x=0.51, y=0.50, E=0.2, pushiness=0.1, action="hold", target=(0.51, 0.50))
    m_heavy = MASS_BASE + MASS_PUSH_SCALE * (heavy.pushiness ** 2) * heavy.E
    m_light = MASS_BASE + MASS_PUSH_SCALE * (light.pushiness ** 2) * light.E
    expected_ratio = m_heavy / m_light
    crowd = CrowdState(agents=[heavy, light], playing=True)
    engine = PhysicsEngine(enable_wander=False)
    x0_h, x0_l = heavy.x, light.x

    # Soft spring collision resolves smoothly over a few frames while honoring exact inverse-mass split
    engine.step(crowd, dt=1.0 / PHYSICS_HZ)
    dx_h = abs(heavy.x - x0_h)
    dx_l = abs(light.x - x0_l)
    sep_1 = abs(light.x - heavy.x)
    ratio = dx_l / max(dx_h, 1e-9)
    assert sep_1 > 0.02 and sep_1 < 2.0 * AGENT_RADIUS, f"Expected soft partial separation on frame 1, got {sep_1}"
    assert abs(ratio - expected_ratio) / expected_ratio < 0.05, (
        f"Expected ~{expected_ratio:.2f}:1 inverse-mass displacement ratio, got {ratio:.3f}"
    )
    assert heavy.pressure > 0.0 and light.pressure > 0.0
    # Over ~12 frames the soft spring resolves cleanly to full diameter
    for _ in range(15):
        engine.step(crowd, dt=1.0 / PHYSICS_HZ)
    sep = abs(light.x - heavy.x)
    assert sep >= 2.0 * AGENT_RADIUS - 2e-3, f"Expected soft spring to settle near {2*AGENT_RADIUS}, got {sep}"

    # Wall clamping at DJ booth (y < DJ_BOOTH_Y + AGENT_RADIUS)
    wall_agent = Agent(id=2, x=0.5, y=0.01, vy=-0.05, action="push", target=(0.5, 0.0))
    crowd_wall = CrowdState(agents=[wall_agent], playing=True)
    engine_wall = PhysicsEngine()
    engine_wall.step(crowd_wall, dt=1.0 / PHYSICS_HZ)
    assert wall_agent.y >= DJ_BOOTH_Y + AGENT_RADIUS - 1e-9
    assert wall_agent.y <= Y_MAX + 1e-9

    # Yield sidestep picks less dense perpendicular side
    # Agent at (0.5, 0.5) facing DJ (0.5, 0.0) -> perpendicular is left (-x) vs right (+x)
    # Place dense cluster on right side x=0.56, y=0.50
    yielder = Agent(id=10, x=0.50, y=0.50, action="yield")
    blockers_right = [Agent(id=11 + i, x=0.56, y=0.48 + 0.01 * i, action="hold") for i in range(4)]
    crowd_y = CrowdState(agents=[yielder, *blockers_right], playing=True)
    engine_y = PhysicsEngine()
    engine_y.recompute_targets(crowd_y, force=True)
    assert yielder.target[0] < yielder.x, f"Expected yielder to sidestep left away from crowd, got {yielder.target}"
    print(
        f"PASS: inverse-mass ratio={ratio:.2f} (expected {expected_ratio:.2f}), "
        f"sep={sep:.4f}, wall & yield verified"
    )


def test_exhaustion_retreat_and_recovery() -> None:
    print("\n=== 3. DIRECT EXHAUSTION -> RETREAT -> RECOVERY TEST ===")
    # Place a low-energy diehard right in front of the DJ surrounded by 6 neighbors
    hero = Agent(id=0, x=0.50, y=0.10, E=0.015, k=0.9, s=0.1, archetype="diehard", action="push", pushiness=0.9)
    crowd_list = [hero]
    offsets = [(-0.04, 0.0), (0.04, 0.0), (-0.02, 0.04), (0.02, 0.04), (-0.04, 0.05), (0.04, 0.05)]
    for idx, (ox, oy) in enumerate(offsets, start=1):
        crowd_list.append(
            Agent(
                id=idx,
                x=0.50 + ox,
                y=0.10 + oy,
                E=0.8,
                k=0.5,
                s=0.5,
                archetype="between",
                action="hold",
                pushiness=0.3,
            )
        )

    crowd = CrowdState(agents=crowd_list, condition="shove", playing=True, music_energy=0.8, energy_5s=0.8)
    engine = PhysicsEngine()
    dt = 1.0 / PHYSICS_HZ

    entered_exhausted_t: float | None = None
    recovered_t: float | None = None
    max_retreat_y = hero.y

    for step_idx in range(int(90.0 * PHYSICS_HZ)):
        engine.step(crowd, dt=dt)
        if hero.exhausted:
            if entered_exhausted_t is None:
                entered_exhausted_t = crowd.t
                assert hero.action == "retreat", f"Expected retreat on exhaustion, got {hero.action}"
            # Verify apply_shove_actions respects exhausted lock
            apply_shove_actions(crowd)
            assert hero.action == "retreat", "apply_shove_actions overwrote exhausted agent action!"
            max_retreat_y = max(max_retreat_y, hero.y)
        elif entered_exhausted_t is not None:
            recovered_t = crowd.t
            assert hero.E >= EXHAUST_RECOVER_E - 1e-9
            break

    assert entered_exhausted_t is not None, "Front low-energy agent never triggered exhaustion!"
    assert recovered_t is not None, f"Exhausted agent never recovered to E >= 0.3 (E={hero.E:.3f}, y={hero.y:.3f})"
    assert max_retreat_y > 0.45, f"Exhausted agent barely retreated (max_y={max_retreat_y:.3f})"
    print(
        f"PASS: exhausted at t={entered_exhausted_t:.2f}s -> retreated to y={max_retreat_y:.2f} -> "
        f"recovered (E={hero.E:.3f}, exhausted=False) at t={recovered_t:.2f}s"
    )


def test_happiness_contagion() -> None:
    print("\n=== 4. HAPPINESS CONTAGION UNIT TEST ===")
    dt = 1.0 / PHYSICS_HZ
    # Cold agent at (0.50, 0.50) surrounded by 4 warm neighbors within 3 * AGENT_RADIUS (0.084)
    # Separated by 0.070 (> 2 * AGENT_RADIUS = 0.056 so zero collision, <= 0.084 so within contagion radius)
    # Set E=0.0 on cold agents so u=0 (EMA pulls H downward); only contagion can pull cold_near upward!
    cold_near = Agent(id=0, x=0.50, y=0.50, E=0.0, H=0.10, action="hold", target=(0.50, 0.50))
    warm_offsets = [(-0.070, 0.0), (0.070, 0.0), (0.0, -0.070), (0.0, 0.070)]
    warm_neighbors = [
        Agent(
            id=1 + i,
            x=0.50 + ox,
            y=0.50 + oy,
            E=0.0,
            H=0.90,
            action="hold",
            target=(0.50 + ox, 0.50 + oy),
        )
        for i, (ox, oy) in enumerate(warm_offsets)
    ]
    # Isolated control cold agent far away (no neighbors)
    cold_isolated = Agent(id=9, x=0.10, y=0.90, E=0.0, H=0.10, action="hold", target=(0.10, 0.90))

    crowd = CrowdState(
        agents=[cold_near, *warm_neighbors, cold_isolated],
        playing=True,
        use_dsp_proxy=False,
        w=0.0,
        music_energy=0.0,
    )
    engine = PhysicsEngine()

    # Single-step analytical verification
    ema_alpha = 1.0 - math.exp(-dt / HAPPINESS_TAU)
    h_cold_after_ema = 0.10 + ema_alpha * (0.0 - 0.10)
    h_warm_after_ema = 0.90 + ema_alpha * (0.0 - 0.90)
    expected_cold_step1 = h_cold_after_ema + CONTAGION_KAPPA * (h_warm_after_ema - h_cold_after_ema) * dt

    engine.step(crowd, dt=dt)
    assert abs(cold_near.H - expected_cold_step1) < 1e-9, (
        f"Step-1 contagion mismatch: got {cold_near.H:.10f}, expected {expected_cold_step1:.10f}"
    )
    assert abs(cold_isolated.H - h_cold_after_ema) < 1e-9

    # Run for 2 full seconds (59 more steps) with warm neighbors kept warm
    for _ in range(2 * PHYSICS_HZ - 1):
        for w_agent in warm_neighbors:
            w_agent.H = 0.90
        cold_near.E = 0.0
        cold_isolated.E = 0.0
        engine.step(crowd, dt=dt)

    assert cold_near.H > 0.25, f"Expected cold_near H to rise strongly via contagion, got {cold_near.H:.4f}"
    assert cold_isolated.H < 0.09, f"Expected isolated cold agent H to decay via EMA, got {cold_isolated.H:.4f}"
    print(
        f"PASS: exact 1-step contagion match (<1e-9); after 2s cold_near.H={cold_near.H:.4f} vs "
        f"cold_isolated.H={cold_isolated.H:.4f}"
    )


def test_headless_shove_envelope() -> dict:
    print("\n=== 5. HEADLESS SHOVE RUN (QUIET -> LOUD -> QUIET -> LOUD) ===")
    res = run_headless_shove_demo(seed=42)
    pm = res["phase_metrics"]

    print(
        f"Timing: avg={res['avg_step_ms']:.3f} ms, p95={res['p95_step_ms']:.3f} ms, "
        f"max={res['max_step_ms']:.3f} ms ({res['total_steps']} steps)"
    )
    for name in ("intro", "drop1", "breakdown", "drop2"):
        m = pm[name]
        print(
            f"  {name:10s} | v: {m['start_v']:.3f}->{m['end_v']:.3f} (max {m['max_v']:.3f}) | "
            f"E: {m['start_E']:.3f}->{m['end_E']:.3f} | rho_mean={m['mean_rho']:.3f} | "
            f"H_end={m['end_H']:.3f} | max_exh={int(m['max_exhausted'])}"
        )

    # Performance
    assert res["avg_step_ms"] < 5.0, f"Step too slow: {res['avg_step_ms']:.3f} ms"
    assert res["avg_step_ms"] < 2.0, f"Exceeded 2 ms optimization target: {res['avg_step_ms']:.3f} ms"

    # Bounds & soft spring collision compliance (~15-25% max squishy overlap in crush)
    assert res["max_wall_violation"] == 0.0, f"Wall violation detected: {res['max_wall_violation']}"
    assert res["final_max_overlap"] < 0.35 * 2.0 * AGENT_RADIUS, (
        f"Excessive soft overlap: {res['final_max_overlap']}"
    )

    # Compression on loud sections
    drop1_v_delta = pm["drop1"]["max_v"] - pm["intro"]["start_v"]
    assert pm["drop1"]["max_v"] > pm["intro"]["end_v"] + 0.10, "Drop 1 failed to compress toward DJ"
    assert pm["drop2"]["max_v"] > pm["breakdown"]["end_v"] + 0.03, "Drop 2 failed to re-compress toward DJ"

    # Dispersal & recovery on quiet breakdown (compare against drop1 peak compression)
    assert pm["breakdown"]["end_v"] < pm["drop1"]["max_v"] - 0.05, "Breakdown failed to disperse back"
    assert pm["breakdown"]["mean_rho"] < pm["drop1"]["mean_rho"] - 0.10, "Breakdown failed to reduce density"
    assert pm["breakdown"]["end_E"] > pm["drop1"]["end_E"] + 0.02, "Breakdown failed to recover crowd energy"

    # Exhaustion & recovery happened organically during the shove run
    assert res["exhaustion_events"] > 0, "Expected organic exhaustion events during loud drop crush"
    assert res["recovery_events"] > 0, "Expected organic exhaustion recoveries during breakdown"

    print(
        f"PASS: compression delta(drop1_max - intro_end)=+{pm['drop1']['max_v'] - pm['intro']['end_v']:.3f}, "
        f"total +{drop1_v_delta:.3f}; breakdown E recovery=+{pm['breakdown']['end_E'] - pm['drop1']['end_E']:.3f}; "
        f"exhaustion={res['exhaustion_events']} recoveries={res['recovery_events']}"
    )
    return res


def test_websocket_streaming() -> dict:
    print("\n=== 6. FASTAPI /health, /ws/echo & /ws 30 Hz STREAMING ===")
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}

        # Verify /ws/echo still works for JSON + binary
        with client.websocket_connect("/ws/echo") as echo_ws:
            echo_ws.send_json({"ping": 1})
            assert echo_ws.receive_json() == {"type": "echo", "data": {"ping": 1}}
            pcm_test = bytes(range(256)) * 4
            echo_ws.send_bytes(pcm_test)
            assert echo_ws.receive_bytes() == pcm_test

        # Verify primary /ws contract
        with client.websocket_connect("/ws") as ws:
            init_frame = ws.receive_json()
            assert init_frame["type"] == "init", init_frame
            assert init_frame["n_lanes"] == 10
            assert len(init_frame["agents"]) == 100
            for entry in init_frame["agents"]:
                assert set(entry.keys()) >= {"id", "archetype", "lane", "k", "s"}

            # Send clock with playing=True and loud DSP
            ws.send_json(
                {
                    "type": "clock",
                    "t": 12.0,
                    "playing": True,
                    "source": "file",
                    "dsp": {
                        "sub": 0.7,
                        "bass": 0.85,
                        "mid": 0.4,
                        "high": 0.3,
                        "rms": 0.65,
                        "kick": True,
                        "energy": 0.85,
                        "energy_5s": 0.80,
                        "trend": 0.15,
                        "bass_gap": 0.0,
                    },
                }
            )

            # Send 250 ms binary PCM chunk (16 kHz Int16 mono = 8000 bytes)
            pcm_before = runtime.pcm_bytes_received
            ws.send_bytes(b"\x00\x01" * 4000)

            # Collect physics frames and measure streaming rate
            t_start = time.perf_counter()
            frames = []
            while len(frames) < 12:
                msg = ws.receive_json()
                if msg.get("type") == "physics":
                    frames.append(msg)
            elapsed = time.perf_counter() - t_start

            # Validate Section 9 physics schema on every frame
            for frame in frames:
                assert frame["type"] == "physics"
                assert isinstance(frame["t"], (int, float))
                agents = frame["agents"]
                assert len(agents) == 100
                for row in agents:
                    assert len(row) == 10
                    aid, x, y, vx, vy, E, rho, H, action, pressure = row
                    assert isinstance(aid, int)
                    assert X_MIN - 1e-3 <= x <= X_MAX + 1e-3
                    assert Y_MIN - 1e-3 <= y <= Y_MAX + 1e-3
                    assert 0.0 <= E <= 1.0
                    assert 0.0 <= rho <= 1.0
                    assert 0.0 <= H <= 1.0
                    assert action in VALID_ACTIONS
                    assert pressure >= 0.0

            # Confirm simulation advanced while playing and shove actions applied
            playing_frames = [f for f in frames if f["t"] >= 12.0]
            assert len(playing_frames) >= 2, f"Expected frames with t >= 12.0, got {[f['t'] for f in frames]}"
            assert playing_frames[-1]["t"] > playing_frames[0]["t"], "Simulation clock did not advance"
            actions_seen = {row[8] for row in playing_frames[-1]["agents"]}
            assert "push" in actions_seen, f"Expected shove push actions on loud clock, got {actions_seen}"

            # Confirm binary PCM chunk was accepted cleanly
            assert runtime.pcm_bytes_received >= pcm_before + 8000

            # Test control messages: pause, condition switch, reset
            ws.send_json({"type": "control", "action": "pause"})
            ws.send_json({"type": "control", "condition": "silent"})
            ws.send_json({"type": "control", "action": "reset", "seed": 7})

            saw_reset_init = False
            for _ in range(30):
                msg = ws.receive_json()
                if msg.get("type") == "init":
                    saw_reset_init = True
                    break
            assert saw_reset_init, "Did not receive fresh init frame after control reset"
            assert runtime.crowd.seed == 7
            assert runtime.crowd.condition == "silent"
            assert runtime.crowd.playing is False

            # Restore default seed=42, condition="shove" for clean server state
            ws.send_json({"type": "control", "condition": "shove", "action": "reset", "seed": 42})

        # Verify ?autostart=1 starts playing automatically on connect
        with client.websocket_connect("/ws?autostart=1") as ws_auto:
            auto_init = ws_auto.receive_json()
            assert auto_init["type"] == "init"
            assert runtime.crowd.playing is True
            ws_auto.send_json({"type": "control", "action": "pause"})
            ws_auto.send_json({"type": "control", "condition": "shove", "action": "reset", "seed": 42})

    approx_hz = len(frames) / max(elapsed, 1e-6)
    print(
        f"PASS: /ws received init + {len(frames)} physics frames in {elapsed:.3f}s (~{approx_hz:.1f} Hz), "
        f"clock/control/binary PCM verified"
    )
    return {"frames": len(frames), "elapsed_s": elapsed, "approx_hz": approx_hz}


def main() -> None:
    test_state_and_seeded_init()
    test_collision_inverse_mass_and_walls()
    test_exhaustion_retreat_and_recovery()
    test_happiness_contagion()
    test_headless_shove_envelope()
    test_websocket_streaming()
    print("\nALL PHASE B CHECKS PASSED.")


if __name__ == "__main__":
    main()

"""Shared configuration and constants (plan.md section 3)."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Gemini models & credentials
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
MODEL_DECISIONS: str = os.getenv("MODEL_DECISIONS", "gemini-decision-1")
MODEL_HYDROGEN: str = os.getenv("MODEL_HYDROGEN", "gemini-4-hydrogen")
MODEL_NEON: str = os.getenv("MODEL_NEON", "gemini-4-neon")

# Paths
TRACKS_DIR = ROOT_DIR / "tracks"
RUNS_DIR = ROOT_DIR / "runs"

# Floor & agents (Phase E3: compact 0.70 x 0.70 floor for denser crowd weaving & bigger visual blobs)
N_AGENTS: int = 100
N_LANES: int = 10
FLOOR_X_MIN: float = 0.15
FLOOR_X_MAX: float = 0.85
FLOOR_Y_MIN: float = 0.0
FLOOR_Y_MAX: float = 0.70
FLOOR_SPAN: float = 0.70  # isotropic 0.70 x 0.70 floor mapped onto full square TLC plate
DJ_POS: tuple[float, float] = (0.5, 0.0)  # center reference
DJ_SEG_X0: float = 0.36  # middle 40% of [0.15, 0.85] (unchanged relative to floor width)
DJ_SEG_X1: float = 0.64
DJ_BOOTH_Y: float = 0.03
AGENT_RADIUS: float = 0.027  # base radius on 0.70 floor (~0.0386 plate units -> ~1.29x bigger on screen)
ARCHETYPE_SIZE_MULT: dict[str, float] = {
    "diehard": 1.20,
    "between": 1.00,
    "dancer": 0.85,
}
SIZE_JITTER_FRAC: float = 0.10
PHYSICS_HZ: int = 30
TICK_SECONDS: float = 3.0
TICK_SECONDS_FALLBACK: float = 4.0

# Rings by distance to DJ segment, rescaled to 0.70 floor depth (rings 1..5)
RING_EDGES: tuple[float, ...] = (0.14, 0.28, 0.42, 0.56)
RING_GAINS: tuple[float, ...] = (1.0, 0.90, 0.75, 0.60, 0.45)

# Movement & collisions (rescaled for compact 0.70 x 0.70 floor)
MAX_SPEED: float = 0.095  # units/s * (0.4 + 0.6 * E) * section_speed
COLLISION_ITERATIONS: int = 4
DENSITY_RADIUS_MULT: float = 2.5
DENSITY_CAP: int = 8
MAX_DIST_TO_DJ: float = 0.73082  # sqrt(0.21^2 + 0.70^2) from corner (0.15, 0.70) to DJ segment [0.36, 0.64]

# Energy constants (per second; Phase E2: front music charge feeds crowd, pushing/long dancing tires)
ENERGY_A: float = 0.010  # front crush (halved from 0.024)
ENERGY_B: float = 0.016  # push drain (pushing tires you away from front drops)
ENERGY_D: float = 0.008  # sustained dance drain
ENERGY_C: float = 0.110  # rest recovery
ENERGY_G: float = 0.048  # music charge near DJ: g * v * local_music * w (calibrated for real MP3 w~1.4-1.9)
EXHAUST_TRIGGER_E: float = 0.05  # rare exhaustion trigger
EXHAUST_RECOVER_E: float = 0.25  # recover & rejoin floor

# Mass / pushiness contrast constants (mass = MASS_BASE + MASS_PUSH_SCALE * pushiness^2 * E)
MASS_BASE: float = 0.08
MASS_PUSH_SCALE: float = 2.5

# Reward, happiness & hype/energy contagion
REWARD_M: float = 0.60  # shared enjoyment of loud music near front
U_NORM_DIVISOR: float = 1.15  # calibrates typical H to ~0.1-0.85 on real MP3 drops/breakdowns
HAPPINESS_TAU: float = 10.0  # seconds
CONTAGION_KAPPA: float = 0.25  # H contagion per second
ENERGY_CONTAGION_KAPPA: float = 0.10  # E contagion per second
CONTAGION_RADIUS_MULT: float = 3.0

# House ear section weights
SECTION_WEIGHTS: dict[str, float] = {
    "drop": 3.0,
    "build": 1.5,
    "groove": 1.0,
    "intro": 0.5,
    "breakdown": 0.5,
    "outro": 0.5,
    "other": 0.5,
}

# Audio capture & clips
AUDIO_SR: int = 16000
PCM_CHUNK_MS: int = 250
BUFFER_SECONDS: int = 30
RING_CLIP_SECONDS: float = 8.0
HOUSE_CLIP_SECONDS: float = 15.0

# Negotiation & concurrency
MAX_ASKS_PER_TICK: int = 8
MAX_ASKS_FALLBACK: int = 4
API_CONCURRENCY: int = 8
API_TIMEOUT_SECONDS: float = 8.0

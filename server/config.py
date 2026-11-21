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

# Floor & agents
N_AGENTS: int = 100
N_LANES: int = 10
DJ_POS: tuple[float, float] = (0.5, 0.0)
DJ_BOOTH_Y: float = 0.03
AGENT_RADIUS: float = 0.036
PHYSICS_HZ: int = 30
TICK_SECONDS: float = 3.0
TICK_SECONDS_FALLBACK: float = 4.0

# Rings by distance to DJ (edges at 0.2, 0.4, 0.6, 0.8 -> rings 1..5)
RING_EDGES: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)

# Movement & collisions (tuned in Phase C for visible drop surge, jostle, and breakdown loosen)
MAX_SPEED: float = 0.11  # units/s * (0.4 + 0.6 * E)
COLLISION_ITERATIONS: int = 4
DENSITY_RADIUS_MULT: float = 2.5
DENSITY_CAP: int = 8
MAX_DIST_TO_DJ: float = 1.118  # sqrt(0.5^2 + 1.0^2)

# Energy constants (per second; tuned in Phase C/C2 for bigger AGENT_RADIUS)
ENERGY_A: float = 0.024  # front crush
ENERGY_B: float = 0.007  # push
ENERGY_D: float = 0.004  # dance
ENERGY_C: float = 0.135  # rest recovery
EXHAUST_RECOVER_E: float = 0.3

# Mass / pushiness contrast constants (mass = MASS_BASE + MASS_PUSH_SCALE * pushiness^2 * E)
MASS_BASE: float = 0.08
MASS_PUSH_SCALE: float = 2.5

# Happiness & contagion
HAPPINESS_TAU: float = 10.0  # seconds
CONTAGION_KAPPA: float = 0.15  # per second
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

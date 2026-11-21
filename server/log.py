"""Run logging: meta.json, ticks.jsonl, asks.jsonl inside runs/<timestamp>_<condition>_<seed>/."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from server import config
from server.state import CrowdState, ring_for_agent

logger = logging.getLogger("agent_crowd.log")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(config.ROOT_DIR),
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


class RunLogger:
    """Writes structured run metadata and JSONL logs alongside session.wav."""

    def __init__(self) -> None:
        self.run_dir: Path | None = None
        self._ticks_fp = None
        self._asks_fp = None

    def open(self, run_dir: Path, crowd: CrowdState) -> Path:
        self.close()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        meta = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "condition": crowd.condition,
            "seed": crowd.seed,
            "source": crowd.source,
            "source_label": crowd.source_label,
            "git_sha": _git_sha(),
            "models": {
                "decisions": config.MODEL_DECISIONS,
                "hydrogen": config.MODEL_HYDROGEN,
                "neon": config.MODEL_NEON,
            },
            "constants": {
                "N_AGENTS": config.N_AGENTS,
                "AGENT_RADIUS": config.AGENT_RADIUS,
                "PHYSICS_HZ": config.PHYSICS_HZ,
                "TICK_SECONDS": config.TICK_SECONDS,
                "MAX_SPEED": config.MAX_SPEED,
                "ENERGY_A": config.ENERGY_A,
                "ENERGY_B": config.ENERGY_B,
                "ENERGY_C": config.ENERGY_C,
                "ENERGY_D": config.ENERGY_D,
                "ENERGY_G": config.ENERGY_G,
                "EXHAUST_TRIGGER_E": config.EXHAUST_TRIGGER_E,
                "EXHAUST_RECOVER_E": config.EXHAUST_RECOVER_E,
                "REWARD_M": config.REWARD_M,
                "U_NORM_DIVISOR": config.U_NORM_DIVISOR,
                "MASS_BASE": config.MASS_BASE,
                "MASS_PUSH_SCALE": config.MASS_PUSH_SCALE,
                "HAPPINESS_TAU": config.HAPPINESS_TAU,
                "CONTAGION_KAPPA": config.CONTAGION_KAPPA,
                "ENERGY_CONTAGION_KAPPA": config.ENERGY_CONTAGION_KAPPA,
                "DJ_SEG": [config.DJ_SEG_X0, config.DJ_SEG_X1],
                "RING_GAINS": list(config.RING_GAINS),
                "RING_CLIP_SECONDS": config.RING_CLIP_SECONDS,
                "HOUSE_CLIP_SECONDS": config.HOUSE_CLIP_SECONDS,
                "MAX_ASKS_PER_TICK": config.MAX_ASKS_PER_TICK,
            },
            "agents": [
                {
                    "id": a.id,
                    "lane": a.lane,
                    "archetype": a.archetype,
                    "k": round(a.k, 4),
                    "s": round(a.s, 4),
                    "size_mult": round(a.size_mult, 4),
                }
                for a in crowd.agents
            ],
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._ticks_fp = (run_dir / "ticks.jsonl").open("a", encoding="utf-8")
        self._asks_fp = (run_dir / "asks.jsonl").open("a", encoding="utf-8")
        logger.info("Opened run logs in %s", run_dir)
        return run_dir

    def log_tick(
        self,
        *,
        crowd: CrowdState,
        house: dict[str, Any],
        rings: dict[str, Any],
        asks: list[dict[str, Any]],
        answers: list[dict[str, Any]],
        api: dict[str, Any],
    ) -> None:
        if self._ticks_fp is None:
            return
        row = {
            "t": round(crowd.t, 3),
            "tick": crowd.tick,
            "condition": crowd.condition,
            "house": house,
            "dsp": crowd.dsp,
            "rings": rings,
            "agents": [
                {
                    "id": a.id,
                    "lane": a.lane,
                    "archetype": a.archetype,
                    "x": round(a.x, 4),
                    "y": round(a.y, 4),
                    "E": round(a.E, 4),
                    "v": round(a.v, 4),
                    "local_music": round(a.local_music, 4),
                    "radius": round(a.radius, 4),
                    "rho": round(a.rho, 4),
                    "H": round(a.H, 4),
                    "u": round(a.u, 4),
                    "action": a.action,
                    "push": round(a.pushiness, 3),
                    "action_probs": a.action_probs,
                    "ring": ring_for_agent(a),
                    "exhausted": a.exhausted,
                }
                for a in crowd.agents
            ],
            "asks": asks,
            "answers": answers,
            "api": api,
        }
        self._ticks_fp.write(json.dumps(row) + "\n")
        self._ticks_fp.flush()

    def log_ask(self, record: dict[str, Any]) -> None:
        if self._asks_fp is None:
            return
        self._asks_fp.write(json.dumps(record) + "\n")
        self._asks_fp.flush()

    def close(self) -> None:
        for attr in ("_ticks_fp", "_asks_fp"):
            fp = getattr(self, attr, None)
            if fp is not None:
                try:
                    fp.flush()
                    fp.close()
                except Exception:
                    pass
                setattr(self, attr, None)

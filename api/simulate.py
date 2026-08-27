"""Vercel Function powering the hosted EW smart-scan demonstration."""

from __future__ import annotations

import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Mapping
from urllib.parse import parse_qs, urlparse

from baselines import Scheduler
from experiments.run_comparison import build_environment, policy_factories
from metrics import average_metrics, evaluate


MAX_BANDS = 20
MAX_EPISODES = 12
MAX_HORIZON = 360


def _integer(values: Mapping[str, list[str]], name: str, default: int, low: int, high: int) -> int:
    try:
        result = int(values.get(name, [str(default)])[0])
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _number(values: Mapping[str, list[str]], name: str, default: float, low: float, high: float) -> float:
    try:
        result = float(values.get(name, [str(default)])[0])
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _finite(value: Any) -> Any:
    """Convert NaN values into JSON-friendly nulls recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def simulate(values: Mapping[str, list[str]]) -> Dict[str, Any]:
    """Run a bounded, seeded comparison of one hosted policy."""
    bands = _integer(values, "bands", 8, 2, MAX_BANDS)
    horizon = _integer(values, "horizon", 160, 24, MAX_HORIZON)
    episodes = _integer(values, "episodes", 6, 1, MAX_EPISODES)
    seed = _integer(values, "seed", 500, 0, 2_000_000_000)
    agility = _number(values, "agility", 0.85, 0.0, 1.0)
    method = values.get("method", ["discounted_ucb"])[0]
    factories: Dict[str, Callable[[], Scheduler]] = policy_factories(bands, seed=seed + 7)
    if method not in factories:
        raise ValueError(f"method must be one of: {', '.join(factories)}")
    environment = build_environment(num_bands=bands, agility=agility, seed=seed)
    result = average_metrics(evaluate(environment, factories[method], episodes=episodes, horizon=horizon, seed=seed + 100))
    return _finite(
        {
            "method": method,
            "scenario": {"bands": bands, "agility": agility, "episodes": episodes, "horizon": horizon, "seed": seed},
            "metrics": result.flat(),
            "time_to_first_intercept": result.time_to_first_intercept,
            "note": "Each method receives only its own noisy single-band observations. Ground truth is used only for scoring.",
        }
    )


class handler(BaseHTTPRequestHandler):
    """Standard-library handler recognized by the Vercel Python runtime."""

    def _respond(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler convention
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler convention
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/").endswith("health"):
            self._respond(HTTPStatus.OK, {"status": "ok"})
            return
        try:
            self._respond(HTTPStatus.OK, simulate(parse_qs(parsed.query)))
        except ValueError as error:
            self._respond(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception:  # Do not leak server details to a public endpoint.
            self._respond(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Simulation failed. Please try again."})


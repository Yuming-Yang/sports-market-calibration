"""Command-line entry point.

smcal data       download (or read from cache) and build data/prematch.csv
smcal tune       choose model hyperparameters on pre-2026 data; write config/frozen.json
smcal forecast   model forecasts for every World Cup match, with frozen hyperparameters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA = Path("data")
RESULTS_DIR = Path("results")
FROZEN = Path("config/frozen.json")


def cmd_data(args: argparse.Namespace) -> int:
    from smcal import polymarket
    from smcal.dataset import build_table
    from smcal.http import get_bytes
    from smcal.prematch import WindowRule
    from smcal.results import RESULTS_URL, load_results

    cache = args.data / "cache"
    results_csv = args.data / "results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_csv.write_bytes(get_bytes(RESULTS_URL, cache, suffix=".csv"))
    results = load_results(results_csv)

    events = polymarket.series_events(polymarket.WORLD_CUP_SERIES_ID, cache)

    def fetch(token: str, start: int, end: int) -> dict:
        return polymarket.price_history(token, start, end, cache)

    table, excluded = build_table(events, results, fetch, WindowRule())
    table.to_csv(args.data / "prematch.csv", index=False)

    RESULTS_DIR.mkdir(exist_ok=True)
    ex = pd.DataFrame([e.__dict__ for e in excluded], columns=["slug", "reason"])
    ex.to_csv(RESULTS_DIR / "exclusions.csv", index=False)
    print(f"{len(table)} matches priced, {len(excluded)} excluded")
    if len(table):
        thin, et = int(table["thin"].sum()), int(table["extra_time"].sum())
        print(f"thin windows: {thin}; extra time inferred: {et}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from smcal.forecast import tune_elo, tune_goals
    from smcal.results import load_results

    results = load_results(args.data / "results.csv")
    results = results[results["date"] < "2026-01-01"].reset_index(drop=True)
    elo, goals = tune_elo(results), tune_goals(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    elo.to_csv(RESULTS_DIR / "tuning_elo.csv", index=False, float_format="%.6f")
    goals.to_csv(RESULTS_DIR / "tuning_goals.csv", index=False, float_format="%.6f")
    best_e, best_g = elo.iloc[0], goals.iloc[0]
    frozen = {
        "tuned_on": "matches before 2026-01-01; validation 2023-2025, non-friendly",
        "elo": {"k_scale": float(best_e["k_scale"]), "home_adv": float(best_e["home_adv"])},
        "goals": {"xi": float(best_g["xi"]), "ridge": float(best_g["ridge"])},
    }
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps(frozen, indent=2))
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    from smcal.forecast import world_cup_forecasts
    from smcal.models.elo import EloParams
    from smcal.models.poisson import FitConfig
    from smcal.results import load_results

    frozen = json.loads(FROZEN.read_text())
    results = load_results(args.data / "results.csv")
    table = pd.read_csv(args.data / "prematch.csv")
    fc = world_cup_forecasts(
        results, table, EloParams(**frozen["elo"]), FitConfig(**frozen["goals"])
    )
    # Model forecasts come from CC0 data only and are committed; market prices are not.
    RESULTS_DIR.mkdir(exist_ok=True)
    fc.to_csv(RESULTS_DIR / "model_forecasts.csv", index=False, float_format="%.6f")
    merged = table.merge(fc.drop(columns=["kickoff", "home", "away", "host"]), on="slug")
    merged.to_csv(args.data / "forecasts.csv", index=False)
    print(f"{len(merged)} matches forecast")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smcal", description="Sports market calibration")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("data", help="download and build data/prematch.csv")
    p.add_argument("--data", type=Path, default=DATA)
    p.set_defaults(func=cmd_data)
    for name, func, help_ in (
        ("tune", cmd_tune, "tune hyperparameters on pre-2026 data"),
        ("forecast", cmd_forecast, "model forecasts for the World Cup matches"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--data", type=Path, default=DATA)
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)

"""Command-line entry point.

smcal data       download (or read from cache) and build data/prematch.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA = Path("data")
RESULTS_DIR = Path("results")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smcal", description="Sports market calibration")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("data", help="download and build data/prematch.csv")
    p.add_argument("--data", type=Path, default=DATA)
    p.set_defaults(func=cmd_data)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)

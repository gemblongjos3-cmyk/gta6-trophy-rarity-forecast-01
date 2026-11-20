"""Forecast GTA VI trophy earn rates for make.fun's gta6-trophy-rarity-forecast-01.

The game launched at 2026-11-19T00:00:00Z; every read the house recorder took
of the Exophase trophy page carries a slot, a players-tracked count and the
earn rate of every trophy it listed. Exophase only re-crawls the page now and
then, so a run of reads showing the same counts is one stale snapshot rather
than fresh evidence. For each trophy the program fits a straight line to the
logit of its rate against ln(1 + hours since launch), weighting recent reads
more heavily, then reads that line at the case's target time. The spread
around the fitted value combines the horizon's expected drift with the fit's
own residual scatter, so a trophy whose history is noisy is forecast more
widely than one that follows a clean line.

Trophies with too little history -- and cases whose whole history collapses to
a single snapshot -- keep their last observed rate and take the plain random
walk's spread.

Standard library only, deterministic, no network. Run with:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The instant every hour count is measured from.
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Standard normal quantiles for the five levels the contest scores, in order.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Reads older than this many hours before the last read are left out of the fit.
RECENT_SPAN_HOURS = 24.0
# A line needs at least this many usable points; fewer means fall back.
MIN_FIT_POINTS = 3

# Weight of the newest read, and the extra weight lost per 24 hours of age.
NEWEST_WEIGHT = 2.0
WEIGHT_DECAY_HOURS = 24.0

# How much of the combined spread the forecast actually uses.
SPREAD_SCALE = 0.55
# Daily drift of a logit rate under a plain random walk.
DAILY_DRIFT = 0.5

# Horizon length bounds, in days, that the spread is computed within.
MIN_HORIZON_DAYS = 0.10
MAX_HORIZON_DAYS = 3.0

# Rates are reported as percentages and stay inside what Exophase can print.
PCT_FLOOR = 0.01
PCT_CEIL = 99.99

# Used when a trophy never had a usable rate at all.
NO_RATE = 0.02


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp_pct(value: float) -> float:
    """Keep a percentage inside the printable Exophase range."""
    if value < PCT_FLOOR:
        return PCT_FLOOR
    if value > PCT_CEIL:
        return PCT_CEIL
    return value


def logit_of(pct: float) -> float:
    """Log-odds of a percentage, clamped away from 0 and 100 first."""
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of(log_odds: float) -> float:
    """Inverse of logit_of: a percentage from log-odds, guarded against overflow."""
    if log_odds >= 0.0:
        return 100.0 / (1.0 + math.exp(-log_odds))
    e = math.exp(log_odds)
    return 100.0 * e / (1.0 + e)


def round_four(value: float) -> float | int:
    """Round to 4 decimals, halves upward, writing whole numbers without a point."""
    shifted = value * 10000.0
    base = math.floor(shifted)
    if shifted - base >= 0.5:
        base += 1
    out = base / 10000.0
    if out == int(out):
        return int(out)
    return out


def to_instant(text: str) -> datetime:
    """Parse an ISO 8601 instant, treating a trailing Z as UTC."""
    stamp = text.strip()
    if stamp.endswith(("Z", "z")):
        stamp = stamp[:-1] + "+00:00"
    moment = datetime.fromisoformat(stamp)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def hours_from_launch(text: str) -> float:
    """Hours between launch and a slot, with anything before launch pinned to 0."""
    delta = (to_instant(text) - LAUNCH).total_seconds() / 3600.0
    return delta if delta > 0.0 else 0.0


def horizon_in_days(minutes: float) -> float:
    """The case horizon in days, held between the spread's bounds."""
    days = float(minutes) / 1440.0
    if days < MIN_HORIZON_DAYS:
        return MIN_HORIZON_DAYS
    if days > MAX_HORIZON_DAYS:
        return MAX_HORIZON_DAYS
    return days


# --------------------------------------------------------------------------
# reading the history
# --------------------------------------------------------------------------

def snapshot_key(read: dict) -> tuple:
    """Identity of a read's content: tracked count plus every rate, nulls kept."""
    rates = read.get("rates") or []
    return (read.get("players_tracked"), tuple(rates))


def collapse_snapshots(history: list[dict]) -> list[dict]:
    """Drop repeated snapshots of one Exophase refresh, keeping its first read.

    Reads that repeat the previous read's tracked count and rates are the same
    page before the next crawl, so only the earliest of them tells us when that
    content became visible.
    """
    kept: list[dict] = []
    previous = None
    for read in history:
        key = snapshot_key(read)
        if key != previous:
            kept.append(read)
            previous = key
    return kept


def recent_points(read: dict, trophy_index: int, last_hours: float) -> tuple[float, float] | None:
    """One (x, y) pair for the fit: log-time and log-odds, or None without a rate."""
    hours = hours_from_launch(read["slot"])
    if last_hours - hours > RECENT_SPAN_HOURS:
        return None
    rates = read.get("rates") or []
    if trophy_index >= len(rates):
        return None
    rate = rates[trophy_index]
    if rate is None:
        return None
    return math.log(1.0 + hours), logit_of(float(rate))


def fit_line(points: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    """Weighted least squares of y on x; returns (intercept, slope, rmse) or None.

    Each point arrives as (x, y, weight).
    """
    total_w = 0.0
    sum_wx = 0.0
    sum_wy = 0.0
    for x, y, w in points:
        total_w += w
        sum_wx += w * x
        sum_wy += w * y
    if total_w <= 0.0:
        return None

    mean_x = sum_wx / total_w
    mean_y = sum_wy / total_w

    sxx = 0.0
    sxy = 0.0
    for x, y, w in points:
        dx = x - mean_x
        sxx += w * dx * dx
        sxy += w * dx * (y - mean_y)

    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = mean_y - slope * mean_x

    sse = 0.0
    for x, y, w in points:
        miss = y - (intercept + slope * x)
        sse += w * miss * miss
    rmse = math.sqrt(sse / total_w)
    return intercept, slope, rmse


# --------------------------------------------------------------------------
# forecast construction
# --------------------------------------------------------------------------

def quantile_row(median_pct: float, sigma: float) -> list[float | int]:
    """Turn a median rate and a logit spread into the five reported quantiles."""
    center = logit_of(median_pct)
    row: list[float | int] = []
    previous = 0.0
    for z in Z_SCORES:
        # Per the contract: sigmoid, round half-up to 4 decimals, clamp, then
        # force the row to stay non-decreasing.
        rounded = float(round_four(pct_of(center + z * sigma)))
        bounded = clamp_pct(rounded)
        if bounded < previous:
            bounded = previous
        row.append(round_four(bounded))
        previous = float(row[-1])
    return row


def walk_sigma(horizon_days: float) -> float:
    """Spread of a rate that only random-walks away from where it was last seen."""
    return DAILY_DRIFT * math.sqrt(horizon_days)


def persistence_row(rate: float | None, horizon_days: float) -> list[float | int]:
    """The fallback: hold last seen rate, widen like a random walk."""
    median = float(rate) if rate is not None else NO_RATE
    return quantile_row(median, walk_sigma(horizon_days))


def forecast_case(case: dict) -> dict:
    """Build the output object for one case."""
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    horizon_days = horizon_in_days(case.get("horizon_minutes", 0))

    last_read = history[-1] if history else {}
    last_rates = last_read.get("rates") or []
    last_hours = hours_from_launch(last_read["slot"]) if history else 0.0

    distinct = collapse_snapshots(history)
    # A single surviving snapshot says nothing about a trend.
    can_trend = len(distinct) > 1
    if can_trend:
        # The fit's x axis is ln(1 + hours), so the target enters it the same way.
        target_x = math.log(1.0 + hours_from_launch(case["target"]))
    else:
        target_x = math.log(1.0 + last_hours)

    rows = []
    for index, trophy in enumerate(trophies):
        last_rate = last_rates[index] if index < len(last_rates) else None

        if can_trend:
            collected = []
            for read in distinct:
                point = recent_points(read, index, last_hours)
                if point is None:
                    continue
                x, y = point
                # The newest read weighs NEWEST_WEIGHT; a read 24h older weighs 1.0.
                hours = hours_from_launch(read["slot"])
                weight = NEWEST_WEIGHT + (hours - last_hours) / WEIGHT_DECAY_HOURS
                collected.append((x, y, weight))

            if len(collected) >= MIN_FIT_POINTS:
                fitted = fit_line(collected)
                if fitted is not None:
                    intercept, slope, residual = fitted
                    median = clamp_pct(pct_of(intercept + slope * target_x))
                    spread = SPREAD_SCALE * math.sqrt(
                        (DAILY_DRIFT * math.sqrt(horizon_days)) ** 2 + residual ** 2
                    )
                    rows.append({"id": trophy["id"], "q": quantile_row(median, spread)})
                    continue

        rows.append({"id": trophy["id"], "q": persistence_row(last_rate, horizon_days)})

    return {"trophies": rows}


# --------------------------------------------------------------------------
# driving the run
# --------------------------------------------------------------------------

def pick_dir(variable: str, fallback: str) -> Path:
    """The input or output directory the environment asks for."""
    return Path(os.environ.get(variable, fallback))


def main(argv: list[str]) -> int:
    input_dir = pick_dir("MAKEFUN_INPUT_DIR", "/input")
    output_dir = pick_dir("MAKEFUN_OUTPUT_DIR", "/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_path in sorted(input_dir.glob("*.in")):
        with case_path.open("r", encoding="utf-8") as handle:
            case = json.load(handle)
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":"), ensure_ascii=False)
        (output_dir / (case_path.stem + ".out")).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

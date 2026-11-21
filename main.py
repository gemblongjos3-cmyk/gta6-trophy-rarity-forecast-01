"""gta6-trophy-rarity-forecast-01 — a recency-weighted logit-linear log-time trophy earn-rate forecaster.

The contest hands the program one case: the trophy list of Grand Theft Auto VI on PS5 and every
Exophase read the house made before the case's checkpoint, and asks for the five quantiles
(5%, 25%, 50%, 75%, 95%) of each trophy's earn rate at the case's target.

The model, one trophy at a time:

  * The reads whose slot falls in the last 24 hours of the history become points
    x = ln(1 + hours since the 2026-11-19T00:00:00Z launch) and y = logit(rate in percent).
  * A straight line y = intercept + slope * x is fitted to those points by least squares weighted
    by recency: a point's weight runs from 1.0 (24 hours back) to 2.0 (the newest read).
  * The line is read at the target's hours, the logistic turns it into a median rate, and the
    spread in logit units is 0.55 * sqrt((0.5 * sqrt(horizon_days))^2 + residual_RMSE^2).
  * The five quantiles are the logistic at the median, shifted by z * sigma for the standard
    normal's quantiles [-1.645, -0.674, 0, 0.674, 1.645].

Stale Exophase refreshes are collapsed first: consecutive reads with the same players tracked and
the same rates are one page state, and only its first occurrence is kept, because that read is the
nearest in time to the moment Exophase actually refreshed its numbers. A history that holds one
distinct read, or a trophy with fewer than three points in the window, has no trend to read, so the
forecast falls back to persistence: the rate of the last read, with the house baseline's
random-walk spread.

The program reads every <id>.in in MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to
MAKEFUN_OUTPUT_DIR (else /output). Standard library only, no network, no clock and no randomness:
the same case always yields the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The launch the case's clocks are counted from.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The standard normal's quantile at each of the contest's levels (0.05, 0.25, 0.5, 0.75, 0.95).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The fitted trend's spread is scaled by this: the backtest's compromise between the mean-optimal
# 0.45-0.50 and the more robust 0.60-0.70, because the real week has fatter tails than the
# simulator that chose it.
SIGMA_SCALE = 0.55
# The house baseline's random walk, in logit units per square-root day.
RANDOM_WALK_SIGMA = 0.5
# The fit looks at the last day of reads.
WINDOW_HOURS = 24.0
# Fewer points than this and the trend is not trusted; the trophy falls back to persistence.
MIN_POINTS = 3
# The horizon the spread is built on, in days.
HORIZON_MIN_DAYS = 0.10
HORIZON_MAX_DAYS = 3.0
# The page prints rates to 2 decimals, and 0% and 100% have no logit.
MIN_PCT = 0.01
MAX_PCT = 99.99
# A trophy the last read lacked stands at this rate under persistence.
NULL_RATE = 0.02


def clamp(value: float, low: float = MIN_PCT, high: float = MAX_PCT) -> float:
    """A rate clamped into the range the logit is defined on."""
    return min(high, max(low, value))


def parse_timestamp(text: str) -> datetime:
    """An ISO 8601 stamp of the case file as an aware UTC datetime.

    A stamp that does not parse is read as the launch itself rather than raised: one odd field
    must not take the whole case down, because a case that crashes is scored as invalid.
    """
    stamp = str(text).strip()
    if stamp.endswith(("Z", "z")):
        stamp = stamp[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return LAUNCH
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def as_rate(value) -> float | None:
    """A read's rate as a number, or None when the read carried none worth using."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(value):
        return None
    return float(value)


def hours_since_launch(stamp: str) -> float:
    """The hours from the launch to a stamp; a stamp before the launch counts as zero."""
    return max(0.0, (parse_timestamp(stamp) - LAUNCH).total_seconds() / 3600.0)


def logit(percent: float) -> float:
    """logit of a rate in percent."""
    rate = clamp(percent)
    return math.log(rate / (100.0 - rate))


def sigmoid_percent(value: float) -> float:
    """The logistic of a logit, back in percent, computed without overflow."""
    if value >= 0.0:
        return 100.0 / (1.0 + math.exp(-value))
    grown = math.exp(value)
    return 100.0 * grown / (1.0 + grown)


def round4(value: float) -> float | int:
    """A number rounded to 4 decimals, halves rounding up, a whole number staying whole."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 10000.0
    return int(rounded) if rounded.is_integer() else rounded


def quantiles(center_logit: float, sigma: float) -> list[float | int]:
    """The five quantiles around a median given in logit units, with the spread in logit units."""
    row = []
    for z in Z:
        value = clamp(round4(sigmoid_percent(center_logit + z * sigma)))
        row.append(value)
    # A rate that the rounding pushed out of order is put back in order.
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def dedupe_reads(history: list[dict]) -> list[dict]:
    """Collapse the stale refreshes: consecutive reads showing the same page state are one read.

    Exophase refreshes a game's counts on its own schedule, so a stretch of reads repeats one
    page state. The first read of the stretch is the one nearest to the refresh itself, so it
    dates the numbers best; the rest of the stretch says nothing new.
    """
    kept: list[dict] = []
    previous_state = None
    for read in history:
        rates = read.get("rates") or []
        state = (read.get("players_tracked"), tuple(rates))
        if kept and state == previous_state:
            continue
        kept.append(read)
        previous_state = state
    return kept


def horizon_days(case: dict) -> float:
    """The case's horizon in days, floored and capped the way the spread wants it."""
    minutes = case.get("horizon_minutes", 0) or 0
    return min(HORIZON_MAX_DAYS, max(HORIZON_MIN_DAYS, minutes / 1440.0))


def target_hours(case: dict) -> float:
    """The hours from the launch to the case's target."""
    target = case.get("target")
    if target:
        return hours_since_launch(target)
    return hours_since_launch(case["at"]) + (case.get("horizon_minutes", 0) or 0) / 60.0


def window_points(reads: list[dict], index: int) -> list[tuple[float, float]]:
    """The points of one trophy in the last day of reads: (hours, logit rate) pairs.

    A read outside the last 24 hours before the last read's slot is left out, and a read that
    lacked the trophy contributes nothing. The hours are kept raw here: the fit turns them into
    the log-time of the line, while the recency weight is counted in hours.
    """
    last_hours = hours_since_launch(reads[-1]["slot"])
    cutoff = last_hours - WINDOW_HOURS
    points = []
    for read in reads:
        hours = hours_since_launch(read["slot"])
        if hours < cutoff:
            continue
        rates = read.get("rates") or []
        if index >= len(rates):
            continue
        rate = as_rate(rates[index])
        if rate is None:
            continue
        points.append((hours, logit(rate)))
    return points


def weighted_line(points: list[tuple[float, float]], last_hours: float):
    """The recency-weighted least squares fit of a straight line, with its weighted RMSE.

    The line is fitted on the log-time x = ln(1 + hours since the launch), against the logit of
    the rate, so that a rate that grows quickly early and slowly later still reads as a line.
    A point's weight is 2.0 at the newest read and 1.0 a day earlier. The residual error is
    weighted the same way, so it measures the error the fit itself was judged by.
    """
    weights = [2.0 + (hours - last_hours) / WINDOW_HOURS for hours, _ in points]
    total = sum(weights)
    xs = [math.log(1.0 + hours) for hours, _ in points]
    ys = [y for _, y in points]
    mean_x = sum(w * x for w, x in zip(weights, xs)) / total
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total
    spread_x = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    if spread_x <= 0.0:
        # Every point sits on the same slot: there is no slope to read.
        return None
    covariance = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys))
    slope = covariance / spread_x
    intercept = mean_y - slope * mean_x
    residual = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(weights, xs, ys)) / total
    )
    return intercept, slope, residual


def forecast_case(case: dict) -> dict:
    """The forecast of one case: five quantiles for every trophy of the case."""
    history = case.get("history") or []
    trophies = case.get("trophies") or []

    days = horizon_days(case)
    aim = target_hours(case)

    # One distinct read means one page state: the whole case is flat and persistence is all there is.
    reads = dedupe_reads(history)
    flat = len(reads) < 2

    last_rates = (reads[-1].get("rates") or []) if reads else []
    rows = []
    for index, trophy in enumerate(trophies):
        rates = last_rates if index < len(last_rates) else []
        last_rate = as_rate(rates[index]) if index < len(rates) else None

        median = None
        sigma = RANDOM_WALK_SIGMA * math.sqrt(days)
        if not flat:
            # Anything that goes wrong with one trophy's fit costs that trophy its trend,
            # not the whole case: it falls back to persistence below.
            try:
                points = window_points(reads, index)
                if len(points) >= MIN_POINTS:
                    fit = weighted_line(points, hours_since_launch(reads[-1]["slot"]))
                    if fit is not None:
                        intercept, slope, residual = fit
                        at_target = intercept + slope * math.log(1.0 + aim)
                        median = clamp(sigmoid_percent(at_target))
                        sigma = SIGMA_SCALE * math.sqrt(
                            (RANDOM_WALK_SIGMA * math.sqrt(days)) ** 2 + residual**2
                        )
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                median = None

        if median is None:
            # Persistence: the rate of the last read carries, with the random walk's spread.
            median = clamp(last_rate if last_rate is not None else NULL_RATE)

        trophy_id = trophy.get("id") if isinstance(trophy, dict) else None
        if trophy_id is None:
            continue
        rows.append({"id": trophy_id, "q": quantiles(logit(median), sigma)})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            # A case file that cannot be read is left without a forecast; the other cases of the
            # run still deserve their own.
            print(f"{path.name}: cannot be read ({exc})", file=sys.stderr)
            continue
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

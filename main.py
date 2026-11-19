"""GTA VI trophy rarity forecaster (Python 3.13, standard library only).

Forecasts, for each of the game's 55 PS5 trophies, five quantiles of its
Exophase earn rate (percent of tracked players holding it) at the case's
target, from the house's recorded reads of
https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/.

Model — earners/players decomposition blended with a logit log-time trend:
  * A trophy's rate is earners / players_tracked. Both move: earners grow
    as players progress, and the tracked set grows as more players sync,
    which dilutes every rate (launch night: 638 tracked players, then a
    refresh to 3,868, and every rate but the one at 100% fell).
  * Per trophy, medians are fit on log(1 + hours since release),
    with recency weights (24 h half-life, so the launch surge's fast
    early growth does not dominate the slope when growth later slows):
      (A) log earners and log players_tracked, each least-squares linear,
          then rate = 100 * earners / players at the target;
      (B) logit(rate) least-squares linear in the same clock, read at the
          target (the house curve baseline's shape, over the whole history);
      (C) the same logit trend over the last 24 h of reads only, which
          is the honest slope when growth slows late.
    The median is the mean of the available logits in (A)–(C).
  * When the history is flat (no refresh seen yet — the page frozen since
    the first read), there is no trend to fit, so a refresh is anticipated
    instead: players grow as (1+h)^0.95 and earners as (1+h)^0.80 from the
    first read (the launch-night refresh's own aggregate: tracked 638 to
    3,868, trophies earned 943 to 4,276), which lowers most rates and
    leaves a 100% rate at 100%.
  * Spread on the logit scale combines the trend fit's residual spread
    (prediction-interval widened), a random-walk term in sqrt(days), and
    a refresh-lag floor; it is wide when the history is flat.

Run:  python3 -P -s /bot/main.py   (or, outside the grader,
        MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py)
Reads every <id>.in in the input directory and writes <id>.out:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
Deterministic: no randomness, no network, no wall-clock use.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a module shipped beside this file (none is needed here).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Release day begins 2026-11-19T00:00:00Z; the trend clock is log(1 + h).
T0 = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Power-law exponents for the flat-history refresh anticipation (see module
# docstring): players tracked and earners, from the first read.
N_EXP = 0.95
E_EXP = 0.80
# Flat-history spread on the logit scale (a refresh may or may not land).
FLAT_SIGMA = 0.65
# Random-walk spread per sqrt(day), and the refresh-lag floor, logit units.
RW_SIGMA = 0.35
REFRESH_FLOOR = 0.25


def _hours(iso: str) -> float:
    """Hours from T0 to an ISO-8601 UTC timestamp."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return (dt - T0).total_seconds() / 3600.0


def _x(hours: float) -> float:
    """Trend clock: log(1 + hours since release), floored so pre-release
    reads (up to ~10 minutes early) stay defined."""
    return math.log(max(1.0 + hours, 0.05))


def clamp_pct(p: float) -> float:
    return min(99.99, max(0.01, p))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def ols(xs, ys, ws=None):
    """Least squares y ~ a + b*x, optionally weighted (ws, same length as
    xs). Returns (a, b, sse, n, mx, sxx), or None when the fit is
    degenerate (fewer than 2 points or no x spread)."""
    n = len(xs)
    if n < 2:
        return None
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    return (a, b, sse, n, mx, sxx)


def forecast_trophy(series, horizon_days, x_target, target_h):
    """series: [(x, hours, players, rate)] distinct reads, oldest first.
    target_h: the case target in hours since release.
    Returns (median_pct, sigma_logit)."""
    last_rate = series[-1][3]
    last_lg = logit(last_rate)

    if len(series) < 3:
        # Flat history: no trend to fit — anticipate the next refresh.
        h_first = series[0][1]
        grow_n = ((1.0 + target_h) /
                  max(1.0 + h_first, 0.05)) ** N_EXP
        grow_e = ((1.0 + target_h) /
                  max(1.0 + h_first, 0.05)) ** E_EXP
        grow_n = min(grow_n, 8.0)
        grow_e = min(grow_e, 8.0)
        if last_rate >= 99.99:
            median = 99.99
        else:
            median = clamp_pct(last_rate * grow_e / grow_n)
        return median, FLAT_SIGMA

    xs = [s[0] for s in series]
    # Recency weights (24 h half-life): the launch surge's fast early
    # growth must not dominate the slope when growth later slows (the
    # tracked set's growth decays; a fit over the whole history would
    # extrapolate the surge and overstate dilution).
    h_last = series[-1][1]
    ws = [0.5 ** ((h_last - s[1]) / 24.0) for s in series]
    # (A) earners / players decomposition in log space.
    log_e = [math.log(max(s[3], 0.005) * s[2] / 100.0) for s in series]
    log_n = [math.log(s[2]) for s in series]
    fit_e = ols(xs, log_e, ws)
    fit_n = ols(xs, log_n, ws)
    rate_a = None
    if fit_e is not None and fit_n is not None:
        e_star = math.exp(fit_e[0] + fit_e[1] * x_target)
        n_star = math.exp(fit_n[0] + fit_n[1] * x_target)
        e_last = max(series[-1][3], 0.005) * series[-1][2] / 100.0
        n_last = series[-1][2]
        # Earners and the tracked set do not shrink (up to re-sync noise).
        e_star = max(e_star, 0.9 * e_last)
        n_star = max(n_star, n_last)
        rate_a = clamp_pct(100.0 * e_star / n_star)

    # (B) direct logit-rate trend on the same clock.
    lgs = [logit(s[3]) for s in series]
    fit_l = ols(xs, lgs, ws)
    rate_b = None
    resid_sd = None
    pred_factor = 1.0
    if fit_l is not None:
        a, b, sse, n, mx, sxx = fit_l
        rate_b = clamp_pct(inv_logit(a + b * x_target))
        dof = max(n - 2, 1)
        resid_sd = math.sqrt(sse / dof)
        pred_factor = math.sqrt(1.0 + 1.0 / n + (x_target - mx) ** 2 / sxx)
        pred_factor = min(pred_factor, 2.5)

    # (C) recent-window logit trend (last 24 h of reads): when growth
    # slows late (dilution easing), the recent slope is the honest one.
    rate_c = None
    recent = [s for s in series if s[1] >= h_last - 24.0]
    if len(recent) >= 3:
        rxs = [s[0] for s in recent]
        rys = [logit(s[3]) for s in recent]
        fit_r = ols(rxs, rys)
        if fit_r is not None:
            rate_c = clamp_pct(inv_logit(fit_r[0] + fit_r[1] * x_target))

    # Blend the medians in logit space; fall back to persistence.
    parts = []
    if rate_a is not None:
        parts.append(logit(rate_a))
    if rate_b is not None:
        parts.append(logit(rate_b))
    if rate_c is not None:
        parts.append(logit(rate_c))
    if not parts:
        median_lg = last_lg
    else:
        median_lg = sum(parts) / len(parts)
    # Never extrapolate wildly away from the last read.
    median_lg = min(max(median_lg, last_lg - 3.5), last_lg + 3.5)
    median = clamp_pct(inv_logit(median_lg))

    # Spread: fit residual (prediction widened) + random walk + refresh lag.
    s_res = resid_sd if resid_sd is not None else 0.30
    s_res = min(max(s_res, 0.12), 0.90)
    sigma = math.sqrt((s_res * pred_factor) ** 2
                      + (RW_SIGMA ** 2) * horizon_days
                      + REFRESH_FLOOR ** 2)
    sigma = min(max(sigma, 0.25), 1.6)
    return median, sigma


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target_h = _hours(case["target"])
    x_target = _x(target_h)

    # Distinct reads: collapse consecutive identical pages (Exophase
    # refreshes hours apart; identical reads carry no new information but
    # would overweight the frozen stretch in a fit).
    states = []
    for read in history:
        rates = read["rates"]
        n = read["players_tracked"]
        key = (n, tuple(rates))
        if states and states[-1][0] == key:
            continue
        states.append((key, read))
    reads = [r for _, r in states]

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        series = []
        for read in reads:
            rate = read["rates"][i]
            if rate is None:
                continue
            series.append((_x(_hours(read["slot"])), _hours(read["slot"]),
                           read["players_tracked"], float(rate)))
        if not series:
            # A trophy no read ever listed: fall back to a wide 1%.
            rows.append({"id": trophy["id"],
                         "q": [round4(v) for v in _quantiles(1.0, 1.2)]})
            continue
        median, sigma = forecast_trophy(series, horizon_days, x_target,
                                        target_h)
        rows.append({"id": trophy["id"],
                     "q": [round4(v) for v in _quantiles(median, sigma)]})
    return {"trophies": rows}


def _quantiles(median_pct: float, sigma: float):
    lg = logit(median_pct)
    out = []
    prev = -1.0
    for z in Z:
        v = clamp_pct(inv_logit(lg + z * sigma))
        v = max(v, prev)  # keep the row non-decreasing after rounding
        out.append(v)
        prev = v
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()

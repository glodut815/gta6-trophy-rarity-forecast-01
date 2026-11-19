"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles per trophy.

Model (written for this entry, standard library only):

  * A trophy's rate is a ratio: rate = 100 * E / N, where E is how many
    tracked players earned it and N is how many players Exophase tracks.
    Both move: players keep earning trophies while Exophase keeps tracking
    new players, and a new player has usually earned nothing yet, so every
    refresh of the page dilutes the rates. Forecasting the ratio means
    forecasting its two parts, not the ratio itself.
  * Time is measured in hours since the release day began
    (2026-11-19T00:00:00Z) and squashed with x = ln(1 + h), which damps
    growth the way launch adoption damps: fast at first, slower later.
  * History reads are collapsed to distinct states (consecutive identical
    reads carry no new information: the page only changes when Exophase
    refreshes it). On those states, log E (per trophy) and log N are each
    fitted against x by recency-weighted least squares, then read at the
    target's x. The median rate is 100 * exp(log E* - log N*).
  * When the history is flat (a single distinct state, as on launch night),
    no slope can be fitted. The prior then is one refresh before the
    target: Exophase's first refresh multiplied the tracked players about
    sixfold while the newcomers had earned almost nothing, so the median
    rate falls to roughly a fifth of the last read (new players earn a
    little: E grows as N^0.1). A trophy at 100% stays at 100%.
  * The spread, in logit units, combines the random-walk floor
    (0.5 * sqrt(horizon in days)), the fit's residual prediction error,
    and the lumpiness of refreshes (the median jump of the log rate
    between distinct states). Quantiles are the median shifted by the
    standard normal's z at each level.
  * True constraints are enforced afterwards: a later story chapter cannot
    be earned by more players than an earlier one, the final mission cannot
    exceed chapter 4, and the platinum cannot exceed 100% completion or
    the final mission. Each quantile of the later trophy is capped at the
    same quantile of its prerequisite.

Usage: python3 main.py (reads $MAKEFUN_INPUT_DIR or /input, writes
$MAKEFUN_OUTPUT_DIR or /output). Deterministic: no randomness, fixed
ordering, fixed rounding.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
RELEASE_START = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# A refresh multiplies the tracked players by about this much (first refresh
# observed on launch night: 638 -> 3,868 players). Used only when the
# history is flat and no slope can be fitted.
FLAT_REFRESH_FACTOR = 6.0
# How fast earned counts grow relative to tracked players across a refresh
# when the history is flat (newcomers earn almost nothing in their first
# hours: the first refresh's total earned trophies barely grew beyond the
# newcomers all completing the opening mission).
FLAT_EARN_EXPONENT = 0.1
# Random-walk spread of a rate's move, in logit units per square-root day.
SIGMA_PER_SQRT_DAY = 0.5

# Prerequisite chains: (earlier, later) trophy ids. The later trophy's
# earners are a subset of the earlier one's, so its rate can never exceed
# the earlier one's.
PREREQUISITES = [(3, 4), (4, 5), (5, 6), (6, 7), (7, 1), (14, 1)]


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE_START).total_seconds() / 3600.0


def x_of_hours(hours):
    """Squashed time: ln(1 + h), floored so pre-release reads stay finite."""
    return math.log(max(0.05, 1.0 + hours))


def clamp_pct(value):
    return min(99.99, max(0.01, value))


def logit_pct(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(logit_value):
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4_half_up(value):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 10000.0
    return int(rounded) if float(rounded).is_integer() else rounded


def z_for_level(level):
    """Standard normal quantile at a level, via the stdlib normal dist."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(level)


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, xbar, sxx, resid)."""
    total_w = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / total_w
    ybar = sum(w * y for w, y in zip(ws, ys)) / total_w
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return ybar, 0.0, xbar, sxx, 0.0
    slope = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys)) / sxx
    intercept = ybar - slope * xbar
    resid2 = sum(
        w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys)
    )
    dof = max(1, len(xs) - 2)
    resid = math.sqrt(resid2 / total_w * len(xs) / dof) if total_w > 0 else 0.0
    return intercept, slope, xbar, sxx, resid


def distinct_states(history):
    """Collapse consecutive reads with identical players and rates.

    The page only changes when Exophase refreshes it, so a flat stretch is
    one state. Keeps the first read of each state (the refresh moment).
    """
    states = []
    last_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != last_key:
            states.append(read)
            last_key = key
    return states


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = x_of_hours(hours_since_release(target))

    states = distinct_states(history)
    # Squashed times of the distinct states, oldest first.
    xs = [x_of_hours(hours_since_release(parse_time(s["at"]))) for s in states]
    ns = [float(s["players_tracked"]) for s in states]
    # Recency weights: 1.0 at the oldest state, 3.0 at the newest.
    if len(xs) > 1:
        span = xs[-1] - xs[0]
        weights = [1.0 + (2.0 * (x - xs[0]) / span if span > 0 else 0.0) for x in xs]
    else:
        weights = [1.0]

    # Global fit: log players-tracked against squashed time.
    log_n = [math.log(max(1.0, n)) for n in ns]
    if len(states) > 1:
        a_n, b_n, _, sxx_n, resid_n = weighted_line(xs, log_n, weights)
        log_n_star = a_n + b_n * x_target
    else:
        log_n_star = None
        resid_n = 0.0
        sxx_n = 0.0

    last = states[-1]
    last_rates = last["rates"]
    last_n = float(last["players_tracked"])

    medians = {}
    sigmas = {}
    for idx, trophy in enumerate(trophies):
        rate_last = last_rates[idx]
        if rate_last is None:
            rate_last = fallback_rate(trophies, last_rates, trophy)
        earned_last = max(0.0, rate_last) * last_n / 100.0

        if len(states) > 1:
            # Per-trophy earned counts across the distinct states.
            ys_e = []
            xs_e = []
            ws_e = []
            for j, state in enumerate(states):
                r = state["rates"][idx]
                if r is None:
                    continue
                earned = max(0.5, float(r) * float(state["players_tracked"]) / 100.0)
                ys_e.append(math.log(earned))
                xs_e.append(xs[j])
                ws_e.append(weights[j])
            if len(ys_e) > 1:
                a_e, b_e, xbar_e, sxx_e, resid_e = weighted_line(xs_e, ys_e, ws_e)
                log_e_star = a_e + b_e * x_target
                # Prediction leverage of the target point.
                leverage = (x_target - xbar_e) ** 2 / sxx_e if sxx_e > 0 else 0.0
                sigma_fit = math.sqrt(resid_e**2 + resid_n**2)
                sigma_fit *= math.sqrt(1.0 + leverage + 1.0 / max(1, len(xs_e)))
                sigma_fit = min(sigma_fit, 1.5)
            else:
                log_e_star = math.log(max(0.5, earned_last))
                sigma_fit = 0.5
            log_rate_star = log_e_star - log_n_star
            median = 100.0 * math.exp(log_rate_star)
            # Refresh lumpiness: median absolute logit jump between states.
            jumps = []
            prev = None
            for state in states:
                r = state["rates"][idx]
                if r is None:
                    continue
                lg = logit_pct(r)
                if prev is not None:
                    jumps.append(abs(lg - prev))
                prev = lg
            jumps.sort()
            jump = jumps[len(jumps) // 2] if jumps else 0.3
            jump = min(jump, 0.8)
            sigma0 = SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)
            sigma = math.sqrt(sigma0**2 + sigma_fit**2 + jump**2)
            sigma = min(2.0, max(0.6, sigma))
        else:
            # Flat history: expect one refresh before the target.
            horizon_hours = case["horizon_minutes"] / 60.0
            if horizon_hours <= 3.0:
                factor = FLAT_REFRESH_FACTOR
            else:
                factor = min(50.0, FLAT_REFRESH_FACTOR * math.sqrt(horizon_hours / 3.0))
            if rate_last >= 99.995:
                median = 100.0
            else:
                median = rate_last * (factor ** (-(1.0 - FLAT_EARN_EXPONENT)))
            sigma = 1.2

        medians[trophy["id"]] = median
        sigmas[trophy["id"]] = sigma

    # Quantiles per trophy from its median and spread.
    rows = {}
    for trophy in trophies:
        tid = trophy["id"]
        median = min(medians[tid], 99.99)
        base = logit_pct(median)
        sigma = sigmas[tid]
        qs = []
        for level in case["levels"]:
            q = inv_logit_pct(base + z_for_level(level) * sigma)
            qs.append(min(100.0, max(0.01, q)))
        qs = [round4_half_up(q) for q in qs]
        # Rounding must not break the ordering.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows[tid] = qs

    # Enforce prerequisite caps, quantile by quantile.
    for earlier, later in PREREQUISITES:
        if earlier in rows and later in rows:
            rows[later] = [min(a, b) for a, b in zip(rows[later], rows[earlier])]

    ordered = [{"id": t["id"], "q": rows[t["id"]]} for t in trophies]
    return {"trophies": ordered}


def fallback_rate(trophies, last_rates, trophy):
    """A trophy with no rate in the last read: same tier's median, else 0.5."""
    same_tier = [
        r for t, r in zip(trophies, last_rates)
        if t["tier"] == trophy["tier"] and r is not None
    ]
    if same_tier:
        same_tier.sort()
        return same_tier[len(same_tier) // 2]
    known = [r for r in last_rates if r is not None]
    if known:
        known.sort()
        return known[len(known) // 2]
    return 0.5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

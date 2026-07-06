"""Propagate the AWS burn-radius sweep results through the empirical fit.

Reads radius_sweep_results.jsonl (as produced by radius_sweep.py) and prints
every burn-radius-sensitivity number needed for the manuscript, keyed by the
placeholder tags used in gcil-manuscript/paper1-v3.md (e.g. [[US-XLO]]).
Old "Toon" rows are counted as thermal-hiroshima (numerically identical);
rows from the old bounding scheme ("overpressure", "nagasaki") are ignored.

    python propagate_aws_results.py [--results radius_sweep_results.jsonl]

The central (point) estimates use `default` (the fiducial R = 0.75 Y^0.38
fit) and are unchanged by the sweep; they are printed as sanity checks with
their expected values. The burn-radius sensitivity ranges span the four
bounding prescriptions of Section 2.1 (exponents 1/3 and 1/2, each anchored
at Hiroshima and at Nagasaki) plus the default.

Known values from the previous AWS run (reused, not re-run):
                       India    Pakistan   US
    default            1.463    7.702      23.620
    thermal-hiroshima  1.380    7.619      27.194   (as "Toon")
"""
import argparse
import json
import os

import numpy as np

BOUNDING = ["thermal-hiroshima", "thermal-nagasaki", "blast-hiroshima", "blast-nagasaki"]
ALL_PRESCRIPTIONS = ["default"] + BOUNDING
SCENARIOS = ["India", "Pakistan", "US"]
ALIASES = {"Toon": "thermal-hiroshima"}

# --- 7 fit events (from industry-destruction-response.ipynb) -------------------
# label, (x, x_lo, x_hi), (y, y_lo, y_hi)
# The model-based 1974 point (x=10, y=35) is EXCLUDED from the fit (as in the
# notebook since 2026-07-06) and used only as an out-of-sample sanity check.
events = [
    ("Katrina&Rita", (0.23, 0.12, 0.40), (1.7, 1.3, 2.1)),
    ("Kobe",         (0.58, 0.25, 0.90), (2.6, None, None)),
    ("Tohoku",       (1.8, 1.2, 2.1),    (15.3, None, None)),
    ("Ukraine",      (16, 8, 20),        (37, 30, 50)),
    ("Germany",      (17, 15, 22),       (50, 42, 59)),
    ("USSR",         (33, 30, 45),       (52, 45, 55)),
    ("Japan",        (34, 25, 40),       (72, 65, 80)),
]
KATZ_X = 10.0  # 1974 model point, out-of-sample check only

xc = np.array([e[1][0] for e in events])
yc = np.array([e[2][0] for e in events])
b, a = np.polyfit(np.log10(xc), np.log10(yc), 1)


def predict(xv):
    return 10 ** (a + b * np.log10(xv))


def _bootstrap_params(n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(events)
    bs, as_ = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        while np.unique(xc[idx]).size < 2:
            idx = rng.integers(0, n, n)
        bs[i], as_[i] = np.polyfit(np.log10(xc[idx]), np.log10(yc[idx]), 1)
    return bs, as_


BOOT_B, BOOT_A = _bootstrap_params()


def boot_ci(xq):
    xq = np.atleast_1d(np.asarray(xq, float))
    preds = 10 ** (BOOT_A[:, None] + BOOT_B[:, None] * np.log10(xq)[None, :])
    lo, hi = np.percentile(preds, [5, 95], axis=0)
    return lo, np.minimum(hi, 100)


def load_sweep(path):
    sweep = {s: {} for s in SCENARIOS}
    if not os.path.exists(path):
        raise SystemExit(f"results file not found: {path}")
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "industry_destroyed_pct" not in d:
                continue
            presc = ALIASES.get(d["prescription"], d["prescription"])
            if presc in ALL_PRESCRIPTIONS and d["scenario"] in sweep:
                sweep[d["scenario"]][presc] = d["industry_destroyed_pct"]
    return sweep


def fmt_range(values, dp):
    lo, hi = round(min(values), dp), round(max(values), dp)
    if lo == hi:
        return f"{lo:.{dp}f}"
    return f"{lo:.{dp}f}-{hi:.{dp}f}"


def check(tag, value, expected, dp=0, tol=0):
    ok = "OK" if abs(round(value, dp) - expected) <= tol else f"** EXPECTED {expected} — TEXT NEEDS UPDATING **"
    print(f"  {tag:<28}= {value:.{dp}f}   [{ok}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="radius_sweep_results.jsonl")
    args = p.parse_args()

    print(f"fit (7 observed points; 1974 model point excluded): y = {10**a:.2f} * x^{b:.3f}")
    b_lo, b_hi = np.percentile(BOOT_B, [5, 95])
    a_lo, a_hi = np.percentile(10**BOOT_A, [5, 95])
    print(f"  b = {b:.2f} [90% CI {b_lo:.2f}, {b_hi:.2f}]   (text: 0.73, CI 0.48-0.81)")
    print(f"  a = {10**a:.1f} [90% CI {a_lo:.1f}, {a_hi:.1f}]   (text: 5.4, CI 4.2-11.5)")

    sweep = load_sweep(args.results)
    missing = [(s, pr) for s in SCENARIOS for pr in ALL_PRESCRIPTIONS if pr not in sweep[s]]
    if missing:
        print("\nMISSING RUNS (finish the sweep before trusting the numbers below):")
        for s, pr in missing:
            print(f"  {s}/{pr}")

    print("\n=== Sweep table: industrial infrastructure destroyed, x (%) ===")
    header = f"{'':<20}" + "".join(f"{s:>10}" for s in SCENARIOS)
    print(header)
    for pr in ALL_PRESCRIPTIONS:
        row = f"{pr:<20}"
        for s in SCENARIOS:
            row += f"{sweep[s].get(pr, float('nan')):>10.3f}"
        print(row)

    # sanity check: burn radius does not affect targeting, and both
    # Hiroshima-anchored radii equal 2.03 km at 15 kt
    for s in ["India", "Pakistan"]:
        th, bh = sweep[s].get("thermal-hiroshima"), sweep[s].get("blast-hiroshima")
        if th is not None and bh is not None and abs(th - bh) > 1e-9:
            print(f"WARNING: {s} thermal-hiroshima != blast-hiroshima ({th} vs {bh}); "
                  "these must coincide at 15 kt — investigate before using the numbers.")

    print("\n=== Per-scenario propagation (central = default; fit-bootstrap 90% CI) ===")
    for s in SCENARIOS:
        d = sweep[s]
        if "default" not in d:
            continue
        xcen = d["default"]
        xs = [d[pr] for pr in ALL_PRESCRIPTIONS if pr in d]
        xlo, xhi = min(xs), max(xs)
        case_lo = min((pr for pr in d), key=lambda pr: d[pr])
        case_hi = max((pr for pr in d), key=lambda pr: d[pr])
        ycen = predict(xcen)
        ci_lo, ci_hi = boot_ci(xcen)
        print(f"\n{s}:")
        print(f"  x default = {xcen:.3f}%  ->  y = {ycen:.1f}%  [fit 90% CI {ci_lo[0]:.1f}, {ci_hi[0]:.1f}]")
        print(f"  x range   = {xlo:.3f}% ({case_lo}) .. {xhi:.3f}% ({case_hi})")
        print(f"  y over x range (central fit): {predict(xlo):.1f}% .. {predict(xhi):.1f}%")
        print(f"  combined plausible range [CI5 at x_min, CI95 at x_max]: "
              f"[{boot_ci(xlo)[0][0]:.1f}, {boot_ci(xhi)[1][0]:.1f}]")

    # --- Manuscript placeholder tags -------------------------------------------
    print("\n=== MANUSCRIPT NUMBERS (paste into paper1-v3.md placeholders) ===")
    for s, tag, dp in [("India", "IN", 1), ("Pakistan", "PK", 1)]:
        d = sweep[s]
        if any(pr not in d for pr in ALL_PRESCRIPTIONS):
            print(f"  {tag}-*: n/a (missing runs)")
            continue
        xs = [d[pr] for pr in ALL_PRESCRIPTIONS]
        xnag = [d["thermal-nagasaki"], d["blast-nagasaki"]]
        print(f"  [[{tag}-XLO]]  = {min(xs):.{dp}f}     (low end of x span; upper end is the default value)")
        print(f"  [[{tag}-XNAG]] = {fmt_range(xnag, dp)}     (x under the two Nagasaki-anchored cases)")
        print(f"  [[{tag}-YNAG]] = {fmt_range([predict(x) for x in xnag], 0)}     (central-fit y under those cases)")
        print(f"  [[{tag}-CLO]]  = {boot_ci(min(xs))[0][0]:.0f}     (low end of combined plausible range)")

    d = sweep["US"]
    if all(pr in d for pr in ALL_PRESCRIPTIONS):
        xs = [d[pr] for pr in ALL_PRESCRIPTIONS]
        print(f"  [[US-XLO]]  = {min(xs):.0f}     (low end of x span; should be blast-nagasaki)")
        print(f"  [[US-XBH]]  = {d['blast-hiroshima']:.0f}     (blast-limited, Hiroshima-anchored)")
        print(f"  [[US-XTN]]  = {d['thermal-nagasaki']:.0f}     (thermal, Nagasaki-anchored)")
        print(f"  [[US-XBN]]  = {d['blast-nagasaki']:.0f}     (blast-limited, Nagasaki-anchored)")
        print(f"  [[US-YLO]]  = {predict(min(xs)):.0f}     (central-fit y at x_min)")
        print(f"  [[US-CLO]]  = {boot_ci(min(xs))[0][0]:.0f}     (low end of combined plausible range)")
    else:
        print("  US-*: n/a (missing runs)")

    # --- Numbers already in the text that must NOT have changed ----------------
    print("\n=== Sanity checks against numbers hard-coded in the text ===")
    if "default" in sweep["India"]:
        check("India x default", sweep["India"]["default"], 1.5, 1)
        check("India y default", predict(sweep["India"]["default"]), 7)
        lo, hi = boot_ci(sweep["India"]["default"])
        check("India fit CI low", lo[0], 6)
        check("India fit CI high", hi[0], 14)
    if "default" in sweep["Pakistan"]:
        check("Pakistan x default", sweep["Pakistan"]["default"], 7.7, 1)
        check("Pakistan y default", predict(sweep["Pakistan"]["default"]), 24)
        lo, hi = boot_ci(sweep["Pakistan"]["default"])
        check("Pakistan fit CI low", lo[0], 20)
        check("Pakistan fit CI high", hi[0], 31)
    if "default" in sweep["US"]:
        check("US x default", sweep["US"]["default"], 24)
        check("US y default", predict(sweep["US"]["default"]), 54)
        lo, hi = boot_ci(sweep["US"]["default"])
        check("US fit CI low", lo[0], 46)
        check("US fit CI high", hi[0], 66)
    if "thermal-hiroshima" in sweep["US"]:
        check("US x thermal-hiroshima", sweep["US"]["thermal-hiroshima"], 27)
        check("US y at x_max", predict(sweep["US"]["thermal-hiroshima"]), 60)
        check("US CI95 at x_max", boot_ci(sweep["US"]["thermal-hiroshima"])[1][0], 73)
    # out-of-sample check against the 1974 Katz point (excluded from the fit)
    check("Katz y at x=10", predict(KATZ_X), 29)
    lo, hi = boot_ci(KATZ_X)
    check("Katz fit CI low", lo[0], 25)
    check("Katz fit CI high", hi[0], 36)
    for s in ["India", "Pakistan"]:
        if "thermal-hiroshima" in sweep[s]:
            expected = {"India": 1.4, "Pakistan": 7.6}[s]
            check(f"{s} x Hiroshima-anchored", sweep[s]["thermal-hiroshima"], expected, 1)

    # --- Global weighting (default case only; unchanged by the sweep) ----------
    print("\n=== Global direct terms (default case; Section 5.1 inputs) ===")
    if all("default" in sweep[s] for s in SCENARIOS):
        w_india, w_pak = 0.07, 0.006  # share of world industrial GDP
        xI, xP = sweep["India"]["default"], sweep["Pakistan"]["default"]
        yI, yP = predict(xI), predict(xP)
        print(f"India  x={xI:.3f}% y={yI:.1f}% ; Pakistan x={xP:.3f}% y={yP:.1f}%")
        print(f"direct global INFRASTRUCTURE loss = {xI * w_india + xP * w_pak:.3f}%")
        print(f"direct global OUTPUT decline      = {yI * w_india + yP * w_pak:.3f}%")
        w_usr = 0.125
        yUS = predict(sweep["US"]["default"])
        print(f"US x={sweep['US']['default']:.3f}% -> y={yUS:.1f}%")
        print(f"US-Russia direct global output (12.5% weight, both at {yUS:.0f}%) = {w_usr * yUS:.1f}%")

    print("\nNote: combined plausible ranges are the union of the 90% bootstrap")
    print("intervals across the burn-radius prescriptions — [CI5 at x_min, CI95 at")
    print("x_max] — quoted in the manuscript as plausible bounding ranges, NOT")
    print("probability intervals (the prescriptions are alternative models with no")
    print("probability weights). Headline parentheticals remain the fit-only CIs.")


if __name__ == "__main__":
    main()

"""Burn-radius sensitivity sweep: rerun the scenario industrial-destruction
fractions under five burn-radius configurations: the fiducial fit implemented
in src/main.py ("default" = 0.75*Y^0.38) plus a clean 2x2 grid of bounding
cases — each exponent {1/3 (blast-limited), 1/2 (unattenuated thermal)}
anchored in turn at each empirical data point {Hiroshima (13 km^2 destroyed at
15 kt), Nagasaki (6.7 km^2 at 21 kt)}:

    thermal-hiroshima   R = 2.03 * (Y/15)^(1/2)   [identical to src "Toon"]
    thermal-nagasaki    R = 1.46 * (Y/21)^(1/2)
    blast-hiroshima     R = 2.03 * (Y/15)^(1/3)
    blast-nagasaki      R = 1.46 * (Y/21)^(1/3)

The Nagasaki-anchored cases have radii 36-39% smaller than their
Hiroshima-anchored counterparts at every yield. At 15 kt (India-Pakistan) the
two Hiroshima-anchored cases coincide exactly at their anchor, so the anchor
choice dominates the burn-radius uncertainty there; the exponent cases only
separate at high yields (US-Russia scenario). The bounding cases are
implemented here by wrapping Country.calculate_max_radius_burn — src/ is not
modified.

Because thermal-hiroshima is numerically identical to the old "Toon"
prescription, existing "Toon" rows in the results file are counted as
thermal-hiroshima and not re-run. Rows from the old bounding scheme
("overpressure", "nagasaki") are ignored. Since burn radius does not affect
targeting in either scenario, the India and Pakistan blast-hiroshima runs must
reproduce the thermal-hiroshima results exactly (both radii equal 2.03 km at
15 kt) — a built-in sanity check; skip them with --scenario/--prescription if
AWS time is tight.

MUST be run from the scripts/ directory (src/main.py uses ../data relative paths):

    cd scripts && python radius_sweep.py                # full sweep (15 runs, done ones skipped)
    python radius_sweep.py --scenario US --prescription blast-nagasaki   # single run

Results append to radius_sweep_results.jsonl as each run completes; runs already
present in that file are skipped automatically, so the script can be re-launched
after a crash or OOM kill and it will resume where it left off. Afterwards, run
propagate_aws_results.py to turn the sweep into manuscript numbers.

US rows produced before the 2026-07-06 targeting fixes (OPEN-RISOP name
collisions silently dropped 187 of 2,030 warheads; the burn ellipse was clipped
by the kill-radius bounding box on ground bursts) are invalid. Drop them with

    python radius_sweep.py --invalidate US

then re-launch the sweep; the India/Pakistan rows are unaffected by both fixes
(15 kt airbursts, Toon kill radius exceeds every burn radius) and are kept.

Settings mirror the paper notebooks exactly — do not change them:
- India/Pakistan: india-pakistan.ipynb (degrade=False, kill_radius 'Toon',
  85 x 15 kt, max-fatality targeting, non_overlapping=False)
- US: nuclear-attack.ipynb cell 29 (degrade=True, factor 3, OPEN-RISOP full plan)
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

sys.path.append("../src")
import main
from main import Country

LOG_PATH = "radius_sweep_results.jsonl"

# Anchor radii: sqrt(13/pi) and sqrt(6.7/pi), rounded to 3 significant figures
# exactly as in src/main.py's "Toon" prescription so that thermal-hiroshima
# reproduces previous "Toon" runs bit-for-bit.
R_HIROSHIMA, Y_HIROSHIMA = 2.03, 15
R_NAGASAKI, Y_NAGASAKI = 1.46, 21

BOUNDING_RADII = {
    "thermal-hiroshima": lambda y: R_HIROSHIMA * (y / Y_HIROSHIMA) ** 0.5,
    "thermal-nagasaki": lambda y: R_NAGASAKI * (y / Y_NAGASAKI) ** 0.5,
    "blast-hiroshima": lambda y: R_HIROSHIMA * (y / Y_HIROSHIMA) ** (1 / 3),
    "blast-nagasaki": lambda y: R_NAGASAKI * (y / Y_NAGASAKI) ** (1 / 3),
}

PRESCRIPTIONS = ["default"] + list(BOUNDING_RADII)
SCENARIOS = ["India", "Pakistan", "US"]

# Old result rows that are numerically identical to a new prescription.
ALIASES = {"Toon": "thermal-hiroshima"}

_orig_burn = Country.calculate_max_radius_burn


def _patched_burn(burn_radius_prescription, yield_kt):
    if burn_radius_prescription in BOUNDING_RADII:
        return BOUNDING_RADII[burn_radius_prescription](yield_kt)
    return _orig_burn(burn_radius_prescription, yield_kt)


Country.calculate_max_radius_burn = staticmethod(_patched_burn)


def already_done():
    done = set()
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "industry_destroyed_pct" in d:
                    presc = ALIASES.get(d["prescription"], d["prescription"])
                    done.add((d["scenario"], presc))
    return done


def run_india_pakistan(name, presc):
    c = Country(
        name,
        landscan_year=2022,
        degrade=False,
        degrade_factor=1,
        burn_radius_prescription=presc,
        kill_radius_prescription="Toon",
    )
    c.attack_max_fatality(arsenal=85 * [15], include_injuries=False, non_overlapping=False)
    return c


def run_us(presc):
    c = Country(
        "United States of America",
        landscan_year=2022,
        degrade=True,
        degrade_factor=3,
        burn_radius_prescription=presc,
    )
    c.apply_OPEN_RISOP_nuclear_war_plan()
    return c


def invalidate_scenario(scenario):
    """Drop a scenario's rows from the results file so it gets re-run.

    Needed after the 2026-07-06 US-targeting fixes (OPEN-RISOP name collisions
    dropped ~9% of warheads; burn box was clipped on ground bursts): old US
    rows are invalid, while India/Pakistan rows (airbursts, unaffected code
    paths) remain valid and should be kept to avoid re-running them.
    """
    if not os.path.exists(LOG_PATH):
        print(f"nothing to invalidate: {LOG_PATH} not found")
        return
    kept, dropped = [], 0
    with open(LOG_PATH) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if d.get("scenario") == scenario:
                dropped += 1
            else:
                kept.append(line)
    with open(LOG_PATH, "w") as f:
        f.writelines(kept)
    print(f"dropped {dropped} {scenario} row(s) from {LOG_PATH}; kept {len(kept)} line(s)")


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--prescription", choices=PRESCRIPTIONS)
    p.add_argument("--invalidate", choices=SCENARIOS, metavar="SCENARIO",
                   help="drop this scenario's rows from the results file and exit "
                        "(use after a code fix invalidates its old runs)")
    args = p.parse_args()

    if args.invalidate:
        invalidate_scenario(args.invalidate)
        return

    prescriptions = [args.prescription] if args.prescription else PRESCRIPTIONS
    scenarios = [args.scenario] if args.scenario else SCENARIOS

    log_f = open(LOG_PATH, "a", buffering=1)

    def log(**kw):
        log_f.write(json.dumps(kw) + "\n")
        print(kw, flush=True)

    done = already_done()
    for presc in prescriptions:
        for scenario in scenarios:
            if (scenario, presc) in done:
                print(f"skip {scenario}/{presc}: already in {LOG_PATH}", flush=True)
                continue
            t0 = time.time()
            try:
                c = run_us(presc) if scenario == "US" else run_india_pakistan(scenario, presc)
                total, immediate, radiation = c.get_total_fatalities()
                log(
                    scenario=scenario,
                    prescription=presc,
                    industry_destroyed_pct=round(100 * c.get_total_destroyed_industrial_area(), 3),
                    fatalities=total,
                    fatalities_immediate=immediate,
                    fatalities_radiation=radiation,
                    minutes=round((time.time() - t0) / 60, 1),
                )
                del c
                gc.collect()
            except Exception as e:
                log(scenario=scenario, prescription=presc, error=str(e))
                traceback.print_exc()

    print("sweep pass complete", flush=True)


if __name__ == "__main__":
    main_cli()

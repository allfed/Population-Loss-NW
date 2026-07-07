#!/usr/bin/env python3
"""
Run the full MRIO cross-check and write all deliverables to MRIO/outputs/.

Deliverables (spec sections 1, 7):
  results.csv              tidy table {year, method, s, lambda, RoW_decline_pct,
                           sigma_MER, global_decline_pct} incl. M1+M2 envelope rows,
                           plus the Thailand backtest rows if the 2011 table is present.
  shares.csv              US+RUS industrial share of world industrial GO and VA (MER).
  sector_mapping.csv      ISIC industry code -> description, industrial flag.
  method2_top_sectors.csv ten most-affected RoW country-sectors (Method 2, central case).
  sensitivity_construction.csv   industrial definition incl. construction (F).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mrio_crosscheck as mc

SHOCK_REGIONS = ("USA", "RUS")
SHOCKS = [0.50, 0.59, 0.72]          # 90% interval ends + central case (7-point fit
                                     # at x = 26.577%, the post-bugfix 2026-07-07
                                     # sweep value; 1974 model point excluded)
CENTRAL_S = 0.59
US_RU = SHOCK_REGIONS

# years to run: 2019 primary baseline (pre-COVID), 2021 & 2022 sensitivities
YEARS = [2019, 2021, 2022]
PRIMARY_YEAR = 2019


def lambdas_for(s: float) -> list[float]:
    """lambda in {1-s, 0.7, 1.0}; 1-s is the benchmark (spec section 5, Method 1)."""
    return sorted({round(1.0 - s, 4), 0.7, 1.0})


def envelope_row(year, s, lam, m1_row, m2_row, sigma) -> dict:
    row_decl = (m1_row["RoW_decline_pct"] + m2_row["RoW_decline_pct"]) / 100.0
    glob = sigma * s + row_decl * (1.0 - sigma)
    return {
        "year": year, "method": "M1+M2_envelope", "s": s, "lambda": lam,
        "RoW_decline_pct": 100.0 * row_decl,
        "sigma_MER": 100.0 * sigma,
        "global_decline_pct": 100.0 * glob,
    }


COLS = ["year", "method", "s", "lambda", "RoW_decline_pct", "sigma_MER", "global_decline_pct"]


def run_us_russia(models: dict[int, mc.ICIO]) -> pd.DataFrame:
    rows = []
    for year, m in models.items():
        sigma = mc.sigma_mer(m, US_RU)["sigma_go"]
        for s in SHOCKS:
            m2 = mc.method2(m, s, US_RU)                     # mixed Ghosh (Method 2 proper)
            rows.append({k: m2[k] for k in COLS})
            m2va = mc.method2_va_shock(m, s, US_RU)          # VA-shock sensitivity (labelled)
            rows.append({k: m2va[k] for k in COLS})
            for lam in lambdas_for(s):
                m1 = mc.method1(m, s, lam, US_RU)
                rows.append({k: m1[k] for k in COLS})
                # envelope uses the mixed-Ghosh M2 only, not the VA-shock sensitivity
                rows.append(envelope_row(year, s, lam, m1, m2, sigma))
    df = pd.DataFrame(rows)[COLS]
    # order: method group then s then lambda, but keep envelope after its M1
    return df.sort_values(["year", "s", "method", "lambda"],
                          na_position="first").reset_index(drop=True)


def run_shares(models: dict[int, mc.ICIO]) -> pd.DataFrame:
    rows = []
    for year, m in models.items():
        sh = mc.sigma_mer(m, US_RU)
        rows.append({
            "year": year,
            "sigma_go_pct": 100.0 * sh["sigma_go"],
            "sigma_va_pct": 100.0 * sh["sigma_va"],
            "world_ind_go_trillion_usd": sh["world_ind_go_musd"] / 1e6,
            "world_ind_va_trillion_usd": sh["world_ind_va_musd"] / 1e6,
        })
    return pd.DataFrame(rows)


def run_construction_sensitivity(models: dict[int, mc.ICIO]) -> pd.DataFrame:
    """Central case with construction (F) folded into the industrial definition."""
    rows = []
    for year, m in models.items():
        for codes, tag in [(mc.INDUSTRIAL, "excl_F"), (mc.INDUSTRIAL_WITH_F, "incl_F")]:
            sigma = mc.sigma_mer(m, US_RU, codes)["sigma_go"]
            s = CENTRAL_S
            lam = round(1 - s, 4)
            m1 = mc.method1(m, s, lam, US_RU, codes)
            m2 = mc.method2(m, s, US_RU, codes)
            env = envelope_row(year, s, lam, m1, m2, sigma)
            for r, meth in [(m1, "M1_mixed_leontief"), (m2, "M2_ghosh"), (env, "M1+M2_envelope")]:
                rows.append({
                    "year": year, "industrial_def": tag, "method": meth,
                    "s": s, "lambda": lam,
                    "RoW_decline_pct": r["RoW_decline_pct"],
                    "sigma_MER": r["sigma_MER"],
                    "global_decline_pct": r["global_decline_pct"],
                })
    return pd.DataFrame(rows)


def run_sector_mapping() -> pd.DataFrame:
    names = mc.load_industry_names()
    all_codes = mc._AG_CODES + mc.INDUSTRIAL_WITH_F + mc._SERVICE_CODES
    # preserve ICIO ordering
    order = (mc._AG_CODES + mc.MINING + mc.MANUF + mc.UTIL + mc.CONSTRUCTION + mc._SERVICE_CODES)
    rows = []
    for c in order:
        rows.append({
            "industry_code": c,
            "description": names.get(c, ""),
            "industrial_excl_F": c in mc.INDUSTRIAL,
            "industrial_incl_F": c in mc.INDUSTRIAL_WITH_F,
        })
    return pd.DataFrame(rows)


THAILAND_PROXY_YEAR = 2016   # earliest table in the 2016-2022 release; used until 2011 is downloaded


def run_thailand_backtest() -> pd.DataFrame | None:
    """Spec section 6: 2011 table, s=0.35 on Thailand industry, world industrial decline.

    The true backtest needs the 2011 ICIO table (not in the 2016-2022 file). Until it is
    downloaded we run a clearly-labelled 2016 proxy: Thailand's share of world industry and
    its supply-chain position are broadly stable over 2011->2016, so the world-magnitude check
    is still informative, but the year is flagged.
    """
    try:
        m = mc.build(2011, verbose=True)
        year_label = "2011"
    except FileNotFoundError:
        print(f"\n[Thailand backtest] 2011 table not found -> using {THAILAND_PROXY_YEAR} PROXY "
              f"(download the 2011 SML zip to run the true backtest).")
        m = mc.build(THAILAND_PROXY_YEAR, verbose=True)
        year_label = f"{THAILAND_PROXY_YEAR}(proxy_for_2011)"
    s = 0.35
    shock = ("THA",)
    rows = []
    # world industrial decline = direct + spillover (world = everyone; RoW here = non-THA)
    for lam in lambdas_for(s):
        m1 = mc.method1(m, s, lam, shock)
        rows.append(_world_row(m, m1, s, lam, "M1_mixed_leontief", shock, year_label))
    m2 = mc.method2(m, s, shock)                             # mixed Ghosh (Method 2 proper)
    rows.append(_world_row(m, m2, s, np.nan, "M2_ghosh", shock, year_label))
    m2va = mc.method2_va_shock(m, s, shock)                  # VA-shock sensitivity (labelled)
    rows.append(_world_row(m, m2va, s, np.nan, "M2_ghosh_va_sens", shock, year_label))
    return pd.DataFrame(rows)


def _world_row(m, rep, s, lam, method, shock, year_label):
    """World industrial decline for the backtest = direct + spillover(1-sigma)."""
    sigma = mc.sigma_mer(m, shock)["sigma_go"]
    row_decl = rep["RoW_decline_pct"] / 100.0
    world = sigma * s + row_decl * (1.0 - sigma)
    return {
        "year": year_label, "method": method, "s": s, "lambda": lam,
        "RoW_decline_pct": rep["RoW_decline_pct"],
        "sigma_MER": 100.0 * sigma,
        "global_decline_pct": 100.0 * world,   # == "world" industrial decline here
    }


def main():
    print("=" * 78)
    print("MRIO cross-check: RoW industrial spillovers from a US-Russia nuclear exchange")
    print("=" * 78)
    models = {y: mc.build(y) for y in YEARS}

    # --- Method 2 diagnostics: realized shocked-sector decline & baseline reproduction ---
    print("\n" + "=" * 78)
    print("METHOD 2 DIAGNOSTICS  (mixed Ghosh delivers full s; VA-shock under-delivers)")
    print("=" * 78)
    for year, m in models.items():
        base_resid = mc.mixed_ghosh_baseline_check(m, US_RU)
        print(f"[{year}] mixed-Ghosh baseline reproduction (no shock): rel L1 = {base_resid:.2e}")
        assert base_resid < 1e-4, f"[{year}] mixed-Ghosh baseline check failed: {base_resid:.2e}"
        for s in SHOCKS:
            r2 = mc.method2(m, s, US_RU)
            r2va = mc.method2_va_shock(m, s, US_RU)
            print(f"[{year}] s={s:.2f}  realized dx_K: mixed Ghosh = "
                  f"{100 * r2['realized_dxK']:6.2f}% (target {100 * s:.0f}%)   "
                  f"VA-shock sensitivity = {100 * r2va['realized_dxK']:5.2f}%")
            assert abs(r2["realized_dxK"] - s) < 1e-9, \
                f"[{year}] mixed-Ghosh realized dx_K != s ({r2['realized_dxK']} vs {s})"

    # --- US-Russia main results ---
    res = run_us_russia(models)

    # --- Thailand backtest (appended to results.csv) ---
    tha = run_thailand_backtest()
    if tha is not None:
        res = pd.concat([res, tha], ignore_index=True)

    res.to_csv(mc.OUT / "results.csv", index=False)

    shares = run_shares(models)
    shares.to_csv(mc.OUT / "shares.csv", index=False)

    cons = run_construction_sensitivity(models)
    cons.to_csv(mc.OUT / "sensitivity_construction.csv", index=False)

    mapping = run_sector_mapping()
    mapping.to_csv(mc.OUT / "sector_mapping.csv", index=False)

    # --- Top-affected sectors, central case, primary year, Method 2 ---
    m = models[PRIMARY_YEAR]
    r2 = mc.method2(m, CENTRAL_S, US_RU)
    top_abs = mc.top_affected(m, r2["dx"], US_RU, n=15)
    top_pct = mc.top_affected_by_pct(m, r2["dx"], US_RU, n=15)
    top_abs.insert(0, "rank_by", "abs_usd")
    top_pct.insert(0, "rank_by", "pct")
    pd.concat([top_abs, top_pct], ignore_index=True).to_csv(
        mc.OUT / "method2_top_sectors.csv", index=False)

    # --- console summary ---
    print("\n" + "=" * 78)
    print(f"SECTOR MAPPING (industrial = {len(mc.INDUSTRIAL)} ISIC codes, excl. construction F)")
    print("=" * 78)
    print(mapping.to_string(index=False))

    print("\n" + "=" * 78)
    print("SHARES (MER)  -- sigma feeds the PPP-vs-MER paragraph")
    print("=" * 78)
    print(shares.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"HEADLINE: {PRIMARY_YEAR}, central shock s={CENTRAL_S}")
    print("=" * 78)
    head = res[(res.year == PRIMARY_YEAR) & (res.s == CENTRAL_S)]
    print(head.to_string(index=False))

    print("\n" + "=" * 78)
    print("FULL US-RUSSIA RESULTS (all years, all s, all lambda)")
    print("=" * 78)
    us = res[res.year.isin(YEARS)]
    with pd.option_context("display.max_rows", None):
        print(us.to_string(index=False))

    if tha is not None:
        print("\n" + "=" * 78)
        print("THAILAND 2011 BACKTEST  (s=0.35 on THA industry; 'global_decline_pct' = world)")
        print("observed transient world industrial decline ~2.5%")
        print("=" * 78)
        print(tha.to_string(index=False))

    print("\n" + "=" * 78)
    print("CONSTRUCTION SENSITIVITY (central case)")
    print("=" * 78)
    print(cons.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"TOP-AFFECTED RoW SECTORS (Method 2, {PRIMARY_YEAR}, s={CENTRAL_S})")
    print("=" * 78)
    print("By absolute $ decline:")
    print(top_abs.to_string(index=False))
    print("\nBy % decline (baseline output > $1bn):")
    print(top_pct.to_string(index=False))

    print(f"\nAll CSVs written to {mc.OUT}")


if __name__ == "__main__":
    main()

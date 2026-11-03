#!/usr/bin/env python3
"""
Global industrial output loss for the Toon, Robock & Turco (2008) SORT exchange (the
4,400 x 100 kt laydown behind the 150 Tg nuclear winter scenario), with and without HEMP.

Per-country share of industrial area destroyed (d_c, %) comes from rerunning that laydown with
src/main.py under the paper's burn-radius scaling (results/Toon2008_SORT_revised_burn_radius.csv;
same allocation and targeting rule as Toon et al.: most-populated, non-overlapping).

  step 1  infrastructure incapacitated   x_c
  step 2  national output decline        y(x_c) = 5.4 * x_c^0.73   (paper's fitted relation)
  step 3  decline in all other countries from the 2019 OECD ICIO table, same two channels as
          the US-Russia cross-check (mixed Leontief with the struck countries' final demand
          scaled by 1 - s_c, plus mixed Ghosh), generalised here to country-specific shocks

HEMP is assumed to cover all industry in the nine struck countries and nowhere else. A share g
of what survives blast and fire is affected, applied in two bracketing ways:

  via_fit    HEMP-disabled plant counts as incapacitated infrastructure (step 1):
             x_c = 1 - (1 - d_c)(1 - g), then steps 2 and 3  -> domestic cascades apply to it
  after_fit  HEMP idles a share g of surviving OUTPUT after step 2:
             s_c = 1 - (1 - y(d_c))(1 - g), then step 3      -> no domestic cascade (conservative)

Writes outputs/sort_scenario.csv.  ~6 min.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

import mrio_crosscheck as mc

YEAR = 2019
RESULTS = mc.HERE.parent / "results"
LAYDOWN = "Toon2008_SORT_revised_burn_radius.csv"
G_GRID = [0.0, 0.1, 0.2, 0.3, 1 / 3, 0.4, 0.5, 0.7, 0.75, 0.8]


def national_fit(d_pct: float) -> float:
    """Paper's power law (Figure 3): % infrastructure incapacitated -> output decline (fraction)."""
    return min(1.0, 5.4 * d_pct ** 0.73 / 100.0)


def propagate(m: mc.ICIO, shocks: dict[str, float]) -> tuple[float, float]:
    """Industrial decline (fractions) of all non-shocked regions through the customer (mixed
    Leontief) and input (mixed Ghosh) channels, for region-specific industrial shocks."""
    ind = mc.industrial_mask(m)
    K = ind & np.isin(m.regions, list(shocks))
    U = ~K
    s_vec = np.array([shocks.get(r, 0.0) for r in m.regions])
    x_K = (1.0 - s_vec[K]) * m.x[K]

    f = m.f.copy()
    for r, s in shocks.items():
        f -= s * m.f_dom([r])
    x_U1 = linalg.solve(np.eye(U.sum()) - m.A[np.ix_(U, U)], m.A[np.ix_(U, K)] @ x_K + f[U])

    v = m.w + m.tls
    x_U2 = linalg.solve((np.eye(U.sum()) - m.B[np.ix_(U, U)]).T, v[U] + x_K @ m.B[np.ix_(K, U)])

    row = (ind & ~K)[U]
    base = m.x[U][row].sum()
    return 1.0 - x_U1[row].sum() / base, 1.0 - x_U2[row].sum() / base


def main():
    m = mc.build(YEAR)
    ind = mc.industrial_mask(m)
    share = pd.Series(m.x[ind], index=m.regions[ind]).groupby(level=0).sum()
    share /= share.sum()

    d = pd.read_csv(RESULTS / LAYDOWN).set_index("iso3")["industry_destroyed_pct"]
    sigma = share[d.index].sum()
    print(f"struck countries: {100 * sigma:.1f}% of world industrial gross output (MER)")

    rows = []
    for g in G_GRID:
        for method in ("via_fit", "after_fit"):
            if g == 0 and method == "after_fit":
                continue                       # identical to via_fit without HEMP
            if method == "via_fit":
                shocks = {c: national_fit(100 * (1 - (1 - d[c] / 100) * (1 - g))) for c in d.index}
            else:
                shocks = {c: 1 - (1 - national_fit(d[c])) * (1 - g) for c in d.index}
            direct = sum(share[c] * s for c, s in shocks.items())
            m1, m2 = propagate(m, shocks)
            rows.append({
                "hemp_share": g, "hemp_method": method,
                "direct_global_pct": 100 * direct,
                "RoW_customer_pct": 100 * m1, "RoW_input_pct": 100 * m2,
                "global_decline_pct": 100 * (direct + (m1 + m2) * (1 - sigma)),
                **{f"s_{c}": 100 * shocks[c] for c in d.index},
            })
    df = pd.DataFrame(rows)
    df.to_csv(mc.OUT / "sort_scenario.csv", index=False)
    print(df.iloc[:, :6].round(1).to_string(index=False))


if __name__ == "__main__":
    main()

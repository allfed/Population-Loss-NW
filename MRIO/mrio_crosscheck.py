#!/usr/bin/env python3
"""
MRIO cross-check of rest-of-world industrial spillovers from a US-Russia nuclear exchange.

Implements the two required computations of MRIO/mrio-crosscheck-spec.md:

  Method 1 - Mixed (partially exogenous) Leontief model: backward / demand channel.
  Method 2 - Ghosh supply-side model: forward / supply channel.

Data: OECD Inter-Country Input-Output (ICIO) Tables, 2025 edition, SML CSV format,
current USD millions (market exchange rates). 81 regions (80 economies + ROW) x 50
ISIC Rev.4 industries = 4050 country-sectors.

CSV layout (verified against ReadMe_ICIO_small.xlsx):
  columns: V1 (row label), 4050 intermediate-use cols 'CCC_III',
           486 final-demand cols 'CCC_{HFCE,NPISH,GGFC,GFCF,INVNT,DPABR}', OUT
  rows:    4050 sector rows 'CCC_III', then TLS, VA, OUT

Run as a script to produce all deliverables in MRIO/outputs/.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "icio"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Sector definitions (spec section 3). Industrial = ISIC B (mining 05-09),
# C (manufacturing 10-33), D-E (utilities 35-39). Construction F excluded from
# the main definition; included only as a sensitivity.
# ---------------------------------------------------------------------------
MINING = ["B05", "B06", "B07", "B08", "B09"]
MANUF = ["C10T12", "C13T15", "C16", "C17_18", "C19", "C20", "C21", "C22", "C23",
         "C24A", "C24B", "C25", "C26", "C27", "C28", "C29", "C301", "C302T309", "C31T33"]
UTIL = ["D", "E"]
CONSTRUCTION = ["F"]
INDUSTRIAL = MINING + MANUF + UTIL           # 26 industries, excl. construction
INDUSTRIAL_WITH_F = INDUSTRIAL + CONSTRUCTION

FD_CATS = ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"]
SPECIAL_ROWS = ("TLS", "VA", "OUT")


def region_of(label: str) -> str:
    """'AGO_C10T12' -> 'AGO' (region codes are always 3 letters)."""
    return label[:3]


def industry_of(label: str) -> str:
    """'AGO_C10T12' -> 'C10T12'."""
    return label[4:]


def load_industry_names() -> dict[str, str]:
    """Human-readable ISIC descriptions, read from the ICIO ReadMe if present."""
    readme = DATA / "ReadMe_ICIO_small.xlsx"
    if not readme.exists():
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(readme, sheet_name="Area_Activities", header=None)
    names: dict[str, str] = {}
    # The right-hand block of the sheet holds: (idx, Code, Industry, ISIC Rev.4).
    for _, row in df.iterrows():
        for c in range(len(row) - 2):
            code = row.iloc[c]
            name = row.iloc[c + 1]
            if isinstance(code, str) and code in (INDUSTRIAL_WITH_F + _SERVICE_CODES + _AG_CODES):
                if isinstance(name, str) and name.strip():
                    names[code] = name.strip()
    return names


_AG_CODES = ["A01", "A02", "A03"]
_SERVICE_CODES = ["G", "H49", "H50", "H51", "H52", "H53", "I", "J58T60", "J61",
                  "J62_63", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]


# ---------------------------------------------------------------------------
# Baseline construction (spec section 4)
# ---------------------------------------------------------------------------
@dataclass
class ICIO:
    year: int
    labels: np.ndarray          # (N,) country-sector labels, order of rows == cols of Z
    regions: np.ndarray         # (N,) region code per sector
    industries: np.ndarray      # (N,) industry code per sector
    Z: np.ndarray               # (N, N) intermediate transactions
    f: np.ndarray               # (N,) total final demand by product (all FD cats, all buyers)
    fd_by_buyer: pd.DataFrame    # (N x R) final demand by product, by purchasing region
    x: np.ndarray               # (N,) gross output (producer/row side)
    w: np.ndarray               # (N,) value added
    tls: np.ndarray             # (N,) taxes less subsidies (column side)
    A: np.ndarray               # (N, N) technical coefficients  A = Z diag(1/x)
    B: np.ndarray               # (N, N) allocation coefficients  B = diag(1/x) Z
    G: np.ndarray               # (N, N) Ghosh inverse (I - B)^-1
    checks: dict                # identity-check residuals

    @property
    def N(self) -> int:
        return len(self.labels)

    def f_dom(self, regions) -> np.ndarray:
        """Final demand (by product) purchased by the given set of regions."""
        cols = [r for r in regions if r in self.fd_by_buyer.columns]
        if not cols:
            return np.zeros(self.N)
        return self.fd_by_buyer[cols].to_numpy(dtype=np.float64).sum(axis=1)


def build(year: int, shock_regions=("USA", "RUS"), verbose=True) -> ICIO:
    path = DATA / f"{year}_SML.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the OECD ICIO SML zip covering {year} "
            f"from https://oe.cd/icio and extract into {DATA}."
        )
    if verbose:
        print(f"[{year}] reading {path.name} ...")
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    sectors = [r for r in df.index if r not in SPECIAL_ROWS]
    labels = np.array(sectors)
    regions = np.array([region_of(s) for s in sectors])
    industries = np.array([industry_of(s) for s in sectors])

    Z = df.loc[sectors, sectors].to_numpy(dtype=np.float64)

    # Final demand by product, aggregated over the 6 FD categories, per purchasing region.
    fd_cols = [c for c in df.columns if industry_of(c) in FD_CATS]
    fd_wide = df.loc[sectors, fd_cols].copy()
    fd_wide.columns = [region_of(c) for c in fd_cols]
    fd_by_buyer = fd_wide.T.groupby(level=0).sum().T          # (N x R), R purchasing regions
    f = fd_by_buyer.to_numpy(dtype=np.float64).sum(axis=1)

    x = df.loc[sectors, "OUT"].to_numpy(dtype=np.float64)
    w = df.loc["VA", sectors].to_numpy(dtype=np.float64)
    tls = df.loc["TLS", sectors].to_numpy(dtype=np.float64)

    del df

    N = len(sectors)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_x = np.where(x > 0, 1.0 / x, 0.0)      # guard empty sectors (spec s.4)
    A = Z * inv_x[np.newaxis, :]                     # A[i,j] = Z[i,j]/x[j]  (col-normalised)
    B = Z * inv_x[:, np.newaxis]                     # B[i,j] = Z[i,j]/x[i]  (row-normalised)

    if verbose:
        print(f"[{year}] N={N} sectors; inverting (I-A), (I-B) ...")
    I = np.eye(N)
    L = np.linalg.inv(I - A)                         # Leontief inverse (validation only)
    G = np.linalg.inv(I - B)                         # Ghosh inverse (used by Method 2)

    checks = _identity_checks(Z, f, x, w, tls, A, L, G, verbose=verbose)
    del L                                            # free ~131 MB; not needed downstream

    return ICIO(year, labels, regions, industries, Z, f, fd_by_buyer, x, w, tls, A, B, G, checks)


def _identity_checks(Z, f, x, w, tls, A, L, G, verbose=True) -> dict:
    """Hard requirements from spec section 4."""
    def relmax(resid):
        denom = np.maximum(np.abs(x).sum(), 1.0)
        return float(np.abs(resid).sum() / denom)

    v = w + tls                                              # full primary input (Ghosh)
    row_resid = x - (Z.sum(axis=1) + f)                       # x = Z.rowsum + f
    col_resid = x - (Z.sum(axis=0) + w + tls)                # x = Z.colsum + VA + TLS
    lf_resid = x - (L @ f)                                    # L f = x
    vg_resid = x - (v @ G)                                    # v' G = x (Ghosh identity, v=VA+TLS)
    L_min = float(L.min())
    G_min = float(G.min())

    checks = {
        "row_identity_relL1": relmax(row_resid),
        "col_identity_relL1": relmax(col_resid),
        "Lf_identity_relL1": relmax(lf_resid),
        "vG_identity_relL1": relmax(vg_resid),
        "L_min": L_min,
        "G_min": G_min,
    }
    if verbose:
        print("  identity checks (relative L1 residual, want ~0):")
        for k in ("row_identity_relL1", "col_identity_relL1",
                  "Lf_identity_relL1", "vG_identity_relL1"):
            print(f"    {k:24s} {checks[k]:.3e}")
        print(f"    L_min {L_min:+.3e}   G_min {G_min:+.3e}  (want >= ~0)")
    return checks


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------
def industrial_mask(m: ICIO, industrial_codes=INDUSTRIAL) -> np.ndarray:
    return np.isin(m.industries, industrial_codes)


def shocked_mask(m: ICIO, shock_regions, industrial_codes=INDUSTRIAL) -> np.ndarray:
    """K = industrial sectors of the shocked regions."""
    return industrial_mask(m, industrial_codes) & np.isin(m.regions, list(shock_regions))


def row_industrial_mask(m: ICIO, shock_regions, industrial_codes=INDUSTRIAL) -> np.ndarray:
    """RoW industrial = industrial sectors of every economy except the shocked ones."""
    return industrial_mask(m, industrial_codes) & ~np.isin(m.regions, list(shock_regions))


# ---------------------------------------------------------------------------
# Method 1 - mixed (partially exogenous) Leontief model  (backward/demand)
# ---------------------------------------------------------------------------
def method1(m: ICIO, s: float, lam: float, shock_regions=("USA", "RUS"),
            industrial_codes=INDUSTRIAL) -> dict:
    """
    Fix constrained outputs x_K = (1-s) x_K0; scale shocked regions' final-demand
    columns by lam; solve the mixed model for the unconstrained outputs x_U:
        x_U = (I - A_UU)^-1 (A_UK x_K + f_U),   f_U scaled by lam on shock-region demand.
    Report the induced RoW industrial decline.
    """
    K = shocked_mask(m, shock_regions, industrial_codes)
    U = ~K
    A_UU = m.A[np.ix_(U, U)]
    A_UK = m.A[np.ix_(U, K)]

    x_K0 = m.x[K]
    x_K = (1.0 - s) * x_K0

    # f(lam): baseline final demand with the shock regions' purchases scaled by lam.
    f_dom = m.f_dom(shock_regions)
    f_scaled = (m.f - f_dom) + lam * f_dom
    f_U = f_scaled[U]

    x_U = linalg.solve(np.eye(U.sum()) - A_UU, A_UK @ x_K + f_U)

    x_new = m.x.copy()
    x_new[K] = x_K
    x_new[U] = x_U

    return _report(m, x_new, s, shock_regions, industrial_codes,
                   method="M1_mixed_leontief", lam=lam)


# ---------------------------------------------------------------------------
# Method 2 - Ghosh supply-side model  (forward/supply)
# ---------------------------------------------------------------------------
def method2(m: ICIO, s: float, shock_regions=("USA", "RUS"),
            industrial_codes=INDUSTRIAL) -> dict:
    """
    Mixed (partially exogenous) Ghosh model - the exact supply-side dual of
    Method 1's mixed Leontief model (Miller & Blair, ch. on supply-constrained /
    mixed models). Fix the shocked set's gross output exogenously at
    x_K = (1 - s) x_K0 - i.e. US/Russian industry delivers exactly s less to
    every customer, with allocation coefficients (the B-rows of K) held fixed -
    and solve the Ghosh row balance x_j = sum_i x_i B_ij + v_j restricted to the
    unconstrained set U for the induced outputs:
        x_U'(I - B_UU) = x_K' B_KU + v_U',   v = w + tls  (primary inputs, v'G = x).
    By construction the shocked sectors' own output falls by exactly s (unlike the
    naive VA-shock forward-propagation in method2_va_shock, which under-delivers
    the shock because unshocked upstream inputs keep flowing forward).

    Returns realized_dxK = (x_K0.sum() - x_K.sum())/x_K0.sum(), which must equal s
    to machine precision - a diagnostic printed by the runner.
    """
    K = shocked_mask(m, shock_regions, industrial_codes)
    U = ~K
    v = m.w + m.tls                                   # primary inputs (Ghosh: v'G = x)
    B_UU = m.B[np.ix_(U, U)]
    B_KU = m.B[np.ix_(K, U)]

    x_K0 = m.x[K]
    x_K = (1.0 - s) * x_K0                            # fixed exogenously
    rhs = v[U] + x_K @ B_KU
    x_U = linalg.solve((np.eye(int(U.sum())) - B_UU).T, rhs)

    x_new = m.x.copy()
    x_new[K] = x_K
    x_new[U] = x_U
    dx = x_new - m.x

    rep = _report(m, x_new, s, shock_regions, industrial_codes,
                  method="M2_ghosh", lam=np.nan)
    rep["dx"] = dx
    rep["realized_dxK"] = (x_K0.sum() - x_K.sum()) / x_K0.sum()   # == s by construction
    return rep


def method2_va_shock(m: ICIO, s: float, shock_regions=("USA", "RUS"),
                     industrial_codes=INDUSTRIAL) -> dict:
    """
    SENSITIVITY ONLY - the naive value-added-shock Ghosh formulation from the
    original spec (MRIO/mrio-crosscheck-spec.md, Method 2): shock the primary
    inputs of the shocked set K by -s and forward-propagate through the Ghosh
    inverse,
        dw_j = -s w_j for j in K,   dx' = dw' G.
    Retained as a labelled sensitivity (method "M2_ghosh_va_sens") because it
    under-delivers the intended shock: with VA/GO ~ 0.40 in K, shocking VA by s
    realizes only a ~31% output decline in K at s = 0.55, because unshocked
    upstream inputs keep flowing forward into the shocked sectors. Method 2 proper
    (method2, the mixed Ghosh model) fixes x_K = (1 - s) x_K0 instead and so
    realizes the full s.

    Returns realized_dxK = -dx[K].sum()/x_K.sum() (< s here) for comparison.
    """
    K = shocked_mask(m, shock_regions, industrial_codes)
    dw = np.zeros(m.N)
    dw[K] = -s * m.w[K]
    dx = dw @ m.G                                     # row vector times Ghosh inverse
    x_new = m.x + dx
    rep = _report(m, x_new, s, shock_regions, industrial_codes,
                  method="M2_ghosh_va_sens", lam=np.nan)
    rep["dx"] = dx
    rep["realized_dxK"] = -dx[K].sum() / m.x[K].sum()            # < s: forward channel only
    return rep


def mixed_ghosh_baseline_check(m: ICIO, shock_regions=("USA", "RUS"),
                               industrial_codes=INDUSTRIAL) -> float:
    """
    Baseline reproduction check for method2's mixed-Ghosh solve (analogous to the
    identity checks in _identity_checks): with no shock (x_K = x_K0) the
    partially-exogenous solve must recover the baseline unconstrained outputs x_U.
    Returns the relative L1 residual (want ~1e-6).
    """
    K = shocked_mask(m, shock_regions, industrial_codes)
    U = ~K
    v = m.w + m.tls
    B_UU = m.B[np.ix_(U, U)]
    B_KU = m.B[np.ix_(K, U)]
    x_U = linalg.solve((np.eye(int(U.sum())) - B_UU).T, v[U] + m.x[K] @ B_KU)
    return float(np.abs(x_U - m.x[U]).sum() / np.abs(m.x[U]).sum())


def _report(m: ICIO, x_new, s, shock_regions, industrial_codes, method, lam) -> dict:
    row = row_industrial_mask(m, shock_regions, industrial_codes)
    base = m.x[row].sum()
    decline_frac = (m.x[row].sum() - x_new[row].sum()) / base   # positive == decline
    sigma = sigma_mer(m, shock_regions, industrial_codes)["sigma_go"]
    global_frac = sigma * s + decline_frac * (1.0 - sigma)
    return {
        "year": m.year, "method": method, "s": s, "lambda": lam,
        "RoW_decline_pct": 100.0 * decline_frac,
        "sigma_MER": 100.0 * sigma,
        "global_decline_pct": 100.0 * global_frac,
        "_x_new": x_new,
    }


# ---------------------------------------------------------------------------
# Shares (spec section 7): US+RUS industrial share of world industrial totals, MER
# ---------------------------------------------------------------------------
def sigma_mer(m: ICIO, shock_regions=("USA", "RUS"), industrial_codes=INDUSTRIAL) -> dict:
    ind = industrial_mask(m, industrial_codes)
    K = shocked_mask(m, shock_regions, industrial_codes)
    return {
        "sigma_go": m.x[K].sum() / m.x[ind].sum(),
        "sigma_va": m.w[K].sum() / m.w[ind].sum(),
        "world_ind_go_musd": float(m.x[ind].sum()),
        "world_ind_va_musd": float(m.w[ind].sum()),
    }


def top_affected(m: ICIO, dx, shock_regions=("USA", "RUS"), n=10) -> pd.DataFrame:
    """Ten most affected RoW country-sectors under a Method-2 dx vector."""
    row = ~np.isin(m.regions, list(shock_regions))
    idx = np.where(row)[0]
    names = load_industry_names()
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(m.x[idx] > 0, 100.0 * dx[idx] / m.x[idx], 0.0)
    df = pd.DataFrame({
        "region": m.regions[idx],
        "industry": m.industries[idx],
        "industry_name": [names.get(c, c) for c in m.industries[idx]],
        "baseline_go_musd": m.x[idx],
        "delta_go_musd": dx[idx],
        "pct_change": pct,
    })
    return df.sort_values("delta_go_musd").head(n).reset_index(drop=True)


def top_affected_by_pct(m: ICIO, dx, shock_regions=("USA", "RUS"), n=10,
                        min_baseline_musd=1000.0) -> pd.DataFrame:
    """As top_affected, but ranked by % decline among RoW sectors with output > $1bn."""
    row = ~np.isin(m.regions, list(shock_regions))
    idx = np.where(row & (m.x > min_baseline_musd))[0]
    names = load_industry_names()
    df = pd.DataFrame({
        "region": m.regions[idx],
        "industry": m.industries[idx],
        "industry_name": [names.get(c, c) for c in m.industries[idx]],
        "baseline_go_musd": m.x[idx],
        "delta_go_musd": dx[idx],
        "pct_change": 100.0 * dx[idx] / m.x[idx],
    })
    return df.sort_values("pct_change").head(n).reset_index(drop=True)

# MRIO cross-check of rest-of-world industrial spillovers from a US–Russia nuclear exchange

*Methods note, suitable for adaptation into the paper. Code: `mrio_crosscheck.py`,
`run_crosscheck.py`; outputs in `outputs/`. All figures are model output from the OECD
ICIO 2025 edition.*

## 1. Question and data

We ask: if US and Russian **industrial gross output** is exogenously reduced by a fraction
`s`, by how much does **rest-of-world (RoW) industrial gross output** fall through
international production linkages alone? We bound this with two standard, citable
input–output computations.

Data: **OECD Inter-Country Input-Output (ICIO) Tables, 2025 edition** (current USD millions,
market exchange rates), covering **81 regions** (80 economies + a Rest-of-World aggregate)
× **50 ISIC Rev.4 industries** = **4 050 country-sectors**. China and Mexico are single
economies in this edition (no processing/non-processing split, so no aggregation needed).
Baseline year **2019** (latest pre-COVID; 2020 coefficients are pandemic-distorted); **2021
and 2022** are run as sensitivities.

**Industrial** = ISIC **B** (mining 05–09), **C** (manufacturing 10–33), **D–E** (utilities
35–39): 26 of the 50 industry codes (mapping in `outputs/sector_mapping.csv`). Construction
(**F**) is excluded from the main definition (essentially non-traded) and run as a
sensitivity. **Gross output** (not value added) is the reported concept, as the closest
analogue of the industrial-production indices used elsewhere in the paper.

## 2. Baseline construction

Let `Z` be the 4 050×4 050 intermediate-transactions matrix, `f` total final demand by
product (summed over the six ICIO final-demand categories and all purchasing regions), `x`
gross output, `w` value added, and `tls` taxes-less-subsidies. Technical and allocation
coefficients and their inverses:

    A = Z · diag(1/x)      L = (I − A)⁻¹        (Leontief, demand-driven)
    B = diag(1/x) · Z      G = (I − B)⁻¹        (Ghosh,   supply-driven)

Empty sectors (x = 0) get zero coefficients. All identity checks pass on every table
(relative L1 residuals; 2019 shown):

| check | residual |
|---|---|
| `x = Z·1_row + f` | 2.3×10⁻⁶ |
| `x = Z·1_col + w + tls` | 5.3×10⁻⁷ |
| `L·f = x` | 3.0×10⁻⁶ |
| `(w+tls)·G = x` (Ghosh) | 9.7×10⁻⁷ |
| `min L`, `min G` ≥ 0 | 0, 0 |

Both mixed models (below) with no shock reproduce the baseline: mixed Leontief to 3×10⁻⁶,
mixed Ghosh to 1.0×10⁻⁶.

## 3. The two channels (Miller & Blair, *Input-Output Analysis*)

The methods measure **different, largely non-overlapping** channels — backward (demand) and
forward (supply) — not optimistic/pessimistic versions of one calculation. Neither alone is
complete; we report both and their sum as a rough envelope.

**Method 1 — mixed (partially exogenous) Leontief model — backward/demand channel.**
Captures RoW's *lost sales* to the US and Russia (as suppliers of intermediates to their
industry and of products to their final demand), while letting RoW producers fully replace
any lost US/Russian inputs. With the shocked set `K` = US+RU industrial sectors and the
unconstrained set `U` = everything else:

    x_K = (1 − s)·x_K⁰                                    (constrained outputs fixed)
    x_U = (I − A_UU)⁻¹ · (A_UK·x_K + f_U(λ))              (solve for the rest)

`f_U(λ)` scales **all US and Russian final-demand columns by λ**, run at
`λ ∈ {1−s, 0.7, 1.0}`. `λ = 1−s` is the benchmark (final demand collapses in step with
industry); `λ = 1.0` sustains US/RU final demand at baseline (e.g. reconstruction-driven),
isolating the pure intermediate-purchase channel. RoW final demand is unchanged.

**Method 2 — mixed (partially exogenous) Ghosh model — forward/supply channel.** The exact
supply-side dual of the Method 1 mixed model (Miller & Blair, same chapter on
supply-constrained/mixed models). Fix the shocked sectors' gross output exogenously at
`x_K = (1 − s)·x_K⁰` — US/Russian industry delivers exactly `s` less to every customer, with
the allocation coefficients (the `B`-rows of `K`) held fixed — and solve the Ghosh row balance
`x_j = Σ_i x_i B_ij + v_j` for the unconstrained set `U`:

    x_U′·(I − B_UU) = x_K′·B_KU + v_U′        v = w + tls  (primary inputs, v′·G = x)

This captures downstream *input starvation with zero substitution* — the short-run
complementarity case (Boehm, Flaaen & Pandalai-Nayar 2019 justify near-Leontief short-run
behaviour for imported inputs; the Oosterhaven 1988 critique of Ghosh-as-quantity-model is
acknowledged, with the standard disaster-analysis framing of Santos & Haimes 2004). By
construction the shocked sectors' own output falls by exactly `s`.

*Why not simply shock value added?* The original spec forward-propagated a primary-input cut,
`Δw_j = −s·w_j (j ∈ K)`, `Δx = Δw·G`. But value added is only ≈0.40 of gross output in `K`,
so shocking VA by `s` realizes only a ≈31 % gross-output decline in the shocked sectors
themselves at `s = 0.55` (≈27 % / 38 % at `s = 0.48` / `0.67`): unshocked upstream inputs keep
flowing forward and prop the shocked sectors' output up, so the naive variant under-delivers
the intended shock. The mixed-Ghosh model above realizes the full `s` by construction. The
VA-shock variant is retained as a labelled sensitivity (`M2_ghosh_va_sens`) in
`outputs/results.csv`.

**Blind spot (why both are required):** in Ghosh, suppliers *to* the shocked sectors are
unaffected by construction — a foreign parts maker whose US customer is destroyed keeps
producing at baseline. That backward channel lives only in Method 1. Conversely Method 1
cannot see input starvation downstream of the shocked sectors. The channels overlap
partially and interact at second order; a coherent combination (Method 3) is left as an
extension.

## 4. Results — US–Russia, central case (2019, s = 0.55)

| method | λ | RoW industrial decline | implied global industrial decline |
|---|---|---|---|
| M1 mixed Leontief | 0.45 (=1−s, benchmark) | **4.68 %** | 12.11 % |
| M1 mixed Leontief | 0.70 | 3.09 % | 10.76 % |
| M1 mixed Leontief | 1.00 | 1.19 % | 9.14 % |
| M2 mixed Ghosh | — | **1.32 %** | 9.25 % |
| *M2 VA-shock (sensitivity, not in envelope)* | — | *0.84 %* | *8.85 %* |
| **M1 + M2 envelope** | 0.45 | **6.00 %** | **13.24 %** |

The envelope sums Method 1 (benchmark λ) and the mixed-Ghosh Method 2; the VA-shock row is a
labelled sensitivity and is *not* in the envelope. Across the paper's 90 % interval (envelope,
benchmark λ = 1−s, 2019):

| s | RoW industrial decline | global industrial decline |
|---|---|---|
| 0.48 | 5.23 % | 11.55 % |
| 0.55 | 6.00 % | 13.24 % |
| 0.67 | 7.30 % | 16.13 % |

Year sensitivity (central s = 0.55, envelope, benchmark λ): 2019 → 6.0 % RoW / 13.2 % global;
2021 → 5.9 % / 12.7 %; 2022 → 6.6 % / 13.9 %. Results are stable across years; the 2019
tables predate the post-2022 reorientation of Russian energy trade and so, if anything,
slightly *inflate* the Russia-origin spillover (2022 comes in marginally higher, driven by
higher energy prices, not linkage volume). Including construction (F) in the industrial
definition *lowers* the RoW decline by ≈0.85 pp (to 5.14 % envelope): construction is large
and nearly non-traded, so it dilutes the internationally-propagated share. Full grid in
`outputs/results.csv`; construction sensitivity in `outputs/sensitivity_construction.csv`.

## 5. Share of world industry (σ) — MER vs PPP

US + Russia industrial gross output as a share of the world total, at market exchange rates:

| year | σ (gross output) | σ (value added) |
|---|---|---|
| 2019 | **14.8 %** | 18.5 % |
| 2021 | 13.8 % | 17.8 % |
| 2022 | 15.0 % | 19.5 % |

The paper currently uses a **PPP-based 12.5 %**. At **MER** the figure is **≈15 %** on gross
output and **≈18–20 %** on value added: US and Russian output is priced higher than its
PPP-adjusted physical volume, so both loom larger in market-value terms. This raises the
*direct* global industrial hit (σ·s) from ≈6.9 % (PPP) to **≈8.1 %** (MER GO) at s = 0.55 —
worth a sentence in the valuation-choice paragraph, since "industrial capacity" measured as
market-value flows (MER) is hit harder than the same capacity measured in physical/PPP terms.

## 6. Most-affected RoW sectors (Method 2, 2019, s = 0.55)

The forward channel lands exactly where the supply-chain geography predicts. **By share of
own output** (baseline output > $1 bn) the largest hits are **European refined-petroleum
sectors fed by Russian crude**: Belarus −27 %, Bulgaria −25 %, Slovakia −23 %, Lithuania
−21 %, Hungary/Finland −18 % (all ISIC C19), joined by **Belarusian electricity (−20 %),
basic iron and steel (−19 %), non-ferrous metals (−17 %), chemicals (−17 %), electrical
equipment (−15 %) and fabricated metal (−14 %)**, and by **Canadian and Chilean refined
petroleum (−13 %)**. **By absolute dollars** the largest hits are in **China** —
computer/electronics (C26, −$21 bn), chemicals (C20), machinery (C28), electrical equipment
(C27) and basic metals, plus Chinese construction and utilities — via US intermediates, and
in **North American motor vehicles**: Mexico (C29, −7.2 %), Canada (C29, −12.9 %) and Canadian
refined petroleum (−13.4 %) via the deeply integrated North American chains, with French
other-transport equipment (C302T309, −5.7 %) also prominent. Listing in
`outputs/method2_top_sectors.csv`.

## 7. Backtest — 2011 Thailand floods (s = 0.35 on Thai industry)

*Run on the 2016 table as a labelled proxy pending download of the 2011 ICIO table;
Thailand's world industry share and supply-chain position are broadly stable 2011→2016.*
Thailand is 0.95 % of world industrial output (σ). The model predicts a **world industrial
decline of ≈0.6 %** (M1 benchmark 0.54 %, M2 mixed Ghosh 0.42 %, envelope ≈0.62 %; the
VA-shock M2 sensitivity is lower at 0.36 %) against the **observed ≈2.5 %** transient decline
the paper cites — a **model/observed ratio of ≈0.25**.

This under-prediction is expected and is used as **interpretive calibration, not a rescaling
factor**: (i) annual coefficients with no inventory dynamics smooth a transient *monthly*
peak; (ii) the floods hit a globally-critical chokepoint (Thailand made ~40–45 % of world
hard-disk drives) that division-level ISIC aggregation cannot resolve. **Caveat for
transfer to the US–Russia case:** that shock is *persistent* (destroyed capacity does not
recover within the year) and *broad* (not a single chokepoint), so the annual-averaging and
aggregation biases apply far less — the Thailand ratio should not be multiplied onto the
US–Russia numbers.

## 8. What the framework omits, and the bottom line

**Omitted, direction of bias:** inventories and dynamics (understate the transient peak);
price responses (ambiguous); **finance/insurance/shipping freezes** (understate — a nuclear
exchange would freeze trade finance, marine insurance and shipping far beyond what
production linkages capture); post-2019 trade re-routing (small). Method 1 also lets RoW
fully re-source lost US/RU inputs (understates). **The net of the first-order omissions is
conservative** — the true disruption is very likely larger than these numbers.

**Bottom line.** Through international production linkages under peacetime coefficients, a
55 % cut to US+Russian industry mechanically propagates to a **≈6 % decline in RoW industrial
output** (envelope, central s = 0.55; 5.14 % if construction is included, and 5.2–7.3 % across
the s = 0.48–0.67 interval) and a **≈13 % decline in global industrial output** (central;
11.6–16.1 % across the interval, of which ≈8 pp is the direct US+RU loss at MER). This is
**below the paper's judged 10–20 % RoW decline**. The gap is real and
should be presented honestly: the MRIO measures only the trade-linkage channel and places a
**floor of ~5–7 %** on the RoW effect. Closing the gap to 10–20 % requires the channels this
framework cannot see — finance, insurance and shipping paralysis, inventory and price
dynamics, and the broader collapse of demand and confidence in a nuclear war — all of which
push in the same, upward direction.

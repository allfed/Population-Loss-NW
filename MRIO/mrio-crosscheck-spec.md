# MRIO cross-check of rest-of-world industrial spillovers from a US–Russia nuclear exchange

**Purpose of this document.** Self-contained instructions for an agent (Claude Code or an RA)
to implement a multi-regional input-output (MRIO) cross-check of the paper's judgment that
rest-of-world (RoW) industrial output would decline 10–20% following ~55% industrial output
losses in the US and Russia. No further context should be needed beyond this file.

---

## 1. Objective and success criteria

Estimate the induced percentage decline in **rest-of-world industrial gross output** when US
and Russian industrial gross output is exogenously reduced by 55% (central case; also 48% and
67%, the ends of the paper's 90% interval).

Deliver:
1. A results table: {method} × {shock size 48/55/67%} × {final-demand scaling λ, Method 1
   only} → RoW industrial decline (%), plus the implied global industrial decline
   (direct + spillover), and the Method 1 + Method 2 envelope sum.
2. A backtest of the same procedure on the 2011 Thailand floods (see §6).
3. The US+Russia share of world industrial gross output and value added in the MRIO tables
   (these are at market exchange rates and feed a PPP-vs-MER paragraph in the paper).
4. A short methods write-up (½–1 page) with equations, suitable for adaptation into the paper,
   plus the code and intermediate CSVs.

**Success looks like:** defensible bounding estimates from two standard, citable methods. If
the envelope brackets or approaches 10–20%, the paper's judgment is corroborated; if it comes
in materially lower or higher, report that honestly — the paper will present the tension.

## 2. Data

**Primary: OECD Inter-Country Input-Output (ICIO) tables.**
- Download the latest release from the OECD ICIO page (search "OECD ICIO download"; the
  tables are free). Releases cover roughly 76–80 economies plus a ROW aggregate, ~45
  industries (ISIC Rev. 4), in current US dollars (market exchange rates).
- **Read the release ReadMe before coding.** Verify against it: (a) the industry code list;
  (b) final-demand category codes (household consumption, NPISH, government, GFCF,
  inventories, etc.); (c) the labels of the value-added and gross-output rows; (d) whether
  the release splits China and/or Mexico into processing/non-processing sub-economies
  (often labelled CN1/CN2, MX1/MX2) — if so, **aggregate the splits into single CHN and MEX
  economies** before analysis.
- **Baseline year:** latest pre-COVID year (2019) as the primary baseline; latest available
  year as a sensitivity. 2020 coefficients are pandemic-distorted.
- **Caveat to carry into the write-up:** 2019 tables predate the post-2022 reorientation of
  Russian energy trade, so they overstate current Europe–Russia linkages somewhat; note the
  direction (slightly inflates the Russia-origin spillover) and, if a 2021+ table exists in
  the release, run it as a sensitivity.

**Fallbacks** if OECD download is blocked: WIOD 2016 release (free; 43 countries + RoW, 56
sectors, ends 2014 — dated Russia linkages) or Eora26 (free for academic use; 190 countries,
26 sectors, quality more variable). Prefer OECD ICIO.

## 3. Definitions

- **Industrial sectors** = ISIC Rev. 4 sections B (mining, divisions 05–09), C
  (manufacturing, 10–33), and D–E (utilities, 35–39). **Exclude construction (F, 41–43)**:
  it is essentially non-traded and would dilute the international propagation being tested;
  run one sensitivity including it and report the difference in a footnote. Map these
  divisions onto the release's industry codes programmatically and print the mapping for
  human inspection.
- **Shocked set K** = all industrial sectors of USA and RUS. **Unconstrained set U** = every
  other country-sector.
- **RoW industrial output** = baseline gross output of industrial sectors of all economies
  except USA and RUS (the ROW aggregate region counts as part of RoW).
- **Output concept:** gross output (the row-sum of intermediate sales plus final demand),
  not value added — gross output is the closer analogue of the industrial-production
  indices used for the paper's historical y. Report the value-added version as a secondary
  column.

## 4. Notation and baseline construction

Let `Z` be the (country-industry × country-industry) intermediate transactions matrix, `f`
the vector of total final demand by product (summed over all purchasing countries and
categories), `x` gross output, `w` value added (primary inputs), with N ≈ 3,500 sectors.

- Technical coefficients: `A = Z @ diag(1/x)`; Leontief inverse `L = inv(I − A)`.
- Allocation coefficients: `B = diag(1/x) @ Z`; Ghosh inverse `G = inv(I − B)`.
- Guard against division by zero for empty sectors (set coefficient columns/rows to 0).
- **Identity checks before any counterfactual (hard requirements):**
  `x ≈ Z.sum(axis=1) + f` and `x ≈ Z.sum(axis=0) + w + taxes` to within rounding;
  `L @ f ≈ x`; all entries of `L` and `G` ≥ 0 (tiny negative rounding noise may be zeroed).
- Prefer `scipy.linalg.solve` over explicit inversion where possible. A dense 3,500² float64
  matrix is ~100 MB; everything fits comfortably in memory.

## 5. The 2 computations

Run methods 1 and 2 as the required pair. **They measure different, largely non-overlapping
channels — backward (demand) and forward (supply) — not optimistic and pessimistic versions
of the same calculation.** Neither alone is a complete estimate: Method 2 cannot see losses
by suppliers *to* the shocked sectors (see its note below), and Method 1 cannot see input
starvation downstream of them. Report both, and report their sum as a rough envelope of the
combined effect, with an explicit caveat that the channels overlap partially and interact at
second order (Method 3, if run, combines them coherently). Method 3 is optional.
Shock size `s ∈ {0.48, 0.55, 0.67}`.

### Method 1 — Mixed (partially exogenous) Leontief model: the backward (demand) channel

Captures the RoW's **lost sales** to the US and Russia — as suppliers of intermediates to
their industry and of products to their final demand — while allowing RoW producers to fully
replace any lost US/Russian inputs.

- Fix constrained outputs: `x_K = (1 − s) · x_K⁰`. Note that the constrained sectors'
  intermediate purchases then fall mechanically with their output — this part of the
  channel is accounting, not a behavioral assumption about demand.
- Scale **all** US and RUS final-demand columns by a parameter `λ`, run at
  `λ ∈ {1 − s, 0.7, 1.0}`. `λ = 1 − s` is the benchmark (final demand collapses in step
  with industry); `λ = 1.0` holds US/RU final demand at baseline, representing fully
  sustained (e.g., reconstruction-driven) domestic demand, and isolates the pure
  intermediate-purchase channel. Report all three. RoW final demand unchanged.
- Solve the standard mixed model (Miller & Blair, *Input-Output Analysis*, ch. on
  supply-constrained/mixed models):
  `x_U = inv(I − A_UU) @ (A_UK @ x_K + f_U)`,
  where `f_U` is final demand for U-sector products after the `λ` column scaling.
- Report: `ΔRoW_ind = (x_U,ind − x⁰_U,ind).sum() / x⁰_RoW,ind.sum()` restricted to
  industrial sectors outside USA/RUS.

### Method 2 — Ghosh supply-side model: the forward (supply) channel

Captures downstream **input starvation with zero substitution or re-sourcing** — the
short-run complementarity case (Boehm, Flaaen & Pandalai-Nayar 2019 justifies near-Leontief
short-run behaviour for imported inputs). The Oosterhaven (1988) critique of
Ghosh-as-quantity-model should be acknowledged in the paper, with the standard
disaster-analysis framing (e.g., as in inoperability IO practice, Santos & Haimes 2004).
**Blind spot to state explicitly:** in Ghosh, suppliers *to* the shocked sectors are
unaffected by construction — a foreign parts maker whose US industrial customer is destroyed
keeps producing at baseline. That backward channel exists only in Method 1, which is why
both methods are required.

- Shock vector: `Δw_j = −s · w_j` for `j ∈ K`, else 0.
- Forward-propagate: `Δx' = Δw' @ G`.
- Report the RoW industrial change as in Method 1. Also report, for colour, the ten most
  affected RoW country-sectors (expect European chemicals/refining via Russian energy, and
  aerospace/electronics chains via US intermediates).

**Amendment (2026-07-04):** the `Δw = −s·w` shock above was found to realize only a ≈31%
gross-output decline in the shocked set `K` at `s = 0.55` (≈27%/38% at `s = 0.48`/`0.67`),
because value added is only ≈0.40 of gross output in `K`, so unshocked upstream inputs keep
flowing forward and prop the shocked sectors' output up — the naive VA shock under-delivers
the intended shock. It was replaced by the **partially-exogenous (mixed) Ghosh model** — the
exact supply-side dual of the Method 1 mixed model — which fixes `x_K = (1 − s)·x_K⁰`
exogenously and solves the Ghosh row balance `x_U′(I − B_UU) = x_K′ B_KU + v_U′` (with
`v = w + tls`) for the unconstrained outputs, realizing the full `s` by construction. The
original VA-shock formulation is retained as a labelled sensitivity (`M2_ghosh_va_sens`) in
`outputs/results.csv`. See `methods-writeup.md` §3 for the full statement.


## 6. Backtest: 2011 Thailand floods

Same pipeline, ICIO table for **2011**, shock `s = 0.35` applied to Thailand's industrial
sectors. Compare the predicted **world** industrial decline against the observed ~2.5%
(temporary) decline the paper cites.

Interpretation guidance: annual-coefficient IO with no inventory dynamics should
*under*-predict a transient monthly peak (annual averaging smooths the spike). Report the
model/observed ratio for each method and use it as an interpretive calibration in the paper's
text — do **not** silently rescale the US–Russia results by it; if a scaled variant is shown,
label it explicitly as such.

## 7. Aggregation and reporting

- **Direct component:** US+RUS industrial gross output share of the world total in the table
  (call it `σ`, at MER), times `s`. Report `σ` explicitly — the paper currently uses a
  PPP-based 12.5% and needs the MER figure for a valuation-choice paragraph (expect roughly
  15–20%).
- **Global decline** = direct + RoW spillover × (1 − σ). Never double count: the spillover
  term sums only non-US/RUS sectors.
- Deliver one tidy CSV: columns {year, method, s, lambda (Method 1 only, else blank),
  RoW_decline_pct, sigma_MER, global_decline_pct}, plus the Thailand backtest row(s), plus
  the top-affected-sectors listing from Method 2.

## 8. Known pitfalls

- Inventory-change final-demand entries can be negative; keep them as-is.
- Do not conflate the value-added row with the taxes-less-subsidies row when building `w`.
- If the release includes adjustment columns (e.g., residents' purchases abroad), follow the
  ReadMe so nothing is counted twice.
- Units are current USD at market exchange rates — every share derived here is MER.
- ROW is an aggregate; its internal composition is unknowable. Fine for this purpose.
- State clearly in the write-up what the framework omits: inventories and dynamics, price
  responses, finance/insurance/shipping freezes, and any post-2019 re-routing of trade. The
  first and last omissions cut in opposite directions; the finance channel makes all
  estimates conservative.


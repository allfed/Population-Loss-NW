# MRIO cross-check

Multi-regional input-output cross-check of the paper's judgment that rest-of-world (RoW)
industrial output would decline 10–20% following ~55% industrial output losses in the US
and Russia. See `mrio-crosscheck-spec.md` for the full brief and `methods-writeup.md` for
results and interpretation.

## Files

| file | what |
|---|---|
| `mrio_crosscheck.py` | library: ICIO loader, baseline, Method 1 (mixed Leontief), Method 2 (mixed Ghosh; `method2_va_shock` VA-shock sensitivity), shares |
| `run_crosscheck.py` | runner: produces every CSV in `outputs/` and prints a summary |
| `methods-writeup.md` | ½–1 page methods note with equations, results, caveats (for the paper) |
| `outputs/results.csv` | tidy results {year, method, s, lambda, RoW_decline_pct, sigma_MER, global_decline_pct} + Thailand rows |
| `outputs/shares.csv` | US+RU industrial share of world industry (σ), MER, GO and VA |
| `outputs/sector_mapping.csv` | ISIC industry code → description, industrial flags |
| `outputs/method2_top_sectors.csv` | most-affected RoW sectors (Ghosh, central case) |
| `outputs/sensitivity_construction.csv` | industrial definition incl./excl. construction (F) |

## Data (not committed — see `data/.gitignore`)

OECD ICIO 2025 edition, SML CSV format, from <https://oe.cd/icio>. Expected in
`data/icio/`:

- `2016_SML.csv` … `2022_SML.csv` (from `2016-2022_SML.zip`) — **present**
- `ReadMe_ICIO_small.xlsx` — **present**
- `2011_SML.csv` (from the earlier-period SML zip, e.g. `2011-2015_SML.zip`) — **needed for
  the Thailand 2011 backtest**; until present the runner uses the 2016 table as a labelled
  proxy.

## Reproduce

```bash
cd MRIO
python3 run_crosscheck.py      # ~90 s; writes outputs/, prints summary
```

Requires numpy, scipy, pandas, openpyxl (all in the repo `environment.yml`).

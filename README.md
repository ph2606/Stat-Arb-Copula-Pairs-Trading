# The profitability of pairs trading strategies using copula methods

**Panagiotis Housos · ph2606**  
NYU · Statistical Arbitrage · Fall 2026

[Read the PDF report](report/Panagiotis_Housos_Copula_Pairs_Trading.pdf) · [Open the executed notebook](copula_pairs_trading.ipynb)

GitHub repository: [ph2606/Stat-Arb-Copula-Pairs-Trading](https://github.com/ph2606/Stat-Arb-Copula-Pairs-Trading).

This homework reproduces the first supplied paper's mixed-copula procedure on **AAPL–GOOG, IBM–SPY and DIA–SPY**, and implements the second paper's marked-to-market PnL and committed/employed-capital conventions. Yahoo Finance daily observations support 40 nonoverlapping six-month trading windows, **2006–2025**, each preceded by 12 calendar months of formation data.

The primary selected-copula strategy loses **4.95% per year as annualized arithmetic PnL / initial committed capital**, including 5 bps per leg per fill. Its gross result is also negative. An account that instead sizes each new trade from current equity loses **63.07% cumulatively**, with **−4.87% CAGR**. These negative findings are retained; better formation likelihood does not establish a profitable strategy.

## Read the capital convention carefully

The primary study uses a fixed $100,000 long-side stake per pair, initially $200,000 gross exposure, and $300,000 total committed capital. Units remain fixed during each trade. Profits do not enlarge the next fixed-stake trade. Cumulative PnL is therefore added, not compounded.

The selected strategy's **AAPL–GOOG sleeve exhausts its initial capital on 13 August 2013**. Its aggregate uncapped account first becomes nonpositive on **6 September 2024**. Results beyond those dates are hypothetical constant-notional research PnL requiring added capital. Drawdowns exceeding 100% and full-sample fixed-stake Sharpe ratios are diagnostic statistics, not achievable account performance.

`reinvested_performance.csv` is a separate sizing extension: a new trade's long stake equals that sleeve's current equity, with fixed units until exit. All modeled daily balances remain positive in this sample. Portfolio sleeve weights drift; there is no daily rebalancing across sleeves. Positive modeled equity does not establish broker-margin compliance. Cash interest, stock-lending availability, and maintenance-margin rules are not modeled.

## Assignment coverage

| Requirement / example | Implementation |
|---|---|
| Read both papers | Detailed page-referenced notes in `references/paper1_notes.md` and `references/paper2_notes.md`; comparison and equations in the report |
| Reproduce one paper | Paper 1: ARMA–GARCH marginals; empirical residual CDF; five single copulas and two mixtures; conditional ranks; accumulated flags; 12/6-month rolling procedure |
| Suggested pairs | All three, with independent capital slots and shared SPY exposures left separate |
| Copula examples | Gaussian, Student-t, Clayton, Frank, Gumbel, CFG and CtG; individual-family strategies plus likelihood-selected, best-single and BIC controls |
| Benchmarks | Distance and screened Engle–Granger cointegration, following Paper 2's operational dollar sizing |
| PnL estimation | Fixed entry units, daily mark-to-market, drifted weights, next-close execution, entry/exit fees, final liquidation, monthly RCC/REC |
| Robustness | 11 entry thresholds; both/either exits; six fee levels; three borrow rates; five subperiods; block-bootstrap paired comparisons; equity-based sizing |
| Reproducibility | Cached observations and fitted probabilities, hashes, pinned dependencies, numerical/accounting tests, executed notebook, LaTeX source and PDF |

## Methods and explicit choices

- Fit four ARMA orders, `p,q in {0,1}`, jointly with GARCH(1,1), by Gaussian quasi-maximum likelihood on percentage log returns. AIC chooses the marginal order. Paper 1 leaves the innovation likelihood and order-selection convention open.
- Transform standardized formation residuals with right-rank `rank/(n+1)`. Freeze parameters and the formation empirical CDF; update the future recursion causally. Clip future ranks to `[1/(n+1), n/(n+1)]`.
- Fit all seven copulas by bounded multistart maximum pseudo-likelihood. Mixtures jointly estimate component parameters and simplex weights, retaining exact nested candidates. Only converged starts can win. Bounds, parameter counts, convergence and likelihoods are recorded.
- Select maximum formation likelihood for the main method, minimum BIC for the BIC control, and maximum likelihood over the five single families for best-single. Trading-period PnL never chooses a fitted family.
- Enter only when both cumulative flags exceed opposing ±0.20 barriers. Main exit requires either flag to reach/cross zero, following Paper 1's numbered algorithm. Its conflicting “both” wording is tested as simultaneous opposite-side signs. Flags reset only at window starts, with no stop-loss added.
- Observe signals at a close and execute at the following close. Terminal liquidation is predetermined; terminal-day new entries are suppressed. This differs from Paper 1's same-close assumption.
- Distance rebases normalized prices at formation and again at trading start. Cointegration uses normalized second price on first, intercept, AIC lag selection, 5% Engle–Granger eligibility and positive beta. Rejected slots remain cash. The dollar allocation follows Paper 2, not an exact beta-share hedge.
- Base cost is 5 bps per actual dollar traded per leg fill: about 20 bps per complete equal-dollar pair at unchanged prices. Borrow scenarios charge prior-close short notional at annual rate/252 per held interval. Adjusted marks already include distributions; no dividends are added again.

Paper 1's six-month window advance is used. Paper 2's six overlapping monthly cohorts, full CRSP universe, rotated copula families and parametric marginal models are not reconstructed. No point-in-time constituent selection, value weighting, delisting sample, historical cost schedule or factor-alpha replication is claimed. This is a methodological reproduction on the assignment's prescribed pairs, not a replication of the original published return numbers.

## Data and source integrity

Five Yahoo Finance CSVs contain 5,305 common daily dates from 1 December 2004 through 31 December 2025. The evaluation has 5,031 dates from 3 January 2006. Downloads retain unadjusted OHLC fields, adjusted close and corporate-action fields; calculations use adjusted close as a synthetic total-return mark. No missing prices are filled and no provider series are spliced.

`data_manifest.json` records request bounds, provider/version, retrieval time, coverage, event counts and SHA-256 for each file. Analysis checks every input hash. Each of the 40 fitting caches records the input/algorithm fingerprint, training and evaluation dates, marginals, all copula parameters and conditional probabilities. Dependencies are pinned, but later numerical-library versions or vendor revisions can produce different estimates. The fit fingerprint covers the raw data, marginal/copula modules and declared fitting design; it is not a cryptographic guarantee for every surrounding script.

Sources are the two user-supplied PDFs and [Yahoo historical data](https://finance.yahoo.com/quote/SPY/history/) through [yfinance](https://ranaroussi.github.io/yfinance/reference/yfinance.stock.html). Paper 2 is also [available from its author](https://randlow.github.io/2016_QF_PairsTrading.pdf), DOI [10.1080/14697688.2016.1164337](https://doi.org/10.1080/14697688.2016.1164337). The original PDFs are not republished in the project archive.

## Reproduce

The GitHub repository contains the report, executed notebook, source, tests, figures, data-source manifest and summary results. Raw market data, fitted-window caches, detailed daily/trade ledgers and installed dependencies remain in the local submission package. A fresh clone must run the download, fitting and analysis commands below to recreate those local files before rerunning the notebook. Saved notebook outputs and the PDF can be read immediately.

Python 3.12 was used. From this project directory, create and activate a virtual environment, then run:

```sh
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/fit_models.py --workers 4
python scripts/run_analysis.py
python -m pytest -q
python scripts/build_figures.py
python scripts/build_notebook.py --execute
python scripts/build_report.py
```

`download_data.py` and `fit_models.py` reuse verified caches. `--refresh` on the downloader deliberately fetches a new provider sample and replaces its manifest. Changed input or model hashes invalidate fitted caches; rerun `fit_models.py` before analysis. The downloaded historical dates are fixed rather than moved forward automatically. The notebook reruns analysis from verified fitted caches and displays the resulting tables and figures; it does not silently refit hundreds of models on every run.

Compile twice from `report/`, using MiKTeX or TeX Live:

```sh
xelatex -interaction=nonstopmode -halt-on-error -jobname=Panagiotis_Housos_Copula_Pairs_Trading report.tex
xelatex -interaction=nonstopmode -halt-on-error -jobname=Panagiotis_Housos_Copula_Pairs_Trading report.tex
```

Once the downloaded prices and fitting caches are available, analysis and tests need no network access. These inputs are included in the local submission archive but excluded from GitHub. Notebook building requires a functioning `python3` Jupyter kernel. The local `.packages` folder was an environment convenience, is excluded from the archive, and is unnecessary in a properly installed virtual environment.

## Main outputs

| File | Contents |
|---|---|
| `results/performance.csv` | Gross/net diagnostic statistics for 12 methods, three pairs and their portfolio |
| `results/daily_ledger.csv` | Prior/next quantities, signals/executions, daily PnL, costs and events |
| `results/trades.csv` | Complete entries/exits, reason, duration, sizing and reconciled PnL |
| `results/copula_fits.csv` | 840 family fits with likelihood, AIC/BIC, parameters and selection |
| `results/marginal_fits.csv` | 200 selected marginal models; full candidate diagnostics in caches |
| `results/monthly_returns.csv` | Monthly RCC, REC and active-pair counts |
| `results/cost_sensitivity.csv` | Transaction-cost and borrowing scenarios |
| `results/sensitivity_windows.csv` | Pair/window results for each threshold and exit interpretation |
| `results/subperiods.csv` | Predetermined reporting periods |
| `results/reinvested_daily.csv` | Supplementary equity-at-entry sleeve and portfolio accounts |
| `results/reinvested_performance.csv` | Funded-account total return, CAGR, Sharpe and drawdown |
| `results/capital_diagnostics.json` | Dates of fixed-stake account exhaustion |
| `results/audit.json` | Coverage, convergence, reconciliation and paired bootstrap intervals |

HAC inference uses six monthly lags. Bootstrap comparisons use 5,000 circular six-month block resamples with seed 2606, rather than Paper 1's automatic stationary-bootstrap procedure. Results do not correct for searching across many strategy variants. The baseline threshold and model-selection rules remain unchanged after inspecting their PnL.

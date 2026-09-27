# Paper 2: methodology and replication notes

Source: Hossein Rad, Rand Kwong Yew Low, and Robert Faff (2016), *The profitability of pairs trading strategies: distance, cointegration and copula methods*, DOI 10.1080/14697688.2016.1164337. Citations below use **PDF page numbers**; printed journal pages are one lower because the supplied PDF has a cover page. The original PDF pages 8–9 were visually checked to resolve extraction of the equations and cost paragraph.

## Experimental design

- Data: CRSP daily US shares, 1 July 1962–31 December 2014, 23,616 distinct stocks, 13,216 dates. Ordinary shares only (CRSP share codes 10 and 11); remove bottom market-cap decile within each formation period, sub-$1 shares, and stocks with at least one day of no volume during formation (PDF p.5, §3).
- **12 calendar months formation; 6 calendar months trading; launch a new cohort every month**, giving six overlapping portfolios. Fitted parameters remain fixed during their associated trading period (PDF p.5, §4; PDF p.6 footnote specifies calendar months).
- Normalize total-return indices, adjusted for dividends and corporate actions, to one at formation start. Rank candidate pairs by sum of squared differences (SSD). Select 20 pairs; the same stock may belong to several different pairs (PDF p.6, §4.1).
- Copula and distance use SSD ranking; cointegration scans the ranking until 20 pairs also satisfy its cointegration criterion. Therefore the three methods need not trade exactly the same pair sets (PDF pp.6–7).

## Distance benchmark

At formation, store the sample spread standard deviation. At trading start **rebase each price to one again**. Open when the new normalized-price spread reaches ±2 formation standard deviations, buy the lower normalized-price asset and short the higher one. Close when the spread returns to/crosses zero; multiple round trips are possible (PDF p.6, §4.1). No stop loss is specified. All outstanding trades terminate at the cohort's final date (explicit discussion PDF p.12, §5.3). The paper does not specify close-to-close signal/execution latency.

## Cointegration benchmark

- Use cumulative return series from formation and the two-step Engle–Granger procedure: fit OLS, assess residual stationarity/cointegration, estimate beta. The significance level, lag-selection details, orientation selection, and intercept convention are not supplied (PDF p.7, §4.2.2).
- Spread is `X2 - beta*X1`. Freeze formation residual mean and standard deviation, and trade standardized spread `(spread - mean)/std` at ±2 and close at zero (PDF pp.6–7, equations 1–6).
- The operational dollar allocation explicitly says: when the spread is below −2, long $1 of stock 2 and short beta dollars of stock 1; when above +2, long $1 of stock 1 and short `1/beta` dollars of stock 2 (PDF p.7). This maintains $1 long exposure but not dollar neutrality. The earlier derivation uses one **share** and beta **shares**, so a literal dollar implementation is not identical to the theoretical share-hedged spread unless the normalization makes the distinction immaterial. State the implemented convention.
- No universal mean reversion follows merely from a small-sample rejected unit-root test; no test must use future trading data. With only three nominated pairs there may be no eligible cointegrated pair. Leave rejected slots in cash or explicitly define the accepted-pair denominator; do not silently force eligibility to obtain trades.

## Copula fitting and signals

Inference for margins (IFM), with formation daily returns, is used in two stages (PDF pp.7–8, §4.3.2):

1. Fit each asset's marginal independently, considering extreme value, generalized extreme value, logistic, and normal distributions. Transform daily returns with their frozen fitted CDFs. The paper does not explicitly specify log versus simple returns, EV orientation, or a unique marginal-selection rule.
2. Fit Clayton, 180-degree rotated/survival Clayton, Gumbel, 180-degree rotated/survival Gumbel, and Student-t copulas by maximum likelihood. Compute information criteria. The paper says choose the **highest** AIC/BIC; with the conventional definitions `AIC=2k-2logL`, `BIC=k*log(n)-2logL`, these must be **minimized**. A criterion or tie-breaking rule when AIC and BIC disagree is not given; choose one primary criterion and disclose it.

For daily uniforms `(u1,u2)`,

`h1 = partial C(u1,u2)/partial u2 = P(U1 <= u1 | U2=u2)`

`h2 = partial C(u1,u2)/partial u1 = P(U2 <= u2 | U1=u1)`.

Set `m1=h1-0.5`, `m2=h2-0.5` and accumulate `M1_t=M1_(t-1)+m1_t` and similarly for M2. Both flags start at zero at the **beginning of trading**, not at formation. Enter short 1/long 2 when `M1>0.5 and M2<-0.5`; enter long 1/short 2 for the reverse. Close when **both** accumulated flags return to zero (PDF p.8, equations 12–13). These are conditional ranks of contemporaneous **returns**, not calibrated probabilities that future prices fall or rise, despite loose price/probability language in the paper.

PDF p.9, Table 2 has the conditional distributions. In particular:

`h1_student = t_(nu+1)((x1-rho*x2)/sqrt((nu+x2^2)*(1-rho^2)/(nu+1)))`, with `xi=t_nu^(-1)(ui)`.

`h1_clayton = u2^(-theta-1)*(u1^(-theta)+u2^(-theta)-1)^(-1/theta-1)`.

For a survival copula, `h1_survival(u1,u2)=1-h1_base(1-u1,1-u2)`. Obtain h2 by interchanging the two arguments for these exchangeable families. Gumbel requires `theta >= 1`, despite Table 2's printed `theta>0`.

Unresolved trading details must be documented: exact real-valued equality to zero is almost impossible; interpret return-to-zero as sign crossing. Does each flag merely need to have crossed sometime, or must both be on the opposite side at the same close? Are flags reset following convergence? The paper only explicitly mandates resetting at cohort start. A reproducible primary convention is simultaneous favorable signs, no reset after exits, and re-entry only after a later entry signal. A latched-crossing or reset implementation is a declared variation, not an exact textual replication.

## PnL and transaction costs

Distance and copula open **$1 long and $1 short**. The paper reports return using **$1 as the denominator**, even though gross exposure is $2. Cointegration also uses $1 as denominator because its long allocation is $1 (PDF p.9, continuation of §4.5). This is a long-side capital convention, not a margin rule. Dividing by gross $2 gives exactly half the paper-style return for equal-dollar pairs before considering different financing assumptions.

For fixed entry units, long A/short B with stake K and entry date e:

`qA=K/P_A,e`, `qB=-K/P_B,e`

`gross_PnL_t=qA*(P_A,t-P_A,t-1)+qB*(P_B,t-P_B,t-1)`

`trade_return_gross=[P_A,exit/P_A,entry-1]-[P_B,exit/P_B,entry-1]`.

This realizes the paper's marked-to-market accounting without introducing unmentioned daily rebalancing. Using `position_(t-1)*(rA_t-rB_t)` every day with constant ±1 weights rebalances both legs daily and is a different strategy. On adjusted-close total-return marks, the units are synthetic total-return units rather than literal shares; dividends must not be added again.

Monthly equations (14)–(15), PDF p.8:

`REC_m = sum_i marked_to_market_pair_return_i,m / n_m`

`RCC_m = sum_i marked_to_market_pair_return_i,m / NP`, where paper `NP=20`.

Here `n_m` counts pairs traded/held during that month, including carry-in positions, and the numerator contains both realized and unrealized PnL for that month. Aggregate trade returns only at exit would move PnL into the wrong months. Average the six cohort-level monthly returns equally. No interest accrues to idle capital (PDF pp.8–9). Define the no-active-pair REC as zero or missing explicitly. The source calls these excess returns, but it does not give a separate risk-free deduction in the two equations; do not imply an independently verified cash financing model.

Cost model (PDF p.8, §4.4): commissions decline from 70 bps in 1962 to 9 bps in recent years; market-impact estimates are 30 bps for 1962–1988 and 20 bps for 1989 onward; no explicit stock-borrow cost. The paragraph says these costs are **doubled** because a complete pairs trade contains two round-trip trades. Read literally, recent costs are `2*(9+20)=58 bps` per complete $1/$1 pair. The source refers to Do and Faff (2012), §3, for the detailed schedule; it does not supply that schedule here. Do not invent a linear annual interpolation or silently charge all four individual fills 29 bps (which produces 116 bps). A modern per-fill fee scenario is a useful disclosed alternative: `cost_t=c*sum_j |delta q_j|*P_j,t`. At flat prices 5 bps per fill implies 20 bps per complete equal-dollar pair. A 14.5 bps-per-fill scenario gives 58 bps only when entry/exit notionals remain unchanged.

## Reported empirical comparisons suitable for a results table

- PDF p.10, Table 3: mean monthly REC before costs is 0.91% distance, 0.85% cointegration, 0.43% copula; after costs 0.38%, 0.33%, 0.05%. These are published CRSP-universe results, not targets that a three-pair modern sample must reproduce.
- PDF p.9, Table 1: Student-t is selected 61.64% of the time; logistic marginals 86.14%. Cite these exact table values rather than rounded/inconsistent introductory prose.
- PDF p.13, Table 5: converged trades are 62.53% distance, 61.35% cointegration, 39.98% copula; mean days open among converged trades 21.15, 22.65, 26.30. Classification is signal convergence versus forced expiry, not profit versus loss.
- PDF pp.14 and 16, Figure 5 and Table 7: sensitivity to copula entry thresholds 0.2 through 0.8, versus base 0.5; report trade count, convergence duration, and monthly return. Avoid optimizing the threshold on the final test sample.
- PDF pp.10–11: monthly Sharpe ratios in the paper are **not annualized** (roughly mean/std). A reproduction can report both this and annualized `sqrt(12)*mean/std` with unambiguous names.

## Recommended bounded homework reproduction

Use the user-provided AAPL–GOOG, IBM–SPY and DIA–SPY pairs with a cached, dated, documented adjusted-close dataset. Label this a small-universe methodological reproduction. ETFs are outside paper 2's original ordinary-share sample; preselecting present-day pairs has selection/survivorship limitations. Apply 12-month formation and 6-month trading, start new cohorts monthly, preserve parameters within each cohort, and distinguish a full six-sleeve evaluation period from startup warmup.

Use simple returns consistently for the paper-2 marginal fits; use BIC as primary and report AIC as a diagnostic; retain all five copula families and all four marginal candidates if stable fitting permits. Clamp CDF inputs slightly inside (0,1) for numerical stability and record fit failures. Use the distance benchmark and an optional screened cointegration benchmark. With three preselected slots, set committed denominator to three; rejected cointegration slots stay in cash as an explicit adaptation because no broader universe exists to replace them.

Calculate signals at a close and execute at the next close, so the new position earns only subsequent returns. This conservative execution lag is a declared implementation convention. Fix units until an exit, debit transaction costs on actual traded notional, force-close at the predefined cohort end, and suppress terminal-day new entries. Report gross and net PnL, paper-style REC/RCC, trade convergence counts, holding times, monthly Sharpe, drawdown, and a small preset fee/entry-threshold sensitivity grid. Include hand-computed accounting tests, lag-causality checks, and a terminal-liquidation test. No claimed full replication of the CRSP 1962–2014 coefficients or economic rankings is justified by this sample.

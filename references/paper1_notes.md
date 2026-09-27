# Paper 1: reading and replication notes

Source: supplied `Mixed_Copula_Pairs_Trading_on_the_S_P_500.pdf`, read in full (32 PDF pages, including the ResearchGate wrapper, tables, appendix and references). The manuscript title is *Mixed Copula Pairs Trading Strategy on the S&P 500*, by Fernando A. B. Sabino da Silva, Flavio A. Ziegelmann and João F. Caldeira. The wrapper labels it an April 2017 working paper and records upload on 22 February 2019. Page references below mean physical PDF pages; manuscript page numbers are one lower from PDF page 2 onward.

## Data, selection and windows

- PDF pp. 6, 11–12: adjusted daily closing prices, including dividends, splits and other corporate actions. Bloomberg supplied 1,100 historical S&P 500 constituent series over 2 July 1990–31 December 2015, 6,426 trading days. Selection uses membership during each formation period, not today's constituent list. The reported trading sample is July 1991–December 2015 (6,173 observations and 49 half-year trading windows).
- PDF p. 6: 12 calendar months of formation, January–December or July–June; trade the following six months and advance by six months. Normalize each price to one at formation start. Rank distinct pairs by sum of squared distances between normalized prices; study top 5, 10, 15, 20, 25, 30 and 35. The printed `n(n+1)/2` candidate count appears to be a typo: distinct unordered stock pairs number `n(n-1)/2`.
- PDF p. 6 footnote 5 says missing values were interpolated. A replication that drops invalid paired observations should disclose this deliberate difference. Interpolating with future prices is inappropriate for a causal trading experiment.
- PDF p. 11 assumes executions at that day's close. A next-close execution convention is more conservative and differs from the paper.

## Distance comparison

PDF p. 6: open when normalized-price spread departs from its formation mean by at least two formation standard deviations; short the outperformer and buy the underperformer. Close when normalized prices cross; force close at the six-month end. More than one completed trade per pair per window is permitted. Distinguish crossing the normalized prices (zero spread) from crossing a fitted nonzero mean when implementing this wording.

## Marginal filtering and pseudo-observations

PDF pp. 9–10, equation (11): fit a separate ARMA(p,q)-GARCH(1,1) marginal to each stock's daily formation returns, considering ARMA orders through (1,1), i.e. p,q in {0,1}. Form standardized residuals

`z_t = (r_t - fitted_conditional_mean_t) / fitted_conditional_sd_t`.

The empirical residual CDF supplies `u_t = n/(n+1) * F_n(z_t)`, equivalently rank/(n+1) for distinct observations. The scaling avoids the upper boundary; handling new test residuals below the training minimum still needs an explicit lower clipping convention. Freeze the formation EDF in trading, forecast conditional mean/variance causally and standardize each newly observed return. Do not fit a CDF or volatility model using the trading sample.

The paper does not specify the innovation likelihood, ARMA selection criterion, estimation optimizer/bounds, tie handling, or variance initialization. Gaussian quasi-maximum likelihood and AIC among the four orders are defensible explicit implementation choices, not details documented in the manuscript.

## Copula candidates and fitting

PDF pp. 8–12: Sklar decomposition separates marginals and dependence. Fit seven dependence candidates to formation pseudo-observations:

1. Gaussian.
2. Student t.
3. Clayton.
4. Frank.
5. Gumbel.
6. CFG: `pi_C C_Clayton + pi_F C_Frank + (1-pi_C-pi_F) C_Gumbel`.
7. CtG: `pi_C C_Clayton + pi_t C_Student_t + (1-pi_C-pi_t) C_Gumbel`.

All weights are nonnegative and sum to one. CFG has the three component dependence parameters plus two independent weights. CtG replaces Frank with t-copula correlation and degrees of freedom, hence has four component parameters plus two independent weights. Equations (14) and (15) on PDF p. 10 define these mixtures.

Minimize the negative log likelihood of the **weighted sum of copula densities**, `-sum_t log(sum_j pi_j*c_j(u_t,v_t))`; it is not a weighted sum of component log likelihoods. Select the candidate with smallest formation negative log likelihood (PDF p. 12). The paper does not penalize the extra mixture parameters. Because the mixtures contain single component limits, better in-sample likelihood is partly mechanical; it does not establish superior out-of-sample prediction or profits. AIC/BIC are useful separately labeled diagnostics or robustness alternatives, but using them to select the baseline changes the paper's rule.

## Conditional probabilities, cumulative flags and positions

PDF pp. 8–11: calculate `h_X = dC(u,v)/dv = P(U<=u | V=v)` and `h_Y = dC(u,v)/du = P(V<=v | U=u)`. Mixture derivatives are the same weighted combinations of component derivatives. The second conditional probability in printed equation (10) repeats the first probability in error; implement the correctly indexed derivative.

Initialize both cumulative flags at zero at the beginning of each six-month trading period. Update `M_X,t = M_X,t-1 + h_X,t - 0.5` and likewise for Y. These sums are not guaranteed stationary or mean reverting; the paper itself notes random-walk, divergent and convergent possibilities on PDF p. 9.

PDF p. 11, numbered step 4:

- Flat and `M_X > +0.2` **and** `M_Y < -0.2`: short X, buy Y.
- Flat and `M_X < -0.2` **and** `M_Y > +0.2`: buy X, short Y.
- The paper forbids entry if X or Y already has a position. With overlapping assigned pairs (IBM/SPY and DIA/SPY), separate pair sleeves are an adaptation unless cross-pair conflicts are explicitly prevented.

**Exit inconsistency:** PDF p. 9 prose says close when **both** cumulative flags return to zero. PDF p. 11 numbered step 5 says close if `M_X` reaches zero **or** `M_Y` reaches zero, or at window end. The latter is the more explicit algorithm; an implementation should disclose which convention it follows and interpret reaching zero as crossing it in the convergence direction, since exact floating-point equality is unlikely. The paper states only a reset at the start of the trading window, not an explicit reset after every exit. If flags are reset after exits, label that additional convention. There is no stated stop-loss flag threshold in this paper's numbered algorithm.

PDF pp. 9, 12–13: entry sensitivity is 0.05, 0.10, ..., 0.55 with equal negative counterparts. The authors choose ±0.2 partly to make trade counts/costs comparable to distance top-five portfolios; this is a choice using the historical experiment, not a universally optimal threshold. Hold the published value fixed before evaluating new data; do not select the best homework backtest afterward.

## P&L, costs and aggregation

PDF p. 6, equation (1): daily pair contribution is `w_L,t*r_L,t - w_S,t*r_S,t`, with both weights initially one and evolving by the preceding leg return, `w_i,t = w_i,t-1*(1+r_i,t-1)`. This is fixed-entry holdings/value drift; resetting both weights to one every day silently changes the strategy into daily dollar rebalancing. The equation uses one unit on each leg, gross exposure two units. Returns should state whether the denominator is one unit of committed capital or two units of gross notional.

PDF pp. 11–12: trading cost proxy is 20 bps **per round-trip pair trade**, consistent with 5 bps per leg per one-way transaction (four leg transactions). Applying 20 bps separately to all four orders overcharges relative to the paper. Proportional turnover accounting will make actual round-trip cost vary with the exit leg values; disclose it if used. Idle committed capital earns zero. Borrow charges, financing, market impact and execution constraints are not fully modeled.

PDF p. 11: committed-capital portfolios allocate equal amounts to all selected pairs, including inactive pairs. Fully invested portfolios divide across open pairs. The exact timing of this denominator must be documented; changing the active-pair count contemporaneously can hide capital reallocations. An exposure-scaled summary alone is not a fully specified self-financing execution strategy.

## Results and reproducible examples

- PDF p. 12, Figure 1: model selection frequency (top 1–5/6–20/21–35). Mixtures selected 92.3%, 91.7%, 89.3%; CtG dominates. These are historical fit frequencies, not success probabilities.
- PDF p. 13, Figures 2–3: threshold sensitivity of annualized mean excess return and Sharpe, CC and FI.
- PDF p. 15, Table 1: top-five net CC annualized mean excess return 3.68% mixed versus 2.30% distance; Sharpe 0.58 versus 0.28; Sortino 1.00 versus 0.52; annual volatility 6.30% versus 8.23%. These are the paper's reported outcomes, not values expected from three chosen pairs. Top-20 and top-35 CC mean returns favor distance (2.86%, 2.78%) over mixed (1.08%, 0.71%), so the conclusion is not uniform superiority.
- PDF pp. 14–16: HAC t-statistics use Newey–West six lags. Stationary bootstrap comparisons use 10,000 replications and automatic block length. The printed p-value equation (17) appears to count the wrong tail; do not copy it without validation.
- PDF p. 17, Figure 4: cumulative investment of one dollar. PDF pp. 19–20, Figures 5–6: five-year rolling Sharpe and its estimated density.
- PDF p. 21, Table 2: entry deviation, number of trades, round trips, holding-time mean/median and standard deviations. Top-five distance/mixed total opens 352/348 and mean holding days 50.70/37.70.
- PDF pp. 21–23, 29: five Fama–French factors plus momentum and long-term reversal; short-term reversal dropped by BIC. HAC six lags. Factor-adjusted alpha is not proof of risk-free arbitrage.
- PDF pp. 23–27: fixed subperiods 1991–95, 1996–2000, 2001–05, 2006–10 and 2011–15; return, Sharpe and copula-selection comparisons. Distance beats mixed in some subperiods.
- PDF pp. 27–28: conclusions emphasize top-five comparison with similar trade counts and warn that fixed distance and cumulative-flag thresholds produce different trade frequencies.

## Honest scope for this assignment

The user explicitly permits AAPL/GOOG, IBM/SPY and DIA/SPY. Implementing these named pairs with public daily data is a methodological reproduction, not an exact reproduction of a 1,100-stock historical-membership Bloomberg backtest. No top-five/top-twenty portfolio selection claim is appropriate with only three assigned pairs. It is possible to reproduce model-frequency charts, threshold and cost sensitivity, summary tables, cumulative P&L, rolling Sharpe, holding-time statistics and chronological subperiods on the smaller universe. Record all differences in market sample, pair selection, marginal estimation, execution lag, flags, cost denominators and capital treatment.

Two technical cautions in the manuscript should not propagate: (i) PDF p. 8 footnote 7 claims price and return copulas are identical by monotonic invariance, but taking intertemporal returns is not a componentwise monotone transform of contemporaneous prices; (ii) a conditional CDF of 0.5 means the observed return is at its modeled conditional median, not proof that a stock has its fundamental fair value.

## Prior-homework context inspected

`ASSIGNMENT1` is empty. `Volatility-Signature/README.md`, the report's LaTeX source, manifests, and `final_project/README.md`, proposal and technical plan were inspected. No file containing verbatim earlier user instructions was found. Therefore the following are observed delivery precedents, not recovered conversation memory: author **Panagiotis Housos · ph2606**; NYU Statistical Arbitrage, Fall 2026; an executed notebook plus modular Python and an independently readable LaTeX PDF; explicit source/date coverage and data hashes; causal positions/cash-flow accounting; gross/net costs and sensitivity checks; clear prose that accepts negative results; and a README with reproducibility commands. The prior report uses Cambria/Calibri, navy/teal graphics and page headers. Prior artifacts exclude raw vendor data and supplied course documents from GitHub; this does not independently authorize publishing the new assignment.

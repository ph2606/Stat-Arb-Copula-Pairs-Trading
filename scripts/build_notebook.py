"""Build the worked homework notebook; optionally execute it from project root.

Usage: python scripts/build_notebook.py --execute
The local .packages directory is an environment convenience, not a portable
dependency. A clean environment installed from requirements.txt is sufficient.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
sys.path.insert(0, str(ROOT))

import nbformat
from nbclient import NotebookClient


def markdown(source):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nbformat.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook():
    cells = [
        markdown(r"""
        # The profitability of pairs trading strategies using copula methods

        **Panagiotis Housos · ph2606**  
        NYU · Statistical Arbitrage · Fall 2026

        This is a worked, reproducible study of **AAPL–GOOG, IBM–SPY and DIA–SPY**.
        The main method reproduces the mixed-copula framework of Sabino da Silva,
        Ziegelmann and Caldeira, *Mixed Copula Pairs Trading Strategy on the S&P 500*
        (supplied working paper; hereafter paper 1). Profit measurement follows
        the fixed-entry long/short and marked-to-market conventions in Rad, Low
        and Faff (2016), *The profitability of pairs trading strategies: distance,
        cointegration and copula methods* (paper 2).

        Both supplied papers were read. Detailed page-specific notes are in
        `references/paper1_notes.md` and `references/paper2_notes.md`. The new
        experiment uses 2006–2025 trading observations, 12-month formation and
        nonoverlapping six-month trading windows. It is a **methodological
        reproduction on the three assigned pairs**, not a reconstruction of the
        papers' Bloomberg/CRSP universes or published coefficients. ETFs and
        today's nominated pairs also differ from those historical universes.

        The primary published-style calculation is an **uncapped fixed-stake
        P&L diagnostic**. It can exhaust initial capital and is then not a
        feasible funded trading history. A supplementary account scales each
        new trade to surviving equity and reports conventional compounded wealth.

        This notebook verifies data and fitted caches, reruns the trading and
        reporting calculations, audits a trade by hand, and displays the results.
        Model refitting is a separate, slower command. No live orders are placed.
        """),
        code("""
        from pathlib import Path
        import json
        import sys
        import numpy as np
        import pandas as pd
        from IPython.display import display, Markdown, Image

        candidates = [Path.cwd(), Path.cwd() / 'Copula-Pairs-Trading', Path.cwd().parent]
        ROOT = next((p.resolve() for p in candidates if (p / 'src' / 'copulas.py').exists()), None)
        if ROOT is None:
            raise RuntimeError('Launch this notebook from the Copula-Pairs-Trading project folder.')
        sys.path.insert(0, str(ROOT))
        RESULTS = ROOT / 'results'
        pd.set_option('display.max_rows', 200)
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
        print('Project:', ROOT.name)
        print('Python:', sys.version.split()[0], '| NumPy:', np.__version__, '| pandas:', pd.__version__)
        """),
        markdown(r"""
        ## 1. Data and causal design

        Yahoo adjusted closes supply total-return price indices. We use decimal
        log returns for the marginal/copula signal models, and price changes of
        the adjusted indices for cash P&L. These indices represent synthetic
        total-return units; dividends are **not** credited a second time.
        Missing observations are not filled. The data manifest records the
        request, provider, timestamps, row counts and SHA-256 hashes.

        Each fit uses only the preceding 12 calendar months. Copula parameters,
        ARMA–GARCH coefficients and the residual empirical CDF remain frozen
        through the following six-month trading period. The GARCH state updates
        recursively as returns arrive. A close's signal executes at the **next
        close**; established quantities earn only subsequent price changes.
        This lag is a disclosed conservative execution choice.
        """),
        code("""
        from scripts.download_data import verify_data
        from scripts.fit_models import load_prices, fingerprint, WINDOWS, PAIRS

        manifest = verify_data()  # Stops if an input hash or requested period differs.
        prices = load_prices()
        coverage = pd.DataFrame(manifest['files']).T
        display(coverage[['rows', 'first_date', 'last_date', 'dividend_events', 'split_events']])
        print('Provider:', manifest['provider'])
        print('Retrieved UTC:', manifest['retrieved_utc'])
        print('OOS windows:', len(WINDOWS), '| Assigned pairs:', PAIRS)
        print('Current model/input fingerprint:', fingerprint())
        """),
        markdown(r"""
        ## 2. Reproduce the experiment

        Paper 1 models marginal returns using ARMA(p,q)–GARCH(1,1), with
        p,q in {0,1}. This implementation fits Gaussian quasi-maximum likelihood
        and uses AIC to select those four marginal orders; the paper leaves
        these two choices unspecified. Standardized formation residuals become
        empirical uniforms, $u_i=\operatorname{rank}(z_i)/(n+1)$, with a frozen
        formation CDF and boundary clipping in trading.

        The dependence candidates are Gaussian, Student t, Clayton, Frank,
        Gumbel, CFG (Clayton–Frank–Gumbel), and CtG (Clayton–t–Gumbel). A mixture
        has nonnegative weights summing to one. Its likelihood is
        $\sum_t\log[\sum_j\pi_j c_j(u_t,v_t)]$.
        **Selected** uses maximum formation log likelihood, as in paper 1.
        **Best-single** restricts this to the five single copulas; **BIC** adds a
        parameter penalty. Every single family and both mixtures are also
        traded separately for comparison. Better in-sample mixture likelihood
        is partly mechanical because mixtures include their component limits.

        The following cell recomputes signals, all ledgers and result tables from
        the verified fitted caches. It does not optimize choices on trading P&L.
        To rebuild the expensive fits after installing `requirements.txt`, run
        `python scripts/fit_models.py --workers 4` first. Original vendor CSVs
        must be present or retrieved with `python scripts/download_data.py`.
        """),
        code("""
        from scripts.run_analysis import run, load_cache

        # Full model refit, if needed, from a terminal in this project:
        # python scripts/fit_models.py --workers 4
        caches = load_cache()  # Verifies every cache against input/model hashes.
        print('Verified fitted windows:', len(caches))
        run()
        """),
        markdown(r"""
        ## 3. Full strategy comparison

        Let $K=100{,}000$ be the fixed initial **long-side** capital per pair.
        Equal-dollar strategies open $K$ long and $K$ short, so gross exposure
        starts at $2K$, while the paper-style return denominator is $K$.
        Quantities remain fixed until exit. Reported daily contributions are
        $\Delta\mathrm{PnL}_t/K$, and cumulative equity is
        $K+\sum_t\Delta\mathrm{PnL}_t$; these contributions are added rather
        than compounded as if each trade reinvested all account wealth.

        All baseline methods pay 5 bps per dollar per leg fill. There is no
        baseline stock-borrow fee or cash interest. The portfolio averages three
        equal committed slots, including idle ones. The SPY sleeves are kept
        separately and trading costs are not netted across pairs, a difference
        from paper 1's restriction on simultaneous positions in the same stock.

        Annual mean and volatility use 252 observations per year. Sharpe uses
        fixed-capital P&L contributions and a zero financing/risk-free convention.
        The primary drawdown is an **algebraic drawdown of uncapped P&L plus
        initial capital**. Losses can exceed 100% because the calculation
        continues to assume a fixed stake after funding has been exhausted.
        Such paths are not feasible funded wealth or ordinary investment returns.
        Distance uses paper 2's rebasing rule. Cointegration admits only pairs
        passing the formation Engle–Granger 5% screen; rejected slots stay in cash.
        """),
        code("""
        performance = pd.read_csv(RESULTS / 'performance.csv')
        fits = pd.read_csv(RESULTS / 'copula_fits.csv')
        margins = pd.read_csv(RESULTS / 'marginal_fits.csv')
        monthly = pd.read_csv(RESULTS / 'monthly_returns.csv', parse_dates=['Date'])
        trades = pd.read_csv(RESULTS / 'trades.csv', parse_dates=['entry_date', 'exit_date'])
        daily = pd.read_csv(RESULTS / 'daily_ledger.csv', parse_dates=['Date'])
        audit = json.loads((RESULTS / 'audit.json').read_text())
        columns = ['method', 'pair', 'basis', 'cumulative_pnl_pct', 'annual_mean_pct',
                   'annual_vol_pct', 'sharpe', 'max_drawdown_pct', 'trades']
        for basis in ['net', 'gross']:
            display(Markdown(f'### {basis.title()} uncapped fixed-stake P&L / initial long-side capital'))
            display(performance.loc[performance.basis.eq(basis), columns]
                    .rename(columns={'max_drawdown_pct': 'algebraic_drawdown_pct'})
                    .reset_index(drop=True))
        """),
        markdown(r"""
        ### Funding diagnosis and an equity-funded comparison

        Continuing a constant dollar stake after the original capital is gone
        would require new external funding. The dates below identify this
        concrete failure in the primary uncapped calculation. Its subsequent
        P&L is a shadow calculation, not the value of a solvent trading account.

        The supplementary account starts each pair with equal capital and
        sizes **each new entry** to that sleeve's then-current surviving equity.
        Quantities remain fixed within a trade and all published signals, costs
        and expiry rules stay unchanged. The three sleeves are not rebalanced
        into each other. Daily percentage returns use preceding account equity,
        so their compounded total and CAGR are meaningful for this comparison.
        Financing, broker margin requirements and intra-trade leverage limits
        remain idealized. Positivity of a simulated balance alone does not
        establish compliance with real margin rules.
        """),
        code("""
        capital_diagnostics = pd.DataFrame(json.loads((RESULTS / 'capital_diagnostics.json').read_text()))
        funded = pd.read_csv(RESULTS / 'reinvested_performance.csv')
        funded_daily = pd.read_csv(RESULTS / 'reinvested_daily.csv', parse_dates=['Date'])
        exhausted = capital_diagnostics.loc[capital_diagnostics.first_nonpositive_equity.notna()]
        display(Markdown('**Funding failure in the uncapped fixed-stake calculation:**'))
        display(exhausted)
        assert funded_daily.equity_per_initial_dollar.gt(0).all()
        display(Markdown('**Supplementary equity-at-entry account — net, funded-return conventions:**'))
        display(funded[['method', 'pair', 'cumulative_return_pct', 'cagr_pct', 'sharpe',
                        'max_drawdown_pct', 'final_equity_per_initial_dollar']])
        selected_capital = capital_diagnostics.loc[capital_diagnostics.method.eq('Selected') &
                                                    capital_diagnostics.pair.eq('Portfolio')].iloc[0]
        if pd.notna(selected_capital.first_nonpositive_equity):
            display(Markdown(f'**The selected fixed-stake portfolio exhausts its original capital on '
                             f'{selected_capital.first_nonpositive_equity}. Its later shadow P&L '
                             'cannot be interpreted as feasible investment wealth.**'))
        """),
        markdown(r"""
        ## 4. What was fitted?

        Count selections across formation windows rather than treating every
        daily probability as a separate model. NLL and BIC can select different
        models because mixture flexibility incurs a BIC penalty. These counts
        describe fitted dependence, not probabilities of earning a profit.
        """),
        code("""
        family_order = ['Gaussian', 'Student-t', 'Clayton', 'Frank', 'Gumbel', 'CFG', 'CtG']
        selection_counts = pd.concat({
            'Log likelihood': fits.loc[fits.selected].groupby(['pair', 'family']).size(),
            'BIC': fits.loc[fits.selected_bic].groupby(['pair', 'family']).size(),
        }, axis=1).fillna(0).astype(int)
        display(selection_counts)
        display(margins.groupby(['symbol', 'p', 'q']).size().rename('formation_windows').to_frame())
        print('All marginal fits converged:', bool(margins.converged.all()))
        print('All copula fits successful:', bool(fits.success.all()))
        print('Copula fits:', len(fits), '| Marginal fits:', len(margins))
        """),
        markdown(r"""
        ## 5. Paper 2's committed and employed capital

        For month $m$, each pair's contribution includes both realized and
        unrealized marked-to-market P&L. With three nominated slots,
        $RCC_m=\sum_i\mathrm{PnL}_{i,m}/(3K)$.
        If $n_m$ distinct pairs were held or traded at any point during that
        month, $REC_m=\sum_i\mathrm{PnL}_{i,m}/(n_mK)$.
        Carry-in positions count even if there is no new entry. A month with no
        activity has RCC zero and REC undefined (NaN). REC is an exposure-scaled
        summary; it does not imply executable monthly capital reallocation.

        Paper 2 launches monthly cohorts and averages six overlapping books.
        Here the window design follows paper 1's nonoverlapping half-years;
        paper 2 contributes P&L sizing and the monthly denominators.
        """),
        code("""
        capital_summary = monthly.groupby('method').agg(
            months=('committed', 'size'), employed_months=('employed', 'count'),
            mean_RCC=('committed', 'mean'), mean_REC_active_months=('employed', 'mean'),
            mean_active_pairs=('active_pairs', 'mean'))
        capital_summary['mean_RCC_bps'] = 1e4 * capital_summary.pop('mean_RCC')
        capital_summary['mean_REC_active_months_bps'] = 1e4 * capital_summary.pop('mean_REC_active_months')
        display(capital_summary)
        display(monthly.loc[monthly.method.eq('Selected')].head(12))
        """),
        markdown(r"""
        ## 6. Audit one completed trade directly

        At entry $e$, $q_L=K/P_{L,e}$ and $q_S=-K/P_{S,e}$. Therefore
        $\mathrm{PnL}_{gross}=q_1(P_{1,x}-P_{1,e})+q_2(P_{2,x}-P_{2,e})$.
        Transaction cost is $c\sum_j|q_j|(P_{j,e}+P_{j,x})$, with
        $c=0.0005$. Four unchanged $K$ fills cost $0.002K$: 20 bps of the
        long-side denominator. Actual exit notionals can differ from entry.
        This check uses a real completed trade, not an invented example.
        """),
        code("""
        selected_trades = trades.loc[trades.method.eq('Selected')].sort_values(['entry_date', 'pair'])
        if selected_trades.empty:
            display(Markdown('No selected-model trades occurred; a completed-trade audit is unavailable.'))
        else:
            t = selected_trades.iloc[0]
            q = np.array([t.q1, t.q2], dtype=float)
            entry_marks = np.array([t.entry_price1, t.entry_price2], dtype=float)
            exit_marks = np.array([t.exit_price1, t.exit_price2], dtype=float)
            manual_gross = float(q @ (exit_marks - entry_marks))
            manual_cost = float(5e-4 * np.abs(q) @ (entry_marks + exit_marks))
            manual_net = manual_gross - manual_cost - float(t.borrow_cost)
            check = pd.DataFrame({
                'direct_calculation': [manual_gross, manual_cost, float(t.borrow_cost), manual_net],
                'trade_ledger': [t.pnl_gross, t.transaction_cost, t.borrow_cost, t.pnl_net],
            }, index=['gross_PnL_USD', 'transaction_cost_USD', 'borrow_cost_USD', 'net_PnL_USD'])
            np.testing.assert_allclose(check.direct_calculation, check.trade_ledger, rtol=1e-9, atol=1e-6)
            display(t[['pair', 'entry_date', 'exit_date', 'side_label', 'q1', 'q2',
                       'long_notional', 'short_notional', 'holding_days', 'reason']].to_frame('value'))
            display(check)
            print('Net return on fixed long-side capital:', f'{100 * manual_net / 100000:.6f}%')
        """),
        markdown(r"""
        ## 7. Price and dependence context

        Similar price histories do not guarantee convergence. Conditional
        dependence can differ between ordinary and extreme returns. A mixture
        can represent distinct upper/lower tail behavior, but must earn its
        additional complexity through trading evidence rather than likelihood
        alone. The figures below are generated from this experiment's data.
        """),
        code("""
        def show_figure(filename):
            path = ROOT / 'report' / 'figures' / filename
            if not path.exists():
                raise FileNotFoundError(f'Missing figure: {path}; build the report figures first.')
            display(Image(filename=str(path), width=1100))

        show_figure('context.png')
        show_figure('dependence.png')
        """),
        markdown(r"""
        ## 8. Pair and portfolio capital paths

        Read each figure's capital convention explicitly. Uncapped fixed-stake
        curves show shadow balances and can cross zero; the funded comparison
        scales new entries to surviving equity. They include all idle days,
        entry/exit costs and forced liquidation at each six-month endpoint.
        Separate pair paths reveal whether one assigned relationship dominates
        a portfolio result. Fixed-dollar long/short exposure is not guaranteed
        market-beta neutrality, and the three sleeves share SPY exposure.
        """),
        code("""
        show_figure('pair_equity.png')
        show_figure('portfolio_equity.png')
        """),
        markdown(r"""
        ## 9. From conditional ranks to trades

        For uniforms $(u,v)$, compute
        $h_1=\partial C(u,v)/\partial v$ and
        $h_2=\partial C(u,v)/\partial u$.
        The flags accumulate $M_{i,t}=M_{i,t-1}+h_{i,t}-0.5$ from zero at each
        trading-window start. Enter long first/short second when
        $M_1<-0.2$ **and** $M_2>0.2$; reverse the position for the opposite signs.
        Flags need not be stationary, and $h_i=0.5$ identifies a modeled
        conditional median, not a stock's fundamental fair value.

        Paper 1 is inconsistent: its prose says both flags must return to zero,
        but its numbered algorithm closes when either does. The baseline follows
        the numbered algorithm, interpreting return to zero as a favorable sign
        crossing. Flags are not reset after an exit. The alternative both-flag
        rule is shown in sensitivity results. All positions close at expiry.
        """),
        code("""
        show_figure('signal_example.png')
        show_figure('model_selection.png')
        """),
        markdown(r"""
        ## 10. Entry and exit sensitivity

        The published ±0.2 entry threshold is fixed before evaluating the new
        trading sample. The preset 0.05–0.55 grid and both exit conventions are
        sensitivity checks. Their best retrospective result is not promoted
        into a newly optimized baseline. The table reports fixed-capital
        portfolio totals and annualized mean P&L, not a compounded CAGR.
        """),
        code("""
        sensitivity = pd.read_csv(RESULTS / 'sensitivity_windows.csv')
        threshold_table = sensitivity.groupby(['exit_rule', 'threshold']).agg(
            total_pair_returns=('return_net', 'sum'), total_pair_days=('days', 'sum'),
            trades=('trades', 'sum'), total_pair_turnover=('turnover', 'sum'))
        threshold_table['annual_mean_net_pct'] = 100 * 252 * threshold_table.total_pair_returns / threshold_table.total_pair_days
        threshold_table['cumulative_portfolio_PnL_pct'] = 100 * threshold_table.total_pair_returns / len(PAIRS)
        threshold_table['portfolio_turnover'] = threshold_table.total_pair_turnover / len(PAIRS)
        display(threshold_table[['annual_mean_net_pct', 'cumulative_portfolio_PnL_pct', 'trades', 'portfolio_turnover']])
        show_figure('threshold_sensitivity.png')
        """),
        markdown(r"""
        ## 11. Trading and stock-borrow costs

        Signals and fixed stakes are held unchanged while costs vary. Each
        scenario starts from gross P&L, so the 5-bp baseline cost is not deducted
        twice. Borrow fees apply to the preceding short marked notional with
        252 holding intervals per year. The 14.5-bp-per-fill scenario equals
        58 bps per unchanged-notional pair round trip, a transparent comparison
        with the literal recent-cost interpretation of paper 2; it is not the
        paper's complete historical commission/impact schedule.
        """),
        code("""
        costs = pd.read_csv(RESULTS / 'cost_sensitivity.csv')
        display(costs.loc[costs.pair.eq('Portfolio'),
                         ['method', 'cost_bps', 'borrow_bps', 'cumulative_pnl_pct',
                          'annual_mean_pct', 'sharpe', 'max_drawdown_pct']]
                .rename(columns={'max_drawdown_pct': 'algebraic_drawdown_pct'}).reset_index(drop=True))
        show_figure('cost_sensitivity.png')
        """),
        markdown(r"""
        ## 12. Chronological stability

        The chronological slices describe how performance changes across
        economic regimes. They are reporting periods and were not used to choose
        parameters. The full baseline remains the same throughout.
        """),
        code("""
        subperiods = pd.read_csv(RESULTS / 'subperiods.csv')
        display(subperiods[['period', 'method', 'annual_mean_pct', 'annual_vol_pct',
                           'sharpe', 'cumulative_pnl_pct', 'max_drawdown_pct']]
                .rename(columns={'max_drawdown_pct': 'algebraic_drawdown_pct'}))
        """),
        markdown(r"""
        ## 13. Uncertainty and dependence

        Monthly mean t-statistics use HAC standard errors with six **monthly**
        lags. This is a declared reporting choice, distinct from paper 1's
        six-daily-lag procedure. A seeded circular moving-block bootstrap uses
        six-month blocks and 5,000 replications for differences between the
        selected-copula portfolio and preset comparators. It is a descriptive
        interval, not paper 1's stationary bootstrap with automatic block
        selection and not a correction for testing many strategies.
        """),
        code("""
        display(performance.loc[performance.pair.eq('Portfolio') & performance.basis.eq('net'),
                                ['method', 'monthly_mean_bps', 'monthly_hac_t', 'monthly_hac_p',
                                 'negative_month_pct']].reset_index(drop=True))
        intervals = pd.DataFrame(audit['bootstrap_intervals']).T
        intervals.index.name = 'Selected_minus_comparator'
        display(intervals)
        """),
        markdown(r"""
        ## 14. Accounting and provenance checks

        Daily marked-to-market P&L must reconcile to the completed-trade ledger.
        Baseline net P&L must equal gross less transaction costs and borrow fees
        exactly once. All fitted windows must still match the recorded input and
        model hashes. These checks complement unit tests of conditional
        probabilities, causal filtering, signal timing and hand-computed trades.
        """),
        code("""
        grouped_daily = daily.groupby(['method', 'pair']).pnl_net.sum()
        grouped_trades = trades.groupby(['method', 'pair']).pnl_net.sum().reindex(grouped_daily.index, fill_value=0)
        reconciliation = float((grouped_daily - grouped_trades).abs().max())
        np.testing.assert_allclose(grouped_daily, grouped_trades, atol=1e-6, rtol=1e-9)
        np.testing.assert_allclose(daily.pnl_net,
                                   daily.pnl_gross - daily.transaction_cost - daily.borrow_cost,
                                   atol=1e-8, rtol=1e-10)
        current_signature = fingerprint()
        assert all(cache['fingerprint'] == current_signature for cache in load_cache())
        assert audit['all_marginals_converged'] and audit['all_copula_models_success']
        print('Daily/trade maximum absolute difference, USD:', reconciliation)
        print('Verified fitted windows:', audit['windows'])
        print('OOS trading days:', audit['oos_days'])
        print('Minimum uncapped shadow balance per initial dollar:', audit['minimum_baseline_equity_per_initial_dollar'])
        print('All supplementary equity-funded account balances positive:', funded_daily.equity_per_initial_dollar.gt(0).all())
        display(pd.Series(audit['design'], name='Implemented convention').to_frame())
        """),
        markdown(r"""
        ## 15. What the experiment establishes

        The following conclusions are constructed from the executed results.
        The papers' published profits are literature comparisons, not numerical
        targets imposed on these data. Paper 1 reported top-five net committed
        annual mean excess returns of 3.68% for mixed copulas versus 2.30% for
        distance. Paper 2 reported net monthly employed-capital means of 0.38%,
        0.33% and 0.05% for distance, cointegration and copula, respectively.
        Different pairs, periods, costs, capital conventions and window designs
        prevent interpreting differences from those figures as a coding error.
        """),
        code("""
        net_portfolio = performance.loc[performance.pair.eq('Portfolio') & performance.basis.eq('net')].set_index('method')
        selected = net_portfolio.loc['Selected']
        distance = net_portfolio.loc['Distance']
        funded_selected = funded.loc[funded.method.eq('Selected') & funded.pair.eq('Portfolio')].iloc[0]
        selected_pairs = performance.loc[performance.method.eq('Selected') & performance.basis.eq('net') & performance.pair.ne('Portfolio')]
        positive_pairs = int(selected_pairs.cumulative_pnl_pct.gt(0).sum())
        mixed_share = 100 * fits.loc[fits.selected, 'family'].isin(['CFG', 'CtG']).mean()
        difference = intervals.loc['Distance']
        interval_text = ('contains zero' if difference.lower95_bps <= 0 <= difference.upper95_bps
                         else 'excludes zero for this prespecified comparison')
        result_direction = 'positive' if selected.cumulative_pnl_pct > 0 else 'negative' if selected.cumulative_pnl_pct < 0 else 'zero'
        display(Markdown(f'''
        The supplementary **equity-at-entry selected-copula portfolio** finishes
        with **{funded_selected.final_equity_per_initial_dollar:.4f} dollars per initial dollar**,
        a total net return of **{funded_selected.cumulative_return_pct:.2f}%**,
        CAGR **{funded_selected.cagr_pct:.2f}%**, and maximum drawdown
        **{funded_selected.max_drawdown_pct:.2f}%** under its stated financing assumptions.

        The separate **uncapped fixed-stake** diagnostic has {result_direction}
        cumulative net P&L of **{selected.cumulative_pnl_pct:.2f}%** of original
        committed capital. Its annualized mean contribution is **{selected.annual_mean_pct:.2f}%**,
        Sharpe **{selected.sharpe:.2f}**, and **algebraic** drawdown **{selected.max_drawdown_pct:.2f}%**.
        Distance's annualized mean is **{distance.annual_mean_pct:.2f}%** with Sharpe
        **{distance.sharpe:.2f}**. These uncapped fixed-stake statistics are not
        compounded CAGRs or evidence that the strategy remains funded after
        capital exhaustion. The funding-failure table gives the relevant dates.

        **{positive_pairs} of the 3** selected-copula pair sleeves have positive
        cumulative net P&L. Mixtures win **{mixed_share:.1f}%** of formation
        log-likelihood comparisons. That frequency describes in-sample fit,
        while the tables above establish the realized trading outcomes.

        The six-month-block 95% interval for the selected-minus-distance monthly
        mean difference is **[{difference.lower95_bps:.2f}, {difference.upper95_bps:.2f}] bps**
        and {interval_text}. The interval has the stated descriptive limitations.

        This experiment completes the three assigned pairs, seven dependence
        families, marginal filtering, distance and screened cointegration
        comparisons, paper-style marked-to-market P&L, capital denominators,
        and preset cost/entry/exit checks. Its small preselected universe,
        idealized closing fills, synthetic adjusted-price units, omitted baseline
        financing/borrow constraints and model-search comparisons limit broader
        profitability claims. The evidence does not establish risk-free arbitrage.
        '''))
        """),
        markdown(r"""
        ## Reproduction files

        - `src/marginals.py`, `src/copulas.py`: formation estimation and conditional probabilities.
        - `src/signals.py`, `src/backtest.py`, `src/metrics.py`: signals, holdings, P&L and statistics.
        - `scripts/download_data.py`, `scripts/fit_models.py`, `scripts/run_analysis.py`: the pipeline.
        - `data_manifest.json`: source provenance and original-data hashes.
        - `results/*.csv`, `results/audit.json`: the numerical tables, ledgers and checks.
        - `tests/`: mathematical, timing, capital and accounting checks (`python -m pytest -q`).

        Install the pinned dependencies with `python -m pip install -r requirements.txt`.
        Run the download, fitting and analysis scripts in that order for a clean
        reproduction. The notebook reuses verified cached fits. Future vendor
        revisions can change fresh downloads and must not be confused with the
        preserved, fingerprinted original inputs. Supplied papers and raw vendor
        data are kept local rather than republished in a source repository.
        """),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "title": "The profitability of pairs trading strategies using copula methods",
        "authors": [{"name": "Panagiotis Housos", "NYU_netid": "ph2606"}],
    }
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    notebook = build_notebook()
    output = ROOT / "copula_pairs_trading.ipynb"
    if args.execute:
        if not (ROOT / "results" / "audit.json").exists():
            raise FileNotFoundError("Run analysis and build figures before executing the notebook.")
        old_pythonpath = os.environ.get("PYTHONPATH")
        paths = ([str(LOCAL_PACKAGES)] if LOCAL_PACKAGES.exists() else []) + [str(ROOT)]
        if old_pythonpath:
            paths.append(old_pythonpath)
        os.environ["PYTHONPATH"] = os.pathsep.join(paths)
        try:
            print("Executing notebook from verified fitted caches...", flush=True)
            NotebookClient(notebook, timeout=1200, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        finally:
            if old_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pythonpath
    nbformat.write(notebook, output)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    print(f"Wrote {output.name}: {len(code_cells)} code cells; executed={args.execute}", flush=True)


if __name__ == "__main__":
    main()

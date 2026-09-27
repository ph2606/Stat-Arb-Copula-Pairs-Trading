"""Hand-check capital denominators and fixed-stake reporting conventions."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".packages").exists():
    sys.path.insert(0, str(ROOT / ".packages"))
sys.path.insert(0, str(ROOT))

from src.backtest import backtest
from src.metrics import monthly_capital_returns, summary


def _empty_slot(index):
    return pd.DataFrame({"return_net": 0.0, "position": 0, "turnover": 0.0}, index=index)


def test_no_trade_month_committed_zero_employed_undefined():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    result = monthly_capital_returns([_empty_slot(dates) for _ in range(3)])
    np.testing.assert_array_equal(result.committed, 0.0)
    assert result.employed.isna().all()
    np.testing.assert_array_equal(result.active_pairs, 0)
    np.testing.assert_array_equal(result.nominated_pairs, 3)


def test_carried_trade_counts_in_month_without_new_entry():
    dates = pd.to_datetime(["2020-01-29", "2020-01-30", "2020-01-31",
                            "2020-02-03", "2020-02-04", "2020-03-02"])
    prices = pd.DataFrame({"A": [100, 100, 100, 110, 120, 120], "B": 100.0}, index=dates)
    targets = pd.Series([1, 1, 1, 0, 0, 0], index=dates)
    ledger, trades = backtest(prices, targets, cost_bps=0, capital=1000)
    result = monthly_capital_returns([ledger, _empty_slot(dates), _empty_slot(dates)])
    feb = result.loc["2020-02-29"]
    assert ledger.loc["2020-02"].entry_event.sum() == 0
    assert feb.active_pairs == 1
    assert feb.employed == pytest.approx(0.20)
    assert feb.committed == pytest.approx(0.20 / 3)
    assert trades.iloc[0].pnl_gross == pytest.approx(200)
    assert result.loc["2020-03-31", "committed"] == 0
    assert np.isnan(result.loc["2020-03-31", "employed"])


def test_employed_counts_distinct_monthly_pairs_not_simultaneous_pairs():
    dates = pd.bdate_range("2020-02-03", periods=6)
    first, second, idle = [_empty_slot(dates) for _ in range(3)]
    first.loc[dates[0:2], "position"] = 1
    first.loc[dates[1], "return_net"] = 0.02
    second.loc[dates[3:5], "position"] = -1
    second.loc[dates[4], "return_net"] = -0.01
    result = monthly_capital_returns([first, second, idle]).iloc[0]
    assert result.active_pairs == 2
    assert result.committed == pytest.approx(0.01 / 3)
    assert result.employed == pytest.approx(0.01 / 2)


def test_first_day_loss_is_in_drawdown_from_initial_capital():
    dates = pd.bdate_range("2020-01-01", periods=80)
    returns = pd.Series(0.0, index=dates)
    returns.iloc[:3] = [-0.10, 0.05, 0.05]
    result = summary(returns)
    assert result["max_drawdown_pct"] == pytest.approx(-10.0)
    assert result["min_equity_per_initial_dollar"] == pytest.approx(0.90)
    assert result["cumulative_pnl_pct"] == pytest.approx(0.0)


def test_fixed_initial_stake_weights_are_added_not_compounded():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    returns = pd.Series(0.0, index=dates)
    # Two separate trades each earn $100 on their unchanged $1,000 long stake.
    returns.loc["2020-01-20"] = 0.10
    returns.loc["2020-02-20"] = 0.10
    result = summary(returns)
    assert result["cumulative_pnl_pct"] == pytest.approx(20.0)
    assert result["monthly_mean_bps"] == pytest.approx(2000 / 3)
    assert result["annual_mean_pct"] == pytest.approx(100 * 0.20 / len(dates) * 252)


def test_net_costs_are_aggregated_once_and_denominator_is_long_stake():
    dates = pd.bdate_range("2020-01-02", periods=5)
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=dates)
    targets = pd.Series([1, 1, 0, 0, 0], index=dates)
    ledger, trades = backtest(prices, targets, cost_bps=5, capital=1000)
    result = monthly_capital_returns([ledger, _empty_slot(dates), _empty_slot(dates)]).iloc[0]
    # Four unchanged $1,000 fills at 5bp = $2; paper denominator is $1,000.
    assert trades.iloc[0].transaction_cost == pytest.approx(2.0)
    assert ledger.return_net.sum() == pytest.approx(-0.002)
    assert result.employed == pytest.approx(-0.002)
    assert result.committed == pytest.approx(-0.002 / 3)

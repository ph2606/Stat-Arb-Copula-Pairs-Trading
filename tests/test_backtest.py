"""Small hand-computed accounting tests, including timing and terminal costs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest import backtest


def example(a, b, target):
    dates = pd.bdate_range("2020-01-01", periods=len(a))
    return pd.DataFrame({"A": a, "B": b}, index=dates), pd.Series(target, index=dates)


def test_next_close_execution_fixed_units_and_terminal_cost():
    p, s = example([100, 200, 220, 240], [100, 100, 110, 100], [1, 1, 1, 1])
    ledger, trades = backtest(p, s, capital=1000, cost_bps=5)
    # Entry at day 1 after A doubled: q=(5,-10), no pre-entry profit.
    np.testing.assert_allclose(ledger.pnl_gross, [0, 0, 0, 200])
    np.testing.assert_allclose(ledger.transaction_cost, [0, 1, 0, 1.1])
    np.testing.assert_allclose(ledger.q1_before, [0, 0, 5, 5])
    np.testing.assert_allclose(ledger.q2_before, [0, 0, -10, -10])
    assert ledger.iloc[-1].position == 0
    assert trades.iloc[0].reason == "terminal"
    assert trades.iloc[0].holding_days == 2
    assert trades.iloc[0].pnl_gross == pytest.approx(200)
    assert trades.iloc[0].pnl_net == pytest.approx(197.9)
    assert ledger.iloc[-1].equity_net == pytest.approx(1197.9)
    assert ledger.pnl_net.sum() == pytest.approx(trades.pnl_net.sum())


def test_fixed_units_equal_drifted_weights_not_daily_rebalancing():
    p, s = example([100, 100, 110, 121, 120], [100] * 5, [1] * 5)
    ledger, trades = backtest(p, s, capital=1000, cost_bps=0)
    # On the second +10% day, the marked long allocation is $1100: gain $110.
    assert ledger.iloc[3].pnl_gross == pytest.approx(110)
    entry = p.iloc[1]
    for i in (2, 3, 4):
        drifted_weights = (p.iloc[i - 1] / entry) * [1, -1]
        expected = float(drifted_weights @ (p.iloc[i] / p.iloc[i - 1] - 1))
        assert ledger.iloc[i].return_gross == pytest.approx(expected)
    assert trades.iloc[0].return_gross == pytest.approx(0.2)


def test_flat_roundtrip_costs_twenty_basis_points():
    p, s = example([100] * 5, [50] * 5, [1, 1, 0, 0, 0])
    ledger, trades = backtest(p, s, capital=100_000, cost_bps=5)
    assert ledger.pnl_gross.sum() == 0
    assert ledger.turnover.sum() == pytest.approx(4)
    assert ledger.return_net.sum() == pytest.approx(-0.002)
    assert trades.iloc[0].transaction_cost == pytest.approx(200)
    assert trades.iloc[0].reason == "signal"


def test_future_signal_change_cannot_change_prior_pnl():
    p, s = example([100, 102, 99, 104, 101, 108], [100, 99, 102, 105, 100, 101], [1] * 6)
    changed = s.copy()
    changed.iloc[3:] = -1
    base, _ = backtest(p, s, lag=1)
    alternative, _ = backtest(p, changed, lag=1)
    columns = ["pnl_gross", "pnl_net", "position", "q1_after", "q2_after"]
    pd.testing.assert_frame_equal(base.iloc[:4][columns], alternative.iloc[:4][columns])
    # The first changed target executes day 4 after day 4's holding PnL.
    assert base.iloc[4].pnl_gross == pytest.approx(alternative.iloc[4].pnl_gross)
    assert alternative.iloc[4].position == -1


def test_final_day_entry_suppressed_and_lagged_exit_reason():
    p, s = example([100] * 4, [100] * 4, [0, 0, 1, 1])
    ledger, trades = backtest(p, s)
    assert trades.empty
    assert ledger.transaction_cost.sum() == 0
    s[:] = [1, 0, 0, 0]
    reasons = pd.Series([None, "convergence", None, None], index=s.index)
    ledger, trades = backtest(p, s, close_reasons=reasons)
    assert trades.iloc[0].reason == "convergence"
    assert trades.iloc[0].entry_date == p.index[1]
    assert trades.iloc[0].exit_date == p.index[2]


def test_short_ratio_lagged_and_frozen_until_exit_with_borrow():
    p, s = example([100] * 5, [50] * 5, [-1] * 5)
    ratios = pd.Series([2, 3, 4, 5, 6], index=p.index)
    ledger, trades = backtest(p, s, capital=1000, cost_bps=5,
                              borrow_bps=252, short_ratio=ratios)
    # Signal-day ratio 2, long B=$1000, short A=$2000; no daily resize.
    np.testing.assert_allclose(ledger.q1_before, [0, 0, -20, -20, -20])
    np.testing.assert_allclose(ledger.q2_before, [0, 0, 20, 20, 20])
    assert trades.iloc[0].short_ratio == 2
    assert trades.iloc[0].transaction_cost == pytest.approx(3)
    assert trades.iloc[0].borrow_cost == pytest.approx(0.6)
    assert ledger.pnl_net.sum() == pytest.approx(-3.6)


def test_reversal_closes_and_opens_both_legs_but_unchanged_target_does_not():
    p, s = example([100] * 6, [100] * 6, [1, 1, -1, -1, -1, -1])
    ledger, trades = backtest(p, s, capital=1000, cost_bps=5)
    assert len(trades) == 2
    assert trades.iloc[0].reason == "reversal"
    assert ledger.iloc[2].transaction_cost == 0
    assert ledger.iloc[3].transaction_cost == pytest.approx(2)
    assert ledger.iloc[3].entry_event and ledger.iloc[3].exit_event
    assert ledger.transaction_cost.sum() == pytest.approx(4)
    assert ledger.pnl_net.sum() == pytest.approx(trades.pnl_net.sum())


def test_flat_no_trade_and_invalid_inputs():
    p, s = example([100] * 3, [100] * 3, [0, 0, 0])
    ledger, trades = backtest(p, s)
    assert trades.empty
    assert ledger.active.sum() == 0
    assert ledger.pnl_net.sum() == 0
    with pytest.raises(ValueError, match="lag"):
        backtest(p, s, lag=-1)
    with pytest.raises(ValueError, match="short_ratio"):
        backtest(p, s, short_ratio=0)
    with pytest.raises(ValueError, match="targets"):
        backtest(p, s.iloc[:-1])
    with pytest.raises(ValueError, match="prices"):
        backtest(p.assign(A=np.nan), s)

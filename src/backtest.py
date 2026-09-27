"""Fixed-stake, fixed-entry-unit marked-to-market pairs accounting.

The denominator is long-side capital, as in Rad, Low and Faff (2016),
section 4.5: a $1 long / $1 short pair has gross exposure $2 but return
denominator $1. Adjusted prices represent synthetic total-return units;
no separate dividends are credited. Signals at close t execute at close
t + lag. New units earn PnL only after execution. Equity is arithmetic
initial capital plus cumulative PnL, with a fixed stake on every entry.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd


TRADE_COLUMNS = [
    "trade_id", "entry_date", "exit_date", "side", "side_label", "reason",
    "holding_days", "holding_calendar_days", "entry_price1", "entry_price2",
    "exit_price1", "exit_price2", "q1", "q2", "short_ratio", "long_notional",
    "short_notional", "entry_cost", "exit_cost", "transaction_cost", "borrow_cost",
    "pnl_gross", "pnl_net", "return_gross", "return_net",
]


def backtest(
    prices: pd.DataFrame,
    targets: pd.Series,
    close_reasons: pd.Series | None = None,
    cost_bps: float = 5,
    borrow_bps: float = 0,
    lag: int = 1,
    capital: float = 100_000,
    short_ratio: float | pd.Series = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a daily ledger and completed-trade ledger for one pair/window.

    ``targets`` is the desired state at each SIGNAL close: +1 means long
    first/short second, -1 means short first/long second, 0 means flat.
    ``close_reasons`` and a Series ``short_ratio`` also refer to signal
    dates and are shifted with targets. A ratio is used only on entry;
    changing it while the target is unchanged never rebalances units.
    The long leg starts at ``capital`` and the short leg at
    ``capital * short_ratio``. Both quantities remain fixed until exit.

    ``cost_bps`` is per actual dollar traded, per fill. ``borrow_bps`` is
    an annual rate charged on prior-close short marked notional for each
    held trading interval, with 252 intervals per year. There is no cash
    interest or financing-credit assumption. All remaining positions
    close at the final close (reason ``terminal``); final-date entries
    are suppressed. A reversal explicitly closes then reopens both legs.
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] != 2 or prices.empty:
        raise ValueError("prices must be a nonempty two-column DataFrame")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be unique and increasing")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    p = prices.to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("prices must be finite and strictly positive")
    if not isinstance(lag, numbers.Integral) or isinstance(lag, bool) or lag < 0:
        raise ValueError("lag must be a nonnegative integer")
    for name, value in (("capital", capital), ("cost_bps", cost_bps), ("borrow_bps", borrow_bps)):
        if not np.isfinite(value) or value < 0 or (name == "capital" and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'capital' else 'nonnegative'}")
    if not isinstance(targets, pd.Series) or not targets.index.is_unique:
        raise ValueError("targets must be a Series with unique dates")
    target = targets.reindex(prices.index)
    if target.isna().any() or not target.isin([-1, 0, 1]).all():
        raise ValueError("targets must cover all price dates with values -1, 0, or 1")
    executed_targets = target.shift(lag, fill_value=0).astype(int)
    reasons = (close_reasons.reindex(prices.index) if close_reasons is not None
               else pd.Series(index=prices.index, dtype=object)).shift(lag)
    if isinstance(short_ratio, pd.Series):
        if not short_ratio.index.is_unique:
            raise ValueError("short_ratio dates must be unique")
        ratios = short_ratio.reindex(prices.index).shift(lag)
    else:
        if not np.isfinite(short_ratio) or short_ratio <= 0:
            raise ValueError("short_ratio must be positive and finite")
        ratios = pd.Series(float(short_ratio), index=prices.index)

    rate, borrow_rate = cost_bps / 10_000, borrow_bps / 10_000 / 252
    q = np.zeros(2, dtype=float)
    position = 0
    active: dict | None = None
    trade_id = 0
    records, trades = [], []
    dates = prices.index
    for i, date in enumerate(dates):
        marks = p[i]
        before = q.copy()
        position_before = position
        gross = float(before @ (marks - p[i - 1])) if i else 0.0
        short_exposure = float(np.maximum(-before, 0) @ p[i - 1]) if i else 0.0
        borrow = short_exposure * borrow_rate
        transaction_cost = 0.0
        traded_notional = 0.0
        entered = exited = False
        exit_reason = None
        desired = int(executed_targets.iloc[i])
        is_final = i == len(dates) - 1
        if is_final:
            desired = 0
        if active is not None:
            active["pnl_gross"] += gross
            active["borrow_cost"] += borrow

        if position != 0 and desired != position:
            notional = float(np.abs(q) @ marks)
            exit_cost = rate * notional
            transaction_cost += exit_cost
            traded_notional += notional
            reason_at_signal = reasons.iloc[i]
            exit_reason = ("terminal" if is_final else
                           str(reason_at_signal) if pd.notna(reason_at_signal) and str(reason_at_signal)
                           else "reversal" if desired else "signal")
            active.update({
                "exit_date": date, "exit_price1": marks[0], "exit_price2": marks[1],
                "holding_days": i - active.pop("entry_i"),
                "holding_calendar_days": (date - active["entry_date"]).days,
                "reason": exit_reason, "exit_cost": exit_cost,
            })
            active["transaction_cost"] = active["entry_cost"] + exit_cost
            active["pnl_net"] = active["pnl_gross"] - active["transaction_cost"] - active["borrow_cost"]
            active["return_gross"] = active["pnl_gross"] / capital
            active["return_net"] = active["pnl_net"] / capital
            trades.append(active)
            active = None
            q[:] = 0
            position = 0
            exited = True

        if desired != 0 and position == 0 and not is_final:
            ratio = float(ratios.iloc[i])
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"short_ratio at the signal for entry {date} must be positive and finite")
            allocations = (np.array([capital, -capital * ratio]) if desired == 1
                           else np.array([-capital * ratio, capital]))
            q = allocations / marks
            notional = float(np.abs(allocations).sum())
            entry_cost = rate * notional
            transaction_cost += entry_cost
            traded_notional += notional
            position = desired
            trade_id += 1
            active = {
                "trade_id": trade_id, "entry_date": date, "entry_i": i,
                "side": position,
                "side_label": "long_first_short_second" if position == 1 else "short_first_long_second",
                "entry_price1": marks[0], "entry_price2": marks[1],
                "q1": q[0], "q2": q[1], "short_ratio": ratio,
                "long_notional": float(capital), "short_notional": capital * ratio,
                "entry_cost": entry_cost, "pnl_gross": 0.0, "borrow_cost": 0.0,
            }
            entered = True

        net = gross - transaction_cost - borrow
        records.append({
            "date": date, "target_signal": int(target.iloc[i]),
            "target_executable": int(executed_targets.iloc[i]),
            "position_before": position_before, "position": position,
            "q1_before": before[0], "q2_before": before[1],
            "q1_after": q[0], "q2_after": q[1],
            "pnl_gross": gross, "transaction_cost": transaction_cost,
            "borrow_cost": borrow, "pnl_net": net,
            "return_gross": gross / capital, "return_net": net / capital,
            "traded_notional": traded_notional, "turnover": traded_notional / capital,
            "gross_exposure_before": float(np.abs(before) @ marks),
            "gross_exposure": float(np.abs(q) @ marks),
            "entry_event": entered, "exit_event": exited, "exit_reason": exit_reason,
            "active": bool(position_before or position or entered or exited),
        })
    ledger = pd.DataFrame(records).set_index("date")
    ledger["equity_gross"] = capital + ledger["pnl_gross"].cumsum()
    ledger["equity_net"] = capital + ledger["pnl_net"].cumsum()
    ledger["cumulative_return_gross"] = ledger["return_gross"].cumsum()
    ledger["cumulative_return_net"] = ledger["return_net"].cumsum()
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return ledger, trade_frame

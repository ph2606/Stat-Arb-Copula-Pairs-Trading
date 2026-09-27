"""Formation-only spread models and causal trading-state rules."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def copula_targets(h1, h2, index, threshold=0.2, exit_rule='either'):
    """Flags start at zero each window, accumulate without resetting after exits.

    Zero is interpreted as reaching/crossing the entry-side zero barrier, not
    floating-point equality. The either rule follows paper1's numbered step 5;
    both is an explicit alternative for its conflicting prose and paper2.
    """
    if exit_rule not in ('either', 'both') or threshold <= 0:
        raise ValueError('Invalid signal rule')
    a, b = np.asarray(h1), np.asarray(h2)
    if a.shape != b.shape or len(a) != len(index) or not np.isfinite([a, b]).all():
        raise ValueError('Invalid conditional probabilities')
    if np.any((a < 0) | (a > 1) | (b < 0) | (b > 1)):
        raise ValueError('Probabilities outside [0,1]')
    m1, m2 = np.cumsum(a - .5), np.cumsum(b - .5)
    state = 0
    targets, reasons = [], []
    for x, y in zip(m1, m2):
        reason = ''
        if state:
            hit1, hit2 = state*x >= 0, state*y <= 0
            if (hit1 or hit2) if exit_rule == 'either' else (hit1 and hit2):
                state, reason = 0, 'flag_convergence'
        elif x < -threshold and y > threshold:
            state = 1
        elif x > threshold and y < -threshold:
            state = -1
        targets.append(state)
        reasons.append(reason)
    return pd.DataFrame({'target': targets, 'reason': reasons,
                         'h1': a, 'h2': b, 'flag1': m1, 'flag2': m2}, index=index)


def spread_targets(z, entry=2.0, positive_side=-1):
    """Enter outside ±entry, exit after spread crosses zero; no same-day flip."""
    state = 0
    out, reasons = [], []
    for value in z:
        reason = ''
        if state:
            if state * positive_side * value <= 0:
                state, reason = 0, 'spread_convergence'
        elif value > entry:
            state = positive_side
        elif value < -entry:
            state = -positive_side
        out.append(state)
        reasons.append(reason)
    return pd.DataFrame({'target': out, 'reason': reasons, 'z': z}, index=z.index)


def distance_signals(train_prices, test_prices):
    """Paper2 distance benchmark: rebase each period, formation spread SD."""
    norm_train = train_prices / train_prices.iloc[0]
    spread_train = norm_train.iloc[:, 0] - norm_train.iloc[:, 1]
    sd = spread_train.std(ddof=1)
    norm_test = test_prices / test_prices.iloc[0]
    z = (norm_test.iloc[:, 0] - norm_test.iloc[:, 1]) / sd
    return spread_targets(z), {'spread_sd': sd,
                              'ssd': float((spread_train**2).sum())}


def cointegration_signals(train_prices, test_prices):
    """Engle-Granger with intercept, AIC lag selection, fixed 5% eligibility.

    The oriented regression is second normalized price on first. We reproduce
    paper2's dollar sizing (long=1, short=beta or 1/beta), not beta share units.
    Rejected pairs leave cash in their committed slot; no replacement search.
    """
    base = train_prices.iloc[0]
    xtrain = train_prices / base
    xtest = test_prices / base
    x, y = xtrain.iloc[:, 0], xtrain.iloc[:, 1]
    reg = sm.OLS(y, sm.add_constant(x)).fit()
    alpha, beta = map(float, reg.params)
    stat, pvalue, crit = coint(y, x, trend='c', autolag='aic')
    spread = y - beta*x
    center, sd = spread.mean(), spread.std(ddof=1)
    z = (xtest.iloc[:, 1] - beta*xtest.iloc[:, 0] - center) / sd
    eligible = pvalue < .05 and beta > 0 and sd > 0
    signals = spread_targets(z, positive_side=1)
    if not eligible:
        signals['target'] = 0
        signals['reason'] = ''
    signals['short_ratio'] = np.where(signals['target'] == 1,
                                     1/beta if beta > 0 else 1,
                                     beta if beta > 0 else 1)
    return signals, {'alpha': alpha, 'beta': beta, 'eg_stat': float(stat),
                     'eg_pvalue': float(pvalue), 'eligible': bool(eligible),
                     'spread_mean': float(center), 'spread_sd': float(sd)}

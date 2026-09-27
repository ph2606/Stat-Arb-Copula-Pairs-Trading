"""Fixed-stake P&L statistics; no accidental compounding of initial-capital returns."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def summary(daily_returns, monthly_returns=None):
    r=pd.Series(daily_returns,dtype=float)
    if monthly_returns is None:
        monthly_returns=r.resample('ME').sum()
    m=pd.Series(monthly_returns,dtype=float).dropna()
    volatility=r.std(ddof=1)*np.sqrt(252)
    annual_mean=r.mean()*252
    equity=1+r.cumsum()
    # Include starting capital in the running maximum, even after a day-one loss.
    running_high=equity.cummax().clip(lower=1)
    downside=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(252)
    hac=sm.OLS(m.to_numpy(),np.ones((len(m),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    return {'days':len(r),'months':len(m),'cumulative_pnl_pct':100*r.sum(),
            'annual_mean_pct':100*annual_mean,'annual_vol_pct':100*volatility,
            'sharpe':annual_mean/volatility if volatility>0 else np.nan,
            'sortino':annual_mean/downside if downside>0 else np.nan,
            'max_drawdown_pct':100*(equity/running_high-1).min(),
            'monthly_mean_bps':1e4*m.mean(),'monthly_hac_t':float(hac.tvalues[0]),
            'monthly_hac_p':float(hac.pvalues[0]),
            'negative_month_pct':100*(m<0).mean(),
            'monthly_var95_pct':100*m.quantile(.05),
            'monthly_cvar95_pct':100*m[m<=m.quantile(.05)].mean(),
            'min_equity_per_initial_dollar':float(equity.min())}


def monthly_capital_returns(ledgers):
    """Three fixed committed slots; employed denominator counts monthly activity.

    A pair counts as employed if held at any time or an order was filled during
    the month. It is never scaled by today's active count with hindsight.
    No-activity REC is undefined (NaN), while committed return is zero.
    """
    monthly_pnl=[]
    monthly_active=[]
    for ledger in ledgers:
        monthly_pnl.append(ledger['return_net'].resample('ME').sum())
        active=(ledger['position'].ne(0) | ledger['turnover'].gt(0))
        monthly_active.append(active.resample('ME').max())
    pnl=pd.concat(monthly_pnl,axis=1).sum(axis=1)
    active=pd.concat(monthly_active,axis=1).sum(axis=1)
    return pd.DataFrame({'committed':pnl/len(ledgers),
                         'employed':pnl/active.replace(0,np.nan),
                         'active_pairs':active,'nominated_pairs':len(ledgers)})


def block_mean_interval(monthly_differences,seed=2606,reps=5000,block=6):
    """Circular moving-block bootstrap of the mean difference, descriptive CI.

    This is not paper1's stationary bootstrap with automatic block selection.
    The fixed six-month block is declared before inspecting strategy results.
    """
    x=np.asarray(monthly_differences,dtype=float)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(x),size=(reps,int(np.ceil(len(x)/block))))
    indices=(starts[:,:,None]+np.arange(block))%len(x)
    means=x[indices.reshape(reps,-1)[:,:len(x)]].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    return {'mean_monthly_difference_bps':float(x.mean()*1e4),
            'lower95_bps':float(lo*1e4),'upper95_bps':float(hi*1e4),
            'reps':reps,'block_months':block,'seed':seed}

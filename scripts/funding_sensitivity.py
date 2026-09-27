"""Funding diagnostic: resize each new trade to current equity, keep units fixed within it."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def reinvested_account(ledger):
    """Scale the fixed-stake trade path by equity immediately before each entry.

    Causal for non-overlapping trades without same-day reversals. Fees are
    proportional and already included in the unit-stake return. No new signals
    are generated. Raises on exhaustion instead of implying further financing.
    """
    equity=1.0
    stake=0.0
    records=[]
    for date,row in ledger.iterrows():
        previous=equity
        if row.entry_event:
            if row.position_before!=0:
                raise ValueError('Same-day reversals need separate fill accounting')
            stake=equity
        increment=stake*row.return_net
        equity+=increment
        if equity<=0:
            raise ValueError(f'Reinvested account exhausted on {date}')
        records.append({'Date':date,'equity_per_initial_dollar':equity,
            'daily_return':equity/previous-1,'pnl_per_initial_dollar':increment,
            'stake_multiplier':stake})
        if row.position==0:
            stake=0.0
    return pd.DataFrame(records).set_index('Date')

def equity_statistics(frame):
    eq=frame.equity_per_initial_dollar
    r=frame.daily_return
    high=eq.cummax().clip(lower=1)
    vol=r.std(ddof=1)*np.sqrt(252)
    return {'cumulative_return_pct':100*(eq.iloc[-1]-1),
        'cagr_pct':100*(eq.iloc[-1]**(252/len(eq))-1),
        'annual_mean_pct':100*r.mean()*252,'annual_vol_pct':100*vol,
        'sharpe':r.mean()*252/vol if vol>0 else np.nan,
        'max_drawdown_pct':100*(eq/high-1).min(),
        'min_equity_per_initial_dollar':eq.min(),
        'final_equity_per_initial_dollar':eq.iloc[-1]}

def run_funding(daily=None):
    if daily is None:
        daily=pd.read_csv(ROOT/'results'/'daily_ledger.csv',parse_dates=['Date'])
    frames=[]; summary=[]; exhaustion=[]
    for method in ['Selected','Best-single','BIC','Distance','Cointegration']:
        sleeves=[]
        originals=[]
        for pair in ['AAPL-GOOG','IBM-SPY','DIA-SPY']:
            d=daily.loc[(daily.method==method)&(daily.pair==pair)].set_index('Date').sort_index()
            eq=1+d.return_net.cumsum()
            bad=eq.loc[eq<=0]
            exhaustion.append({'method':method,'pair':pair,
                'first_nonpositive_equity':str(bad.index[0].date()) if len(bad) else None,
                'minimum_equity_per_initial_dollar':float(eq.min())})
            originals.append(eq)
            f=reinvested_account(d)
            sleeves.append(f.equity_per_initial_dollar)
            summary.append({'method':method,'pair':pair,**equity_statistics(f)})
            f['method']=method;f['pair']=pair
            frames.append(f)
        eq=pd.concat(sleeves,axis=1).mean(axis=1)
        r=eq.pct_change();r.iloc[0]=eq.iloc[0]-1
        f=pd.DataFrame({'equity_per_initial_dollar':eq,'daily_return':r})
        summary.append({'method':method,'pair':'Portfolio',**equity_statistics(f)})
        f['method']=method;f['pair']='Portfolio';frames.append(f)
        original=pd.concat(originals,axis=1).mean(axis=1)
        bad=original.loc[original<=0]
        exhaustion.append({'method':method,'pair':'Portfolio',
            'first_nonpositive_equity':str(bad.index[0].date()) if len(bad) else None,
            'minimum_equity_per_initial_dollar':float(original.min())})
    pd.concat(frames).reset_index().to_csv(ROOT/'results'/'reinvested_daily.csv',index=False)
    pd.DataFrame(summary).to_csv(ROOT/'results'/'reinvested_performance.csv',index=False)
    (ROOT/'results'/'capital_diagnostics.json').write_text(json.dumps(exhaustion,indent=2)+'\n')
    print('Reinvested account: all checked daily equity values remain positive.')

if __name__=='__main__':
    run_funding()

"""Reproduce all result tables and ledgers from verified prices and fitted caches."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fit_models import PAIRS, WINDOWS, load_prices, fingerprint
from src.backtest import backtest
from src.signals import copula_targets, distance_signals, cointegration_signals
from src.metrics import summary, monthly_capital_returns, block_mean_interval

CAPITAL=100_000
FAMILIES=['Gaussian','Student-t','Clayton','Frank','Gumbel','CFG','CtG']
METHODS=['Selected','Best-single','BIC','Gaussian','Student-t','Clayton','Frank','Gumbel','CFG','CtG','Distance','Cointegration']

def load_cache():
    signature=fingerprint()
    out=[]
    for start in WINDOWS:
        path=ROOT/'results'/'fit_cache'/f'{start.date()}.json'
        if not path.exists():
            raise ValueError(f'Missing fitted window {start.date()}; run scripts/fit_models.py first')
        cache=json.loads(path.read_text())
        if cache['fingerprint']!=signature:
            raise ValueError(f'Stale cache {path.name}; refit required')
        out.append(cache)
    return out

def run():
    prices=load_prices()
    caches=load_cache()
    output=ROOT/'results'
    ledgers, trades, fits, margins, diagnostics, signals_out=[],[],[],[],[],[]
    sensitivity=[]
    latest_example=None
    for cache in caches:
        start=pd.Timestamp(cache['start'])
        index=pd.DatetimeIndex(cache['test_dates'],name='Date')
        train=prices.loc[(prices.index>=start-pd.DateOffset(years=1)) & (prices.index<start)]
        for symbol,detail in cache['marginals'].items():
            margins.append({'window':cache['start'],'symbol':symbol,**{k:v for k,v in detail.items() if k!='candidates'}})
        for first,second in PAIRS:
            pair=f'{first}-{second}'
            paircache=cache['pairs'][pair]
            models=paircache['models']
            if any(not x['success'] for x in models.values()):
                raise ValueError(f'Nonconverged copula: {pair} {start}; inspect cache')
            best=max(models,key=lambda k:models[k]['loglik'])
            single=max(FAMILIES[:5],key=lambda k:models[k]['loglik'])
            bic=min(models,key=lambda k:models[k]['bic'])
            for name,model in models.items():
                fits.append({'window':cache['start'],'pair':pair,'family':name,
                    'loglik':model['loglik'],'aic':model['aic'],'bic':model['bic'],
                    'n_params':model['n_params'],'success':model['success'],
                    'selected':name==best,'best_single':name==single,'selected_bic':name==bic,
                    'parameters':json.dumps(model.get('params',{})),
                    'weights':json.dumps(model.get('weights',[])),
                    'message':model['message']})
            test=prices.loc[index,[first,second]]
            for method in METHODS:
                if method=='Distance':
                    sig,diag=distance_signals(train[[first,second]],test)
                    diagnostics.append({'window':cache['start'],'pair':pair,'method':method,**diag})
                elif method=='Cointegration':
                    sig,diag=cointegration_signals(train[[first,second]],test)
                    diagnostics.append({'window':cache['start'],'pair':pair,'method':method,**diag})
                else:
                    name={'Selected':best,'Best-single':single,'BIC':bic}.get(method,method)
                    h=paircache['conditionals'][name]
                    sig=copula_targets(h['h1'],h['h2'],index)
                ledger,trade=backtest(test,sig.target,sig.reason,
                                     short_ratio=sig.get('short_ratio',1.0))
                for frame in (ledger,trade):
                    frame['pair']=pair; frame['method']=method; frame['window']=cache['start']
                # Each window liquidates, so concatenation is a single fixed-stake account.
                ledgers.append(ledger)
                trades.append(trade)
                if method=='Selected':
                    selected_sig=sig.copy()
                    selected_sig['pair']=pair; selected_sig['window']=cache['start']; selected_sig['family']=best
                    signals_out.append(selected_sig)
                    if pair=='AAPL-GOOG' and cache['start']=='2020-01-01':
                        latest_example=(sig,ledger,trade)
            # Preset threshold and exit variants use the same formation fits.
            h=paircache['conditionals'][best]
            for threshold in np.round(np.arange(.05,.551,.05),2):
                for rule in ['either','both']:
                    sig=copula_targets(h['h1'],h['h2'],index,threshold=threshold,exit_rule=rule)
                    ledger,trade=backtest(test,sig.target,sig.reason)
                    sensitivity.append({'window':cache['start'],'pair':pair,'threshold':threshold,'exit_rule':rule,
                        'return_net':ledger.return_net.sum(),'return_gross':ledger.return_gross.sum(),
                        'turnover':ledger.turnover.sum(),'trades':len(trade),'days':len(ledger)})
        print('Analyzed',cache['start'],flush=True)

    daily=pd.concat(ledgers).reset_index().rename(columns={'date':'Date'})
    alltrades=pd.concat([t for t in trades if not t.empty],ignore_index=True)
    fit_table=pd.DataFrame(fits)
    fit_table.to_csv(output/'copula_fits.csv',index=False)
    pd.DataFrame(margins).to_csv(output/'marginal_fits.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'formation_diagnostics.csv',index=False)
    daily.to_csv(output/'daily_ledger.csv',index=False)
    alltrades.to_csv(output/'trades.csv',index=False)
    pd.concat(signals_out).rename_axis('Date').reset_index().to_csv(output/'selected_signals.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(output/'sensitivity_windows.csv',index=False)

    rows=[]; monthly_rows=[]; portfolio_daily=[]; pair_ledgers={}
    for method in METHODS:
        each=[]
        for first,second in PAIRS:
            pair=f'{first}-{second}'
            ledger=daily.loc[(daily.method==method)&(daily.pair==pair)].set_index('Date').sort_index()
            ledger.index=pd.to_datetime(ledger.index)
            selected_trades=alltrades.loc[(alltrades.method==method)&(alltrades.pair==pair)]
            pair_ledgers[(method,pair)]=ledger
            each.append(ledger)
            for basis,column in [('net','return_net'),('gross','return_gross')]:
                row={'method':method,'pair':pair,'basis':basis,**summary(ledger[column])}
                row.update({'trades':len(selected_trades),'win_rate_pct':100*selected_trades['pnl_'+basis].gt(0).mean(),
                    'converged_pct':100*selected_trades.reason.ne('terminal').mean(),
                    'mean_holding_days':selected_trades.holding_days.mean(),
                    'turnover':ledger.turnover.sum(),
                    'active_days_pct':100*ledger.active.mean(),
                    'cost_pct':100*ledger.transaction_cost.sum()/CAPITAL,
                    'break_even_cost_bps':1e4*ledger.return_gross.sum()/ledger.turnover.sum() if ledger.turnover.sum()>0 else np.nan})
                rows.append(row)
        monthly=monthly_capital_returns(each)
        monthly['method']=method
        monthly_rows.append(monthly.reset_index().rename(columns={'date':'Date'}))
        rnet=pd.concat([x.return_net for x in each],axis=1).mean(axis=1)
        rgross=pd.concat([x.return_gross for x in each],axis=1).mean(axis=1)
        p=pd.DataFrame({'return_net':rnet,'return_gross':rgross,'method':method})
        portfolio_daily.append(p.reset_index().rename(columns={'date':'Date'}))
        for basis,r in [('net',rnet),('gross',rgross)]:
            total_turnover=sum(x.turnover.sum() for x in each)/3
            rows.append({'method':method,'pair':'Portfolio','basis':basis,**summary(r),
                'trades':sum(int(x.entry_event.sum()) for x in each),
                'turnover':total_turnover,'cost_pct':100*sum(x.transaction_cost.sum() for x in each)/(3*CAPITAL),
                'break_even_cost_bps':1e4*rgross.sum()/total_turnover if total_turnover else np.nan})
    stats=pd.DataFrame(rows)
    stats.to_csv(output/'performance.csv',index=False)
    pd.concat(monthly_rows).to_csv(output/'monthly_returns.csv',index=False)
    portfolios=pd.concat(portfolio_daily)
    portfolios.to_csv(output/'portfolio_daily.csv',index=False)
    # Cost-only and borrowing sensitivities do not alter signals or fixed stakes.
    cost_rows=[]
    for method in ['Selected','Best-single','BIC','Distance','Cointegration']:
        for bps in [0,1,5,10,14.5,20]:
            for borrow in ([0,100,300] if bps==5 else [0]):
                pseries=[]
                for first,second in PAIRS:
                    pair=f'{first}-{second}'
                    ledger=pair_ledgers[(method,pair)]
                    # Borrow charges the previous close's marked short position.
                    marks=prices.loc[ledger.index,[first,second]]
                    prior_prices=prices[[first,second]].shift(1).loc[ledger.index]
                    short=((-ledger[['q1_before','q2_before']].to_numpy()).clip(min=0)*prior_prices.to_numpy()).sum(axis=1)
                    ret=ledger.return_gross-bps/1e4*ledger.turnover-borrow/1e4/252*short/CAPITAL
                    pseries.append(ret)
                    cost_rows.append({'method':method,'pair':pair,'cost_bps':bps,'borrow_bps':borrow,**summary(ret)})
                ret=pd.concat(pseries,axis=1).mean(axis=1)
                cost_rows.append({'method':method,'pair':'Portfolio','cost_bps':bps,'borrow_bps':borrow,**summary(ret)})
    pd.DataFrame(cost_rows).to_csv(output/'cost_sensitivity.csv',index=False)
    # Periods are reporting slices; no parameters are chosen using their P&L.
    period_rows=[]
    for label,start,end in [('2006-2015','2006','2015-12-31'),('2016-2025','2016','2025-12-31'),
                            ('GFC 2007-2009','2007','2009-12-31'),('2020','2020','2020-12-31'),('2022','2022','2022-12-31')]:
        for method in ['Selected','Best-single','BIC','Distance','Cointegration']:
            p=portfolios.loc[portfolios.method==method].set_index('Date').sort_index()
            p.index=pd.to_datetime(p.index)
            period_rows.append({'period':label,'method':method,**summary(p.loc[start:end,'return_net'])})
    pd.DataFrame(period_rows).to_csv(output/'subperiods.csv',index=False)
    monthly=pd.concat(monthly_rows).pivot(index='Date',columns='method',values='committed')
    intervals={m:block_mean_interval(monthly.Selected-monthly[m]) for m in ['Distance','Best-single','BIC']}
    # Trade ledger and marked-to-market ledger must reconcile over every pair/method.
    pnl_daily=daily.groupby(['method','pair']).pnl_net.sum()
    pnl_trades=alltrades.groupby(['method','pair']).pnl_net.sum().reindex(pnl_daily.index,fill_value=0)
    error=float((pnl_daily-pnl_trades).abs().max())
    if error>1e-6:
        raise AssertionError(f'Ledger reconciliation failed: {error}')
    audit={'start':str(prices.loc['2006':].index[0].date()),'end':str(prices.index[-1].date()),
        'windows':len(caches),'pair_windows':len(caches)*3,'copula_fits':len(fits),'marginal_fits':len(margins),
        'input_rows_per_symbol':len(prices),'oos_days':len(prices.loc['2006':]),
        'max_trade_daily_pnl_reconciliation_error_dollars':error,
        'all_marginals_converged':bool(pd.DataFrame(margins).converged.all()),
        'all_copula_models_success':bool(fit_table.success.all()),
        'minimum_baseline_equity_per_initial_dollar':float(stats.min_equity_per_initial_dollar.min()),
        'bootstrap_intervals':intervals,
        'design':{'stake_per_pair':CAPITAL,'cost_bps_per_leg_fill':5,'lag_closes':1,
                  'main_entry_threshold':.2,'main_exit':'either','reset_flags':'window only',
                  'return_convention':'fixed initial long-side capital; arithmetic, not compounded'}}
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    from scripts.funding_sensitivity import run_funding
    run_funding(daily)
    print(stats.loc[(stats.pair=='Portfolio')&(stats.basis=='net'),['method','annual_mean_pct','sharpe','max_drawdown_pct','trades']].to_string(index=False))
    print('PnL reconciliation maximum absolute error:',error,flush=True)

if __name__=='__main__':
    run()

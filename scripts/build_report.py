"""Generate report tables and numerical macros from completed analysis outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report'/'tables'

def escape(x):
    return str(x).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')

def num(x,d=2):
    return '--' if pd.isna(x) else f'{x:,.{d}f}'

def table(name,headers,rows,align=None):
    align=align or ('l'+'r'*(len(headers)-1))
    text='\\begin{tabular}{@{}'+align+'@{}}\n\\toprule\n'
    text+=' & '.join(headers)+r' \\'+'\n\\midrule\n'
    for row in rows:
        text+=' & '.join(escape(x) for x in row)+r' \\'+'\n'
    text+='\\bottomrule\n\\end{tabular}\n'
    (OUT/f'{name}.tex').write_text(text,encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=ROOT/'results'
    stats=pd.read_csv(result/'performance.csv')
    fits=pd.read_csv(result/'copula_fits.csv')
    costs=pd.read_csv(result/'cost_sensitivity.csv')
    periods=pd.read_csv(result/'subperiods.csv')
    monthly=pd.read_csv(result/'monthly_returns.csv')
    sensitivity=pd.read_csv(result/'sensitivity_windows.csv')
    audit=json.loads((result/'audit.json').read_text())
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    pairs=['AAPL-GOOG','IBM-SPY','DIA-SPY']
    def get(method,pair='Portfolio',basis='net'):
        return stats.loc[(stats.method==method)&(stats.pair==pair)&(stats.basis==basis)].iloc[0]
    rows=[]
    for pair in pairs:
        n,g=get('Selected',pair),get('Selected',pair,'gross')
        rows.append([pair,num(g.annual_mean_pct),num(n.annual_mean_pct),num(n.cumulative_pnl_pct),num(n.sharpe),num(n.max_drawdown_pct),num(n.trades,0)])
    table('pairs',['Pair',r'Gross/yr (\%)',r'Net/yr (\%)',r'Net total (\%)','Sharpe',r'MDD* (\%)','Trades'],rows)
    rows=[]
    for method in ['Selected','Best-single','BIC','CFG','CtG','Gaussian','Student-t','Clayton','Frank','Gumbel','Distance','Cointegration']:
        n,g=get(method),get(method,basis='gross')
        rows.append([method,num(g.annual_mean_pct),num(n.annual_mean_pct),num(n.sharpe),num(n.max_drawdown_pct),num(n.trades,0)])
    table('portfolio',['Method',r'Gross/yr (\%)',r'Net/yr (\%)','Sharpe',r'MDD* (\%)','Trades'],rows)
    rows=[]
    for method in ['Selected','Best-single','BIC','Distance','Cointegration']:
        m=monthly.loc[monthly.method==method]
        s=get(method)
        rows.append([method,num(m.committed.mean()*1e4),num(m.employed.mean()*1e4),num(s.monthly_hac_t),num(s.monthly_hac_p,3),num(m.active_pairs.mean())])
    table('monthly',['Method','RCC (bps/mo)','REC (bps/mo)','HAC $t$','HAC $p$','Active pairs'],rows)
    rows=[]
    for bps in [0,1,5,10,14.5,20]:
        rows.append([num(bps,1)]+[num(costs.loc[(costs.method==m)&(costs.pair=='Portfolio')&(costs.cost_bps==bps)&(costs.borrow_bps==0),'annual_mean_pct'].iloc[0]) for m in ['Selected','Best-single','BIC','Distance']])
    table('costs',['Cost (bps/fill)','Selected','Best-single','BIC','Distance'],rows)
    rows=[]
    for b in [0,100,300]:
        rows.append([num(b/100,0)]+[num(costs.loc[(costs.method==m)&(costs.pair=='Portfolio')&(costs.cost_bps==5)&(costs.borrow_bps==b),'annual_mean_pct'].iloc[0]) for m in ['Selected','Best-single','BIC','Distance']])
    table('borrow',[r'Borrow (\%/yr)','Selected','Best-single','BIC','Distance'],rows)
    rows=[]
    for period in periods.period.unique():
        p=periods.loc[periods.period==period].set_index('method')
        rows.append([period]+[num(p.loc[m,'annual_mean_pct']) for m in ['Selected','Best-single','BIC','Distance','Cointegration']])
    table('periods',['Period','Selected','Best-single','BIC','Distance','Coint.'],rows)
    rows=[]
    for threshold in sorted(sensitivity.threshold.unique()):
        vals=[]
        for rule in ['either','both']:
            s=sensitivity.loc[(sensitivity.threshold==threshold)&(sensitivity.exit_rule==rule)]
            vals += [num(s.return_net.sum()/s.days.sum()*252*100),num(s.trades.sum(),0)]
        rows.append([num(threshold),*vals])
    table('thresholds',['Entry',r'Either: net/yr (\%)','Trades',r'Both: net/yr (\%)','Trades'],rows)
    rows=[]
    for method,interval in audit['bootstrap_intervals'].items():
        rows.append([method,num(interval['mean_monthly_difference_bps']),num(interval['lower95_bps']),num(interval['upper95_bps'])])
    table('bootstrap',['Selected minus','Mean (bps/mo)',r'Lower 95\%',r'Upper 95\%'],rows)
    rows=[]
    for pair in pairs:
        s=get('Selected',pair)
        rows.append([pair,num(s.converged_pct),num(s.win_rate_pct),num(s.mean_holding_days),num(s.active_days_pct),num(s.break_even_cost_bps)])
    table('trades',['Pair',r'Converged (\%)',r'Winners (\%)','Days/trade',r'Active days (\%)','Break-even bps'],rows)
    rows=[]
    for symbol,info in manifest['files'].items():
        rows.append([symbol,info['first_date'],info['last_date'],num(info['rows'],0),info['dividend_events'],info['split_events']])
    table('coverage',['Symbol','First date','Last date','Rows','Dividends','Splits'],rows,'lllrrr')
    funding=pd.read_csv(result/'reinvested_performance.csv')
    rows=[]
    for method in ['Selected','Best-single','BIC','Distance','Cointegration']:
        row=funding.loc[(funding.method==method)&(funding.pair=='Portfolio')].iloc[0]
        rows.append([method,num(row.cumulative_return_pct),num(row.cagr_pct),num(row.sharpe),num(row.max_drawdown_pct),num(300000*row.final_equity_per_initial_dollar,0)])
    table('funding',['Method',r'Total (\%)',r'CAGR (\%)','Sharpe',r'MDD (\%)',r'Final wealth (\$)'],rows)
    macros={}
    for name,col in [('SelectedAnnual','annual_mean_pct'),('SelectedTotal','cumulative_pnl_pct'),('SelectedSharpe','sharpe'),('SelectedDrawdown','max_drawdown_pct'),('SelectedBreakEven','break_even_cost_bps')]:
        macros[name]=num(get('Selected')[col])
    macros['SelectedDollars']=num(get('Selected').cumulative_pnl_pct/100*300000,0)
    macros['DistanceAnnual']=num(get('Distance').annual_mean_pct)
    macros['BICAnnual']=num(get('BIC').annual_mean_pct)
    macros['SingleAnnual']=num(get('Best-single').annual_mean_pct)
    macros['SelectedTrades']=num(get('Selected').trades,0)
    macros['MixtureShare']=num(100*fits.loc[fits.selected,'family'].isin(['CFG','CtG']).mean(),1)
    macros['MixtureBICShare']=num(100*fits.loc[fits.selected_bic,'family'].isin(['CFG','CtG']).mean(),1)
    macros['OOSDays']=str(audit['oos_days'])
    for prefix,pair in [('AAPL','AAPL-GOOG'),('IBM','IBM-SPY'),('DIA','DIA-SPY')]:
        macros[prefix+'Annual']=num(get('Selected',pair).annual_mean_pct)
        macros[prefix+'Total']=num(get('Selected',pair).cumulative_pnl_pct)
    (ROOT/'report'/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in macros.items())+'\n',encoding='utf-8')
    print('Generated',len(list(OUT.glob('*.tex'))),'result tables and',len(macros),'numerical macros.')

if __name__=='__main__':
    main()

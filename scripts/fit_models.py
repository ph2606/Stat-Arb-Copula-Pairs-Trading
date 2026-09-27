"""Fit frozen formation-window models and cache all out-of-sample probabilities.

No trading return is used to choose a family or its parameters. Independent
six-month windows may run in parallel. Cached inputs are fingerprinted.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_data import verify_data, SYMBOLS
from src.marginals import fit_marginal
from src.copulas import fit_candidates

PAIRS = [('AAPL', 'GOOG'), ('IBM', 'SPY'), ('DIA', 'SPY')]
WINDOWS = pd.date_range('2006-01-01', '2025-07-01', freq='6MS')

def load_prices():
    verify_data()
    frames = {s: pd.read_csv(ROOT/'data'/f'{s}.csv', index_col='Date', parse_dates=True)['Adj Close'] for s in SYMBOLS}
    panel = pd.concat(frames, axis=1)
    if panel.isna().any().any():
        raise ValueError('Common-calendar data required: missing dates are not filled')
    return panel

def fingerprint():
    manifest=verify_data()
    hashes=[manifest['files'][s]['sha256'] for s in SYMBOLS]
    hashes += [hashlib.sha256((ROOT/'src'/f).read_bytes()).hexdigest() for f in ['marginals.py','copulas.py']]
    hashes += ['log_returns;formation12m;trading6m;2006-2025;v1']
    return hashlib.sha256('|'.join(hashes).encode()).hexdigest()

def fit_window(start_text, signature):
    start=pd.Timestamp(start_text)
    path=ROOT/'results'/'fit_cache'/f'{start.date()}.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result.get('fingerprint')==signature:
            return start_text, 'cached', 0.
    begin=time.perf_counter()
    prices=load_prices()
    returns=np.log(prices).diff().dropna()
    train=returns.loc[(returns.index>=start-pd.DateOffset(years=1)) & (returns.index<start)]
    test=returns.loc[(returns.index>=start) & (returns.index<start+pd.DateOffset(months=6))]
    marginal_models={s:fit_marginal(train[s].to_numpy()) for s in SYMBOLS}
    train_pit={s:np.asarray(marginal_models[s].train_pit) for s in SYMBOLS}
    test_pit={s:np.asarray(marginal_models[s].transform_future(test[s].to_numpy())) for s in SYMBOLS}
    result={'fingerprint':signature,'start':str(start.date()),
            'train_dates':[str(d.date()) for d in train.index],
            'test_dates':[str(d.date()) for d in test.index],
            'marginals':{s:m.to_dict() for s,m in marginal_models.items()},'pairs':{}}
    for first,second in PAIRS:
        fitted=fit_candidates(train_pit[first],train_pit[second])
        conditionals={}
        for name,model in fitted.items():
            h1,h2=model.conditional(test_pit[first],test_pit[second])
            if not np.isfinite([h1,h2]).all():
                raise ValueError(f'Invalid conditional: {start} {first} {name}')
            conditionals[name]={'h1':h1.tolist(),'h2':h2.tolist()}
        result['pairs'][f'{first}-{second}']={
            'models':{name:model.to_dict() for name,model in fitted.items()},
            'train_u':train_pit[first].tolist(),'train_v':train_pit[second].tolist(),
            'test_u':test_pit[first].tolist(),'test_v':test_pit[second].tolist(),
            'conditionals':conditionals}
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return start_text,'fitted',time.perf_counter()-begin

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    signature=fingerprint()
    windows=WINDOWS if not args.limit else WINDOWS[:args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fit_window,str(start.date()),signature) for start in windows]
        for i,future in enumerate(as_completed(futures),1):
            start,status,elapsed=future.result()
            print(f'{i}/{len(windows)} {start}: {status} ({elapsed:.1f}s)',flush=True)

if __name__=='__main__':
    main()

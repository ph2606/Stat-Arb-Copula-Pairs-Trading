"""Download a fixed daily Yahoo sample and record verifiable provenance."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ['AAPL', 'GOOG', 'IBM', 'DIA', 'SPY']
START, END = '2004-12-01', '2026-01-01'

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_data():
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    if manifest['request'] != {'start': START, 'end_exclusive': END, 'interval': '1d'}:
        raise ValueError('Unexpected data request in manifest')
    if set(manifest['files']) != set(SYMBOLS):
        raise ValueError('Incomplete symbol manifest')
    for symbol, info in manifest['files'].items():
        path = ROOT / info['path']
        if sha256(path) != info['sha256']:
            raise ValueError(f'Input hash mismatch: {symbol}')
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    # A GitHub clone includes provenance but excludes raw vendor observations.
    # Reuse a cache only when all five observations files are actually present.
    if ((ROOT / 'data_manifest.json').exists() and not args.refresh
            and all((ROOT / 'data' / f'{symbol}.csv').exists() for symbol in SYMBOLS)):
        verify_data()
        print('Using five verified cached inputs.')
        return
    (ROOT / 'data').mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(ROOT / 'data' / '.yf_cache'))
    manifest = {'provider': 'Yahoo Finance via yfinance',
                'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                'request': {'start': START, 'end_exclusive': END, 'interval': '1d'},
                'python': platform.python_version(), 'yfinance': yf.__version__, 'files': {}}
    for symbol in SYMBOLS:
        for attempt in range(3):
            try:
                frame = yf.Ticker(symbol).history(start=START, end=END, interval='1d',
                           auto_adjust=False, actions=True, repair=False, raise_errors=True)
                if frame.empty:
                    raise ValueError('Empty provider response')
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        frame.index = frame.index.tz_localize(None).normalize()
        frame.index.name = 'Date'
        frame = frame.loc[(frame.index >= START) & (frame.index < END)]
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f'Invalid date index: {symbol}')
        for col in ['Close', 'Adj Close', 'Open']:
            if not np.isfinite(frame[col]).all() or not frame[col].gt(0).all():
                raise ValueError(f'Invalid {col}: {symbol}')
        path = ROOT / 'data' / f'{symbol}.csv'
        frame.to_csv(path, float_format='%.12g')
        manifest['files'][symbol] = {'path': str(path.relative_to(ROOT)).replace('\\', '/'),
            'sha256': sha256(path), 'rows': len(frame),
            'first_date': str(frame.index.min().date()), 'last_date': str(frame.index.max().date()),
            'missing_adjusted_closes': int(frame['Adj Close'].isna().sum()),
            'dividend_events': int(frame.get('Dividends', pd.Series(dtype=float)).ne(0).sum()),
            'split_events': int(frame.get('Stock Splits', pd.Series(dtype=float)).ne(0).sum()),
            'source_url': f'https://finance.yahoo.com/quote/{symbol}/history/'}
        print(symbol, len(frame), manifest['files'][symbol]['first_date'], manifest['files'][symbol]['last_date'], flush=True)
    (ROOT / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify_data()

if __name__ == '__main__':
    main()

"""A fresh GitHub clone has a manifest but no ignored vendor observations."""
import json
import sys
from types import SimpleNamespace

import pandas as pd

from scripts import download_data


def test_manifest_without_raw_files_downloads_and_rebuilds_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, 'ROOT', tmp_path)
    monkeypatch.setattr(download_data, 'SYMBOLS', ['TEST'])
    monkeypatch.setattr(sys, 'argv', ['download_data.py'])
    (tmp_path / 'data_manifest.json').write_text('{}')
    calls = []

    def history(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({'Open': [100., 101.], 'Close': [101., 102.],
                             'Adj Close': [101., 102.], 'Dividends': [0., 0.],
                             'Stock Splits': [0., 0.]},
                            index=pd.date_range('2005-01-03', periods=2, tz='UTC'))

    provider = SimpleNamespace(__version__='test',
        set_tz_cache_location=lambda path: None,
        Ticker=lambda symbol: SimpleNamespace(history=history))
    monkeypatch.setattr(download_data, 'yf', provider)
    download_data.main()
    assert len(calls) == 1
    assert (tmp_path / 'data' / 'TEST.csv').exists()
    manifest = download_data.verify_data()
    assert manifest['files']['TEST']['rows'] == 2
    assert json.loads((tmp_path / 'data_manifest.json').read_text()) == manifest

import numpy as np
import pandas as pd
from scripts.funding_sensitivity import reinvested_account

def test_resizes_only_on_entry_not_daily():
    dates=pd.date_range('2020-01-01',periods=5)
    d=pd.DataFrame({'entry_event':[True,False,False,True,False],
        'position_before':[0,1,1,0,1],'position':[1,1,0,1,0],
        'return_net':[-.001,.20,-.099,-.001,.101]},index=dates)
    r=reinvested_account(d)
    assert np.isclose(r.equity_per_initial_dollar.iloc[2],1.1)
    assert np.isclose(r.equity_per_initial_dollar.iloc[-1],1.1*1.1)
    np.testing.assert_allclose(r.stake_multiplier,[1,1,1,1.1,1.1])

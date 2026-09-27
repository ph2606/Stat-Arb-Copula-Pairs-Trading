import numpy as np
import pandas as pd
from src.signals import copula_targets, spread_targets

def test_flags_entry_exit_and_no_post_exit_reset():
    idx = pd.date_range('2020-01-01', periods=4)
    sig = copula_targets([.1,.9,.8,.9], [.9,.3,.0,.1], idx)
    assert sig.target.tolist() == [1,0,-1,-1]
    np.testing.assert_allclose(sig.flag1,[-.4,0,.3,.7])

def test_exit_ambiguity_has_observable_effect():
    idx = pd.date_range('2020-01-01', periods=3)
    a,b=[.1,.9,.5],[.9,.6,.5]
    assert copula_targets(a,b,idx,exit_rule='either').target.iloc[1] == 0
    assert copula_targets(a,b,idx,exit_rule='both').target.iloc[1] == 1

def test_signal_prefix_does_not_depend_on_future():
    idx=pd.date_range('2020-01-01',periods=20)
    rng=np.random.default_rng(8)
    a,b=rng.uniform(size=(2,20))
    expected=copula_targets(a,b,idx).iloc[:10]
    a[10:]=1; b[10:]=0
    pd.testing.assert_frame_equal(expected,copula_targets(a,b,idx).iloc[:10])

def test_distance_direction_and_zero_crossing():
    z=pd.Series([0,3,4,-.1,-3,0.1])
    assert spread_targets(z).target.tolist() == [0,-1,-1,0,1,0]

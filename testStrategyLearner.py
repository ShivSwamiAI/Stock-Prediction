"""
Implement a strategy learner.  (c) 2015 Tucker Balch
2016-10-20 Added "quicksand" and uncertain actions.
"""

import datetime as dt
import pandas as pd
import util as ut
import numpy as np
import random as rand
import time
import math
import StrategyLearner as sl


def test_code():
    learner = sl.StrategyLearner(verbose = False, impact = 0.000) # constructor
    learner.addEvidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
    # df_trades = learner.testPolicy(symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000)


if __name__=="__main__":
    test_code()

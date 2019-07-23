"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import QLearner as ql
import random
from indicators import compute_indicators

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    def author(self):
        return 'rsadiq3'

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "JPM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # add your code to do learning here
        indicators = compute_indicators(symbols=['JPM'], sd=sd, ed=ed, lookback=14, gen_plot=False)
        states = self.discreticize(indicators)
        states = states.astype(int)
        print "max_state", pd.to_numeric(states, downcast='integer')
        num_states = states.max() + 1

        learner = ql.QLearner(num_states=num_states,\
            num_actions = 3, \
            alpha = 0.2, \
            gamma = 0.9, \
            rar = 0.98, \
            radr = 0.999, \
            dyna = 0, \
            verbose=False)

        action = learner.querysetstate(states[0]) #set state and get first action

        # set initial state on first day
        print action

        # Calculate daily returns
        prices = ut.get_data([symbol], pd.date_range(sd, ed))
        daily_returns = ((prices / prices.shift(1)) - 1) * 100
        # print daily_returns[symbol]
        # print state
        for index, state in states[1:].iteritems():
            reward = daily_returns.loc[index, symbol]
            action = learner.query(state, reward)
            print reward, action


        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        # example use with new colname
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "JPM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        indicators = compute_indicators(symbols=['JPM'], sd=sd, ed=ed, lookback=14, gen_plot=False)
        self.discreticize(indicators)

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing
        trades.values[0,:] = 1000 # add a BUY at the start
        trades.values[40,:] = -1000 # add a SELL
        trades.values[41,:] = 1000 # add a BUY
        trades.values[60,:] = -2000 # go short from long
        trades.values[61,:] = 2000 # go long from short
        trades.values[-1,:] = -1000 #exit on the last day
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

    # this method descretizes indicators
    def discreticize(self, indicators):
        # print indicators.loc[:, :].values
        # print indicators.min()['PSMA']
        # print indicators.max()
        # bins = np.linspace(0, 1.3, 10)
        # digitized = indicators.copy()
        # digitized = np.digitize(indicators[:, :], bins)
        # data = digitized
        # print data
        bins_psma = np.linspace(indicators.min()['PSMA'], indicators.max()['PSMA'], 10)
        bins_bbp = np.linspace(indicators.min()['BBP'], indicators.max()['BBP'], 10)
        bins_macd = np.linspace(indicators.min()['MACD'], indicators.max()['MACD'], 10)
        # print 'Bins PSMA ===> ', bins_psma
        # print 'Bins BBP ===> ', bins_bbp
        # print 'Bins MACD ===> ', bins_macd
        digitized = indicators.copy()
        digitized['PSMA'] = np.digitize(indicators['PSMA'], bins_psma, right=True)
        digitized['BBP'] = np.digitize(indicators['BBP'], bins_bbp, right=True)
        digitized['MACD'] = np.digitize(indicators['MACD'], bins_macd, right=True)
        digitized = digitized['PSMA'].map(str) + digitized['BBP'].map(str) + digitized['MACD'].map(str)
        return digitized


if __name__=="__main__":
    # testPolicy()
    print "One does not simply think up a strategy"

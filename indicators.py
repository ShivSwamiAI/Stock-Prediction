'''
MC2-P2: Manual Strategy
Authored by: Rukayat Sadiq - rsadiq3 - 903370071
'''


import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'rsadiq3' # My Georgia Tech username.

def compute_indicators(symbols=['JPM'], sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=14, gen_plot=False):
    new_sd = sd - dt.timedelta(days=lookback + 10) # increase the start date to allow more data to be read to prevent NA values in SMA and BBands calculations.

    date_range = pd.date_range(new_sd, ed)
    prices = get_data(symbols, date_range)

    sma, price_to_sma = compute_sma(sd, ed, prices, symbols, lookback, gen_plot)
    upper_band, lower_band, bbp = compute_bbands(sd, ed, sma, prices, symbols, lookback, gen_plot)
    macd = compute_macd(sd, ed, prices, symbols, lookback, gen_plot)
    print("I got called")
    # Combine the indicators into one dataframe, for ease of processing at point of use
    indicators_df = pd.concat([price_to_sma, bbp, macd], keys=['PSMA', 'BBP', 'MACD'], axis=1)
    indicators_df.columns = ['PSMA', 'BBP', 'MACD']
    return indicators_df

def compute_sma(sd, ed, prices, symbols, lookback, gen_plot):
    sma = prices.rolling(window=lookback, min_periods=lookback).mean()

    # extract prices and sma for specified start to end dates
    prices = prices[sd:ed]
    sma = sma[sd:ed]
    price_to_sma = prices / sma


    if gen_plot:
        # Plot SMA & Prices Curves
        prices = prices / prices.iloc[0, :] # normalize prices
        normed_sma = sma / sma.iloc[0, :]
        fig, (ax) = plt.subplots(1, figsize=(20,7))
        plt.plot(prices[symbols], label='prices')
        plt.plot(price_to_sma[symbols], label='Price/SMA Ratio')
        plt.plot(normed_sma[symbols], label='sma-14')
        plot_graph(ax, sd, ed, 'Dates', 'Adj Price', 'Prices vs SMA Indicator', 'sma')

    return sma[symbols], price_to_sma[symbols]

def compute_bbands(sd, ed, sma, prices, symbols, lookback, gen_plot):
    rolling_std = prices.rolling(window=lookback, min_periods=lookback).std()
    upper_band = sma + (2 * rolling_std)
    lower_band = sma - (2 * rolling_std)
    bbp = ((prices - lower_band) / (upper_band - lower_band))
    upper_band = upper_band[sd:ed]
    lower_band = lower_band[sd:ed]
    bbp = bbp[sd:ed]

    if gen_plot:
        plt.figure(1, figsize=(20, 20))
        ax1 = plt.subplot2grid((9, 1), (0,0), rowspan=6)
        ax2 = plt.subplot2grid((9, 1), (7,0), rowspan=2)
        ax1.plot(prices[symbols], label='prices')
        ax1.plot(sma[symbols], label='sma-14')
        ax1.plot(lower_band[symbols], label='lower_band')
        ax1.plot(upper_band[symbols], label='upper_band')
        plt.title('JPM BBands Indicator')
        plt.ylabel('Adj Close Prices')
        plt.xlim(sd, ed)
        ax1.legend(loc=0)
        # ax2 = plt.subplot2grid((6, 1), (1,0))
        ax2.plot(bbp[symbols], label='bbands %')
        ax2.legend(loc=0)
        plt.title('%BBands Indicator')
        plt.xlabel('Dates')
        plt.ylabel('% BBands')
        plt.subplots_adjust(hspace=0.6)
        plt.xlim(sd, ed)

        plt.show()
        plt.savefig('bbands')

    return upper_band[symbols], lower_band[symbols], bbp[symbols]

def compute_macd(sd, ed, prices, symbols, lookback, gen_plot):
    # compute 12 days and 26 day exponential moving averages
    ema_12 = prices.ewm(span=12, min_periods=1).mean()
    ema_26 = prices.ewm(span=26, min_periods=1).mean()
    macd = ema_12 - ema_26
    macd = macd[sd:ed] # extract macd for specified start & end dates
    signal_ema = prices.ewm(span=9, min_periods=1).mean() # use 9-day ema as signal line
    signal_ema = signal_ema / signal_ema.iloc[0, :] # normalize signal_ema

    if gen_plot:
        fig, (ax) = plt.subplots(1, figsize=(20,7))
        plt.plot(macd[symbols], label='MACD')
        plt.plot(signal_ema[symbols], label='Signal Line')
        plot_graph(ax, sd, ed, 'Dates', 'Adj Price', 'MACD Indicator', 'macd')
    return macd[symbols]

def plot_graph(ax, sd, ed, xlabel, ylabel, title, figname):
    # ax.set_fontsize(15)
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%b'))
    plt.gcf().autofmt_xdate()
    plt.xlim(sd, ed)
    plt.legend(loc=0)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=15)
    plt.show()
    plt.savefig(figname)

# def test_code():
#     compute_indicators()
#
#
# if __name__ == "__main__":
#     # test_code()
#     print "Computing Technical Indicators"

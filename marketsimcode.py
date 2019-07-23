"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

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

def compute_portvals(orders_df, start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here
    # orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
    symbols = orders_df.Symbol.unique().tolist()

    # map +/- value to share depending on BUY/SELL ORDER
    order_mask = orders_df.loc[:, 'Order'] == 'SELL'
    orders_df['Shares'] = -orders_df['Shares'].where(order_mask, -orders_df['Shares'], axis=0)

    # grab start and end date from orders_df
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    date_range = pd.date_range(start_date, end_date)

    # read in historical Adj Close Prices for date range
    prices_all = get_data(symbols, date_range)
    prices_df = prices_all[symbols]
    prices_df['Cash'] = 1.0 # add cash column with default value 1.0

    # Handle possible potentially missing values in range as a result of including non-trading days or actual missing values
    prices_df.fillna(method='ffill', inplace=True)
    prices_df.fillna(method='bfill', inplace=True)

    # Create copy to be used for trading calculations
    trades_df = prices_df.copy()
    trades_df[symbols] = 0 # set all values in symbol columns to 0
    trades_df['Cash'] = 0 # add cash column with default value 1.0

    # scan through orders_df to caluculate trading changes on transaction days
    for index, row in orders_df.iterrows():
        trades_df.loc[index, row.Symbol] = trades_df.loc[index, row.Symbol] + row.Shares
        trades_df.loc[index, 'Cash'] = trades_df.loc[index, 'Cash'] + (-1 * row.Shares * prices_df.loc[index, row.Symbol]) - commission - (impact * np.abs(row.Shares) * prices_df.loc[index, row.Symbol])

    # make a copy of trading dataframe and add start_val to first Cash entry
    trading_with_start_val = trades_df.copy()
    trading_with_start_val.loc[start_date].Cash = trading_with_start_val.loc[start_date].Cash + start_val

    # create holdings_df as cummulative sum of trading dataframe which has start_val included
    holdings_df = trading_with_start_val.cumsum()

    # create values dataframe (monetary value of each assets daily) which is the product of holdings_df and prices_df
    values_df = holdings_df * prices_df

    port_vals = values_df.sum(axis=1)

    return port_vals


def compute_portfolio_stats(df_prices, sv=100000, rfr=0.0, sf=252.0):
    port_val = df_prices.copy()

    # Calculate cumulative returns
    cr = ((port_val[-1] / port_val[0]) - 1)

    # Calculate Daily_returns and its derivative statistics
    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:] # drop first row of daily returns since it contains nan values
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = np.sqrt(sf) * ((adr - rfr) / sddr)
    return cr, adr, sddr, sr

def test_code(df_list, figname, commission=9.95, impact=0.005):
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    sv = 100000

    # Process Best Possible Portfolio & Benchmark orders
    portfolio = compute_portvals(df_list[0], start_val = sv, commission=commission, impact=impact)
    benchmark = compute_portvals(df_list[1], start_val = sv, commission=commission, impact=impact)
    # portfolio_benchmark = pd.concat([portfolio, benchmark], axis=1)
    # portfolio_benchmark.columns = ['Portfolio', 'Benchmark']

    # Normalize portfolio and Benchmark dataframe
    normed_benchmark = benchmark / benchmark.iloc[0]
    normed_portfolio = portfolio / portfolio.iloc[0]
    # plot_data(portfolio_benchmark, title="Best Possible Portfolio JPM", xlabel="Date", ylabel="Adj Close Price")
    fig, (ax) = plt.subplots(1, figsize=(20,7))
    plt.plot(normed_portfolio, label='Portfolio', color='k')
    plt.plot(normed_benchmark, label='Benchmark', color='b')
    plt.legend(loc=0)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Adj Close Price', fontsize=15)
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%b'))
    plt.gcf().autofmt_xdate()

    plt.show()
    plt.savefig(figname)

    if isinstance(portfolio, pd.DataFrame):
        portfolio = portfolio[portfolio.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = df_list[1].index[0]
    end_date = df_list[1].index[-1]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portfolio)
    cum_ret_benchmark, avg_daily_ret_benchmark, std_daily_ret_benchmark, sharpe_ratio_benchmark = compute_portfolio_stats(benchmark)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Best Possible Portfolio: {}".format(sharpe_ratio)
    print "Sharpe Ratio of benchmark : {}".format(sharpe_ratio_benchmark)
    print
    print "Cumulative Return of Best Possible Portfolio: {}".format(cum_ret)
    print "Cumulative Return of benchmark : {}".format(cum_ret_benchmark)
    print
    print "Standard Deviation of Best Possible Portfolio: {}".format(std_daily_ret)
    print "Standard Deviation of benchmark : {}".format(std_daily_ret_benchmark)
    print
    print "Average Daily Return of Best Possible Portfolio: {}".format(avg_daily_ret)
    print "Average Daily Return of benchmark : {}".format(avg_daily_ret_benchmark)
    print
    print "Final Portfolio Value: {}".format(portfolio[-1])

if __name__ == "__main__":
    test_code()

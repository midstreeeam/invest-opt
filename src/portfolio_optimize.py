# -*- coding: utf-8 -*-
import sys
sys.path.append("..")

from optfolio.report import plot_traces, returns_table
from optfolio.returns_projection import cumulative_n_period_returns, sample_returns, mcmc_sample_returns
from optfolio.optimize import Optimizer as Opt1
from optfolio.optimize2 import Optimizer as Opt2
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from plots import *

# load stock data
data = pd.read_csv('stock_data.csv', header=[
                   0, 1], index_col=0, parse_dates=True)
daily_returns = (data['Close'] / data['Close'].shift(1) - 1)[1:]

# load spy500 data
spy_data = pd.read_csv("spy_data.csv")
spy_daily_returns = (spy_data['Close'] / spy_data['Close'].shift(1) - 1)[1:]

# # plot stocks daily return
# plot_all_stock_daily_return(daily_returns)
# # plot stocks corr
# plot_stocks_corr_heat(daily_returns)
# plot S&P500
# plot_spy_daily_return(spy_daily_returns)

"""## Optimization"""

TEST_YEARS = 0
TRAIN_END_DATE = data.index.max() - timedelta(days=TEST_YEARS * 365)

# Ensure the index is of type DatetimeIndex
daily_returns.index = pd.to_datetime(daily_returns.index)
spy_daily_returns.index = pd.to_datetime(spy_daily_returns.index)

train = daily_returns[(daily_returns.index < TRAIN_END_DATE)].fillna(0)
test = daily_returns[(daily_returns.index >= TRAIN_END_DATE)].fillna(0)

spy_train = spy_daily_returns[(spy_daily_returns.index < TRAIN_END_DATE)].fillna(0)
spy_test = spy_daily_returns[(spy_daily_returns.index >= TRAIN_END_DATE)].fillna(0)

# plot rand preformance
# plot_random_preformance(train)

compare_files = [
    '../.data/3000_403iter.json',
    # '../.data/3000_355iter.json',
    # '../.data/3000_362iter.json',
    # '../.data/3000_opt_374iter.json',
    # '../.data/3000_opt_416iter.json',
    '../.data/3000_opt_468iter.json',
    # '../.data/3000_opt2_387iter.json',
    '../.data/3000_opt2_411iter.json',
    # '../.data/3000_opt2_456iter.json',
]

stats_lst, solutions_lst = zip(*[read_log(file) for file in compare_files])

labels = ['nsga-ii', 'select_opt', 'select_opt+dyn_mut+init_opt']
args = {
    'solutions_lst': solutions_lst,
    'train': train,
    'labels': labels
}

# # add the '__main__' guard for multi-processing
# if __name__ == '__main__':
#     run_mcmc(**args)

compare_hvs(stats_lst,labels)
compare_solutions(**args)
compare_sharpe(**args)
compare_density(**args)


# for i in range(1):
#     optimizer = Opt2(
#         mutation_sigma=1.0, 
#         verbose=True, 
#         max_iter=500,
#         population_size=3000,
#         mutation_p_decay=0.995,
#         opt=True
#     )

#     optimizer = Opt1(
#         mutation_sigma=1.0, 
#         verbose=True, 
#         max_iter=500,
#         population_size=3000,
#         mutation_p_decay=0.995,
#     )

#     solutions, stats = log_opt(train,optimizer)

#     # plot_hv(stats)

#     # plot_solutions(train, solutions)


# calculate dominance rate



# CAPITAL = 38000

# def print_allocation(data, allocations, prices):
#     for ticker_id in np.argsort(-allocations):
#         print('%s - %.4f, $%.2f USD, %.2f shares' % (data.columns[ticker_id], allocations[ticker_id] * 100,
#               CAPITAL * allocations[ticker_id], (CAPITAL * allocations[ticker_id]) / prices[data.columns[ticker_id]]))


# print_allocation(train, solution, data['Close'].iloc[-1])

# ','.join(train.columns)



# """### S&P 500 MC Projection"""
# print("Annualized return: %.6f" % (np.mean(spy_train + 1) ** 252 - 1))
# print("Annualized volatility: %.6f" % (np.sqrt(np.var(spy_train) * 252)))

# spy_traces = sample_returns(spy_train, 10 * 252, n_traces=100000)

# spy_cum_returns = plot_traces(spy_traces)
# returns_table(spy_cum_returns)


# """### MC Projection"""
# traces = sample_returns(ret, 10 * 252, n_traces=100000)

# cum_returns = plot_traces(traces)
# returns_table(cum_returns)


# """### MCMC Projection"""
# mcmc_traces = mcmc_sample_returns(ret, 10 * 252, n_traces=100000, mc_states=10, n_jobs=-1)
# mcmc_cum_returns = plot_traces(mcmc_traces)
# returns_table(mcmc_cum_returns)
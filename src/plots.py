import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import imageio


from optfolio import YEAR_BARS

def plot_all_stock_daily_return(daily_returns):
    n_rows = int(np.ceil(daily_returns.shape[1] / 2.0))
    plt.figure()
    for i, ticker in enumerate(daily_returns.columns):
        plt.subplot(n_rows, 2, i + 1)
        plt.title(ticker)
        plt.plot(np.cumprod(daily_returns[ticker] + 1) - 1)
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.axhline(0, color='black')
    plt.show()

def plot_spy_daily_return(spy_daily_returns):
    plt.figure()
    plt.plot(np.cumprod(spy_daily_returns + 1) - 1, label='S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative return')
    plt.legend()
    plt.show()

def plot_stocks_corr_heat(daily_returns):
    plt.figure()
    cm = daily_returns.corr()
    mask = (1 - np.tril(np.ones_like(cm))) == 1
    cm[np.eye(cm.shape[0]) == 1] = np.nan
    cm[mask] = np.nan
    # cm[cm <= 0.5] = np.nan
    sns.heatmap(cm)
    plt.show()

def plot_random_preformance(train):
    # random port weights
    rand_weights = random_population(train.shape[1], 100000)
    rand_solutions = annualized_portfolio_performance(train, rand_weights)
    plt.figure()
    plt.title('Random portfolios')
    plt.scatter(rand_solutions[:, 1], rand_solutions[:, 0], alpha=.5)
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.show()

def plot_solutions(data, solutions):
    rand_weights = random_population(data.shape[1], 100000)
    rand_solutions = annualized_portfolio_performance(data, rand_weights)
    ov = annualized_portfolio_performance(data, solutions)
    plt.figure(figsize=(20, 10))
    plt.title('Solutions')
    plt.scatter(rand_solutions[:, 1], rand_solutions[:, 0], alpha=.5)
    plt.scatter(ov[:, 1], ov[:, 0], alpha=.5)
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.show()

def random_population(n_assets, population_size):
    weights = np.random.uniform(0, 1, size=(population_size, n_assets))
    return weights / weights.sum(axis=-1).reshape((-1, 1))
def annualized_portfolio_return(returns, weights):
    weighted_returns = np.matmul(weights, np.mean(returns.values, 0))
    return (weighted_returns + 1) ** YEAR_BARS - 1
def annualized_portfolio_volatility(returns, weights):
    variance = np.sum(weights * np.matmul(weights,
                    np.cov(returns.T.values)), -1)
    return np.sqrt(variance) * np.sqrt(YEAR_BARS)
def annualized_portfolio_performance(returns, weights):
    return np.stack([
        annualized_portfolio_return(returns, weights),
        annualized_portfolio_volatility(returns, weights)
    ], -1)


def gen_gif(train,optimizer):
    def plot_solutions(data, solutions, rand_solutions, filename):
        ov = annualized_portfolio_performance(data, solutions)
        plt.figure(figsize=(20, 10))
        plt.title('Solutions')
        plt.scatter(rand_solutions[:, 1], rand_solutions[:, 0], alpha=.2)
        plt.scatter(ov[:, 1], ov[:, 0], alpha=.5, color='red')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.savefig(filename)
        plt.close()

    image_files = []
    rand_weights = random_population(train.shape[1], 100000)
    rand_solutions = annualized_portfolio_performance(train, rand_weights)
    for idx, (solutions, stats) in enumerate(optimizer.run_generator(train.values)):
        filename = f"pareto_front_{idx}.png"
        plot_solutions(train, solutions, rand_solutions, filename)
        image_files.append(filename)

    with imageio.get_writer('pareto_evolution.gif', mode='I', duration=0.5) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Optionally, remove the individual image files to clean up
    for filename in image_files:
        os.remove(filename)
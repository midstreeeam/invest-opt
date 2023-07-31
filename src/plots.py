import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import imageio


from optfolio import YEAR_BARS
from optfolio.returns_projection import *
FILE_PATH = '../.data/'

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
    rand_weights = random_population(train.shape[1], 1000)
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


def log_opt(train, optimizer):
    def plot_solutions(data, solutions, other_solutions, rand_solutions, filename, generation):
        ov = annualized_portfolio_performance(data, solutions)
        ov_other = annualized_portfolio_performance(data, other_solutions)
        plt.figure()
        plt.title('Solutions')
        plt.scatter(rand_solutions[:, 1], rand_solutions[:, 0], alpha=.1)
        plt.scatter(ov_other[:, 1], ov_other[:, 0], alpha=.5, color='orange')
        plt.scatter(ov[:, 1], ov[:, 0], alpha=.8, color='red')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        
        # Add generation number to the top right corner
        plt.text(0.85, 0.95, f'Gen: {generation}', transform=plt.gca().transAxes)
        
        plt.savefig(filename)
        plt.close()

    stats_log = {}

    image_files = []
    rand_weights = random_population(train.shape[1], 100000)
    rand_solutions = annualized_portfolio_performance(train, rand_weights)
    for idx, (solutions, others, stats) in enumerate(optimizer.run_generator(train.values)):
        filename = f"pareto_front_{idx}.png"
        plot_solutions(train, solutions, others, rand_solutions, filename, idx)
        image_files.append(filename)
        stats_log = {"stats": stats, "solutions": solutions.tolist()}
        gif_name = f"{FILE_PATH+optimizer.filename}_{idx+1}iter.gif"
        log_name = f"{FILE_PATH+optimizer.filename}_{idx+1}iter.json"

    with imageio.get_writer(gif_name, mode='I', duration=0.025) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    with open(log_name,'w',encoding='utf8') as f:
        f.write(json.dumps(stats_log))

    # Optionally, remove the individual image files to clean up
    for filename in image_files:
        os.remove(filename)
    
    return solutions, stats


def plot_hv(stats):
    hypervolume_values = stats['hv']
    plt.figure()
    plt.plot(hypervolume_values)
    plt.title('Hypervolume across iterations')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.show()


def compare_hvs(stats_lst, labels=None):
    plt.figure()
    plt.title('Hypervolume across iterations')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    
    # Default color cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(stats_lst)))
    
    if labels is None:
        labels = [f'Stats {i}' for i in range(len(stats_lst))]
    
    for idx, stats in enumerate(stats_lst):
        hv = stats['hv']
        plt.plot(hv, color=colors[idx], label=labels[idx])
    
    plt.grid(True)
    plt.legend()
    plt.show()

def compare_solutions(solutions_lst, train, labels=None):
    plt.figure()
    plt.title('Solutions')
    
    # Default color cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(solutions_lst)))
    
    if labels is None:
        labels = [f'Solution {i}' for i in range(len(solutions_lst))]
    
    for idx, solutions in enumerate(solutions_lst):
        solutions = annualized_portfolio_performance(train,np.array(solutions))
        plt.scatter(solutions[:, 1], solutions[:, 0], color=colors[idx], label=labels[idx], alpha=.5)
    
    plt.legend()
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.show()

def compare_sharpe(solutions_lst, train, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('sharpe')

    # Default color cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(solutions_lst)))

    if labels is None:
        labels = [f'Solution {i}' for i in range(len(solutions_lst))]

    for idx, solutions in enumerate(solutions_lst):
        solutions = np.array(solutions)
        ov = annualized_portfolio_performance(train, solutions)
        sharpe = ov[:, 0] / ov[:, 1]

        ax.scatter(ov[:, 1], ov[:, 0], sharpe, color=colors[idx], label=labels[idx], alpha=.5)

        print(f"\nsolution set {idx}: \n",
              f"\tmean: {np.mean(sharpe)}\n",
              f"\tstd: {np.std(sharpe)}\n",
              f"\tbest: {sharpe.max()}\n",
              f"\tworst: {sharpe.min()}\n"
        )
    
    # Setting labels
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.set_zlabel('Sharpe Ratio')
    
    # Displaying the plot
    plt.show()

def compare_density(solutions_lst, train, labels=None):
    plt.figure()
    plt.title('Density of Portfolio Returns for Different Solutions')

    # Default color cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(solutions_lst)))

    if labels is None:
        labels = [f'Solution {i}' for i in range(len(solutions_lst))]

    for idx, solutions in enumerate(solutions_lst):
        ov = annualized_portfolio_performance(train, solutions)
        solution = solutions[np.argmax(ov[:, 0] / ov[:, 1])]

        ret = np.dot(train, solution)
        sns.kdeplot(ret, color=colors[idx], label=labels[idx])

    plt.legend(loc='upper right')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.show()

def run_mcmc(solutions_lst, train, labels=None):

    traces_json = {}
    fname = "../.data/traces.json"

    if labels is None:
        labels = [f'Solution {i}' for i in range(len(solutions_lst))]

    for idx, solutions in enumerate(solutions_lst):
        ov = annualized_portfolio_performance(train, solutions)
        solution = solutions[np.argmax(ov[:, 0] / ov[:, 1])]

        ret = np.dot(train, solution)
        mcmc_traces = mcmc_sample_returns(ret, 10 * 252, n_traces=10000, mc_states=10, n_jobs=10)
        traces_json[labels[idx]] = mcmc_traces.tolist()
        # print(traces_json)

    with open(fname, 'w', encoding='utf8') as f:
        f.write(json.dumps(traces_json))

def read_traces():
    fname = "../.data/traces.json"
    with open(fname, 'r', encoding='utf8') as f:
        data = json.loads(f.read())
    return data

def read_log(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
        log_dict = json.loads(data)
    return log_dict['stats'], log_dict['solutions']
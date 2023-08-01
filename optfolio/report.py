import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _normalize_hist(hist, bins):
    return (
        hist / np.sum(hist),
        bins[:-1] + (bins[1:] - bins[:-1]) / 2.0
    )


def plot_traces(traces, hist_bins=1000, benchmark_returns=None):
    QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    cum_returns = np.cumprod(traces + 1, axis=-1) - 1

    plt.figure(figsize=(20, 10))
    # 95th symmetric confidence interval
    fq = np.percentile(cum_returns, q=[2.5, 97.5], axis=0)
    plt.fill_between(np.arange(
        traces.shape[1]), fq[0], fq[1], alpha=.1, color='gray', label='95% CI')
#     fq = np.percentile(cum_returns, q=[0.5, 99.5], axis=0)
#     plt.fill_between(np.arange(traces.shape[1]), fq[0], fq[1], alpha=.1, color='blue', label='99% CI')
    # Quantiles
    for i, v in enumerate(np.quantile(cum_returns, q=QUANTILES, axis=0)):
        plt.plot(v, label='Q: %dth' % int(QUANTILES[i] * 100))

    if benchmark_returns is not None:
        plt.plot(np.cumprod(1 + benchmark_returns) - 1, label='Benchmark')

    plt.axhline(0, color='black')
    plt.xlabel('Days')
    plt.ylabel('Return')
    plt.legend()
    plt.show()

    # Boxplots
    plt.figure(figsize=(20, 10))
    plt.boxplot(cum_returns[:, 251::252], showfliers=False, whis=[2.5, 97.5])
    plt.xlabel('Year')
    plt.ylabel('Return')
    plt.axhline(0, color='black')
    plt.show()

    def _return_distributions_plot(cum_returns, period_bars=252, period='year'):
        period_returns = cum_returns[:, (period_bars-1)::period_bars]
        range_end = int(np.ceil(np.percentile(
            period_returns[:, -1], q=[99.9])))

        plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 1)
        plt.title('Return density after nth %s' % (period))
        hb = []
        for i in range(period_returns.shape[1]):
            hist, bins = np.histogram(
                period_returns[:, i], bins=hist_bins, density=True, range=(-1, range_end))
            hist, bins = _normalize_hist(hist, bins)
            plt.bar(bins, hist, width=5000 /
                    period_returns.shape[0], alpha=.3, label='%s %d' % (period.title(), i + 1))
            hb.append((hist, bins))
        plt.legend()
        plt.xlabel('Return')
        plt.ylabel('Probability')

        plt.subplot(1, 2, 2)
        plt.title('Min. return P')
        for i, (hist, bins) in enumerate(hb):
            plt.plot(bins, 1 - np.cumsum(hist), label='%s %d' %
                     (period.title(), i + 1))
        plt.axvline(0, color='black')
        plt.xlabel('Return')
        plt.ylabel('P(Return >= X)')
        plt.legend()
        plt.show()

    _return_distributions_plot(
        cum_returns[:, :252], period_bars=21, period='month')
    _return_distributions_plot(cum_returns, period_bars=252, period='year')

    return cum_returns


def returns_table(cum_returns):
    yearly_cum_returns = cum_returns[:, 251::252]
    rows = []
    for i in range(yearly_cum_returns.shape[1]):
        row = [
            i + 1,
            (np.sum(yearly_cum_returns[:, i] < 0) /
             yearly_cum_returns.shape[0]) * 100,
            (np.sum(yearly_cum_returns[:, i] > 0) /
             yearly_cum_returns.shape[0]) * 100
        ]
        for j in range(1, 11):
            row.append(
                (np.sum(yearly_cum_returns[:, i] >= j) / yearly_cum_returns.shape[0]) * 100)

        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=(['Year', '< 0', '> 0'] + ['>= %d' % i for i in range(1, 11)]),
    )
    df.set_index('Year', inplace=True)
    return df



def plot_traces_compare(traces_lst, labels=None):
    """
    Plots boxplots of cumulative returns for a list of traces.

    Args:
    - traces_lst: List of arrays containing traces. 
      Each array has shape (number of traces, number of data points).
    - labels: List of strings to use as labels for the different traces.

    Returns:
    - cum_returns: Array of cumulative returns for the last trace.
    """
    # Default color cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(traces_lst)))
    
    if labels is None:
        labels = [f'Solution {i}' for i in range(len(traces_lst))]

    plt.figure(figsize=(20, 10))
    
    # Dummy lines for legend
    legend_handles = []
    
    gap_between_groups = 1.5  # Control the gap between groups here

    # Go through each trace and plot its boxplot
    for i, traces in enumerate(traces_lst):
        cum_returns = np.cumprod(traces + 1, axis=-1) - 1
        box_data = cum_returns[:, 251::252]
        
        # Plot boxplots with specific positions to avoid overlap
        positions = np.arange(1, box_data.shape[1] + 1) * gap_between_groups + i * 0.3  # Adjust gap here
        boxplot_artist = plt.boxplot(box_data, positions=positions, widths=0.25,
                                     patch_artist=True, showfliers=False, whis=[2.5, 97.5], 
                                     boxprops=dict(facecolor=colors[i]))
        
        # Add the boxplot face color to the dummy legend
        legend_handles.append(plt.Line2D([0], [0], color=colors[i], lw=4))

    # Setting the x-axis ticks and labels
    num_years = box_data.shape[1]
    x_ticks = np.arange(1, num_years + 1) * gap_between_groups + (len(traces_lst) - 1) * 0.3 / 2  # Adjust gap here
    plt.xticks(x_ticks, [str(i + 1) for i in range(num_years)])
    
    plt.xlabel('Year')
    plt.ylabel('Return')
    plt.axhline(0, color='black')
    plt.legend(legend_handles, labels, loc='upper left')
    
    plt.tight_layout()
    plt.show()

    # return cum_returns

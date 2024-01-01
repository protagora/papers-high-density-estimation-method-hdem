import numpy as np
import pandas as pd
from hdem import HDEMethod
import os
import shutil
import matplotlib.pyplot as plt
from scipy.stats import alpha, beta, gamma, weibull_min, norm, expon, chi2, lognorm, t, f, poisson, binom
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import sem, t
import textwrap

# Create (or empty) the 'estimates' directory
estimates_dir = 'estimates'
if os.path.exists(estimates_dir):
    shutil.rmtree(estimates_dir)
os.makedirs(estimates_dir)

def generate_distribution(dist_name, size):
    if dist_name == 'alpha':
        return alpha.rvs(a=3.57, size=size)
    elif dist_name == 'beta':
        return beta.rvs(a=2, b=5, size=size)
    elif dist_name == 'gamma':
        return gamma.rvs(a=2, size=size)
    elif dist_name == 'weibull':
        return weibull_min.rvs(c=1.5, size=size)
    elif dist_name == 'norm':
        return norm.rvs(size=size)
    elif dist_name == 'expon':
        return expon.rvs(size=size)
    elif dist_name == 'chi2':
        return chi2.rvs(df=2, size=size)
    elif dist_name == 'lognorm':
        return lognorm.rvs(s=1, size=size)
    elif dist_name == 't':
        return t.rvs(df=10, size=size)
    elif dist_name == 'f':
        return f.rvs(dfn=5, dfd=2, size=size)
    elif dist_name == 'poisson':
        return poisson.rvs(mu=3, size=size)
    elif dist_name == 'binom':
        return binom.rvs(n=10, p=0.5, size=size)
    elif dist_name == 'combined':
        return np.concatenate([beta.rvs(a=2, b=5, size=size//2), weibull_min.rvs(c=1.5, size=size//2)])
    else:
        raise ValueError("Unknown distribution")

def save_density_plots(data, estimated_densities_hdem, estimated_densities_hist, dist_name, sample_size, num_bins, iteration):
    x_values = np.linspace(np.min(data), np.max(data), len(estimated_densities_hdem))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plotting HDEM Estimate
    axes[0].hist(data, bins=num_bins, density=True, alpha=0.6, label='Sample Data')
    axes[0].plot(x_values, estimated_densities_hdem, label='HDEM Estimate', color='red')
    axes[0].set_title(f'HDEM for {dist_name} (n={sample_size}, bins={num_bins})')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Plotting Regular Histogram
    axes[1].hist(data, bins=num_bins, density=True, alpha=0.6, label='Regular Histogram', color='blue')
    axes[1].plot(x_values, estimated_densities_hist, label='Histogram Estimate', color='green')
    axes[1].set_title(f'Regular Histogram for {dist_name} (n={sample_size}, bins={num_bins})')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()

    filename = f'estimates/compare_{dist_name}_n{sample_size}_bins{num_bins}_iter{iteration}.png'
    plt.savefig(filename)
    plt.close()

def compute_regular_histogram(data, num_bins):
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts, bin_centers

def compute_errors(true_counts, estimated_densities, histogram_densities, bin_width, total_samples, dist_name, iteration, path='tests'):
    """
    Compute error measures for the estimated density against the true distribution and plot them.

    :param true_counts: The counts in each bin of the true histogram.
    :param estimated_densities: The estimated densities for each bin from HDEM.
    :param histogram_densities: The densities from the regular histogram.
    :param bin_width: The width of each bin in the histogram.
    :param total_samples: Total number of samples in the distribution.
    :return: A dictionary of error measures.
    """
    # Convert true counts to a probability distribution
    true_counts = np.array(true_counts, dtype=float)
    total_samples = float(total_samples)
    true_prob_dist = true_counts / total_samples

    # Convert estimated densities to a probability distribution
    estimated_prob_dist = np.array(estimated_densities) * bin_width
    estimated_prob_dist /= np.sum(estimated_prob_dist)

    # Convert histogram densities to a probability distribution
    histogram_prob_dist = np.array(histogram_densities) * bin_width
    histogram_prob_dist /= np.sum(histogram_prob_dist)

    # Calculate error measures
    mse = mean_squared_error(true_prob_dist, estimated_prob_dist)
    mae = mean_absolute_error(true_prob_dist, estimated_prob_dist)

    # Normalized errors
    mise = np.mean((true_prob_dist - estimated_prob_dist) ** 2)
    miae = np.mean(np.abs(true_prob_dist - estimated_prob_dist))
    amise = mise / np.sqrt(total_samples)

    # Plotting the distributions
    x_range = np.linspace(0, len(true_prob_dist), len(true_prob_dist))
    plt.figure()
    plt.plot(x_range, true_prob_dist, color='green', label='True Probability Distribution')
    plt.plot(x_range, estimated_prob_dist, color='red', label='Estimated Probability Distribution (HDEM)')
    # plt.plot(x_range, histogram_prob_dist, color='blue', linestyle='None', marker='o', label='Histogram Probability Distribution')
    plt.xlabel('Bins')
    plt.ylabel('Probability')
    title = '\n'.join(textwrap.wrap(f'True vs Estimated Probability Distribution [bin: {bin_width}, samples count: {total_samples}]', 60))
    plt.title(title)
    plt.grid('on')
    plt.legend()

    # Save the figure
    filename = os.path.join(path, f'plot_{dist_name}_n-{sample_size}_bins-{num_bins}_iter-{iteration}.png')
    plt.savefig(filename)
    plt.close()

    return {
        'MSE': mse,
        'MAE': mae,
        'Normalized MSE': mse / total_samples,
        'Normalized MAE': mae / total_samples,
        'IMSE': mise,
        'MISE': mise * bin_width,
        'MIAE': miae * bin_width,
        'AMISE': amise,
    }


def compute_confidence_interval(data, confidence=0.95):
    """ Compute the confidence interval for a dataset. """
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)
    margin = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin

def accumulate_errors(all_errors, error_key, new_error):
    """ Accumulate errors for multiple runs to compute confidence intervals later. """
    if error_key not in all_errors:
        all_errors[error_key] = []
    all_errors[error_key].append(new_error)

def add_confidence_intervals(result_entry, all_errors):
    """ Add confidence intervals to the result entry. """
    for error_key, errors in all_errors.items():
        mean, ci_lower, ci_upper = compute_confidence_interval(errors)
        result_entry[f'CI_{error_key}_Mean'] = mean
        result_entry[f'CI_{error_key}_Lower'] = ci_lower
        result_entry[f'CI_{error_key}_Upper'] = ci_upper

# def compute_regular_histogram(data, num_bins):
#     counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     densities = counts  # Here, counts are actually densities because density=True
#     return counts, densities, bin_centers

if "__main__" == __name__:

    # Experiment parameters
    sample_sizes = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    bin_counts = [10, 15, 20, 25, 30, 50, 100]
    distributions = ['beta', 'gamma', 'weibull', 'expon', 'chi2', 'poisson', 'binom', 'combined']
    num_iterations = 10  # Number of iterations for each configuration

    results = []

    # Path to save distribution and estimate densities
    path = os.path.join('tests', 'densities')
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)

    for dist_name in distributions:
        for sample_size in sample_sizes:
            for num_bins in bin_counts:
                all_errors = {}
                for iteration in range(num_iterations):
                    data = generate_distribution(dist_name, sample_size)
                    bin_width = (np.max(data) - np.min(data)) / num_bins
                    true_counts, _ = np.histogram(data, bins=num_bins, density=False)

                    # Regular Histogram
                    # _, histogram_densities, _ = compute_regular_histogram(data, num_bins)
                    hist_counts, _ = compute_regular_histogram(data, num_bins)

                    # HDEMethod
                    hde = HDEMethod(observations=data, bin_width=bin_width, anchor=np.min(data))
                    estimated_densities_hdem = [hde.final_density_estimate(x) for x in np.linspace(np.min(data), np.max(data), num_bins)]
                    errors_hdem = compute_errors(true_counts, estimated_densities_hdem, hist_counts, bin_width, sample_size, dist_name, iteration, path)
                    # errors_hist = compute_errors(true_counts, hist_counts, hist_counts, bin_width, sample_size, 'histogram', iteration, path)

                    # Accumulate errors for confidence interval calculation
                    for error_key, error_value in errors_hdem.items():
                        accumulate_errors(all_errors, f'HDEM_{error_key}', error_value)
                    # for error_key, error_value in errors_hist.items():
                    #     accumulate_errors(all_errors, f'Hist_{error_key}', error_value)

                    save_density_plots(data, estimated_densities_hdem, hist_counts, dist_name, sample_size, num_bins, iteration)

                result_entry = {'Distribution': dist_name, 'Sample Size': sample_size, 'Bin Count': num_bins}
                add_confidence_intervals(result_entry, all_errors)
                results.append(result_entry)

    df_results = pd.DataFrame(results)
    df_results.to_csv('experiment_results.csv', index=False)

    print("Experiment completed and results saved.")

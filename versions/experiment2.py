import numpy as np
import pandas as pd
from hdem import HDEMethod
import os
import shutil
import matplotlib.pyplot as plt
from scipy.stats import alpha, beta, gamma, weibull_min, norm, expon, chi2, lognorm, t, f, poisson, binom
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import entropy

# Create (or empty) the 'estimates' directory
estimates_dir = 'estimates'
if os.path.exists(estimates_dir):
    shutil.rmtree(estimates_dir)
os.makedirs(estimates_dir)

# [Rest of the functions, like generate_distribution, save_density_plots, compute_regular_histogram, remain unchanged]
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

def compute_errors(true_counts, estimated_densities, bin_width, total_samples):
    """
    Compute error measures for the estimated density against the true distribution.

    :param true_counts: The counts in each bin of the true histogram.
    :param estimated_densities: The estimated densities for each bin.
    :param bin_width: The width of each bin in the histogram.
    :param total_samples: Total number of samples in the distribution.
    :return: A dictionary of error measures.
    """
    # Convert true counts to a probability distribution
    true_prob_dist = true_counts / total_samples

    # Convert estimated densities to a probability distribution
    estimated_prob_dist = np.array(estimated_densities) * bin_width
    estimated_prob_dist /= np.sum(estimated_prob_dist)

    # Calculate error measures
    mse = mean_squared_error(true_prob_dist, estimated_prob_dist)
    mae = mean_absolute_error(true_prob_dist, estimated_prob_dist)

    # Normalized errors
    mise = np.mean((true_prob_dist - estimated_prob_dist) ** 2)
    miae = np.mean(np.abs(true_prob_dist - estimated_prob_dist))
    amise = mise / np.sqrt(total_samples)

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

if "__main__" == __name__:

    # Experiment parameters
    sample_sizes = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    bin_counts = [10, 15, 20, 25, 30, 50, 100]
    distributions = ['alpha', 'beta', 'gamma', 'weibull', 'norm', 'expon', 'chi2', 'lognorm', 'poisson', 'binom', 'combined']
    distributions = ['beta', 'gamma', 'weibull', 'expon', 'chi2', 'poisson', 'binom', 'combined']

    # Conduct the experiment
    results = []
    iteration = 0

    for dist_name in distributions:
        for sample_size in sample_sizes:
            data = generate_distribution(dist_name, sample_size)
            for num_bins in bin_counts:
                bin_width = (np.max(data) - np.min(data)) / num_bins
                true_counts, _ = np.histogram(data, bins=num_bins, density=False)

                # HDEMethod
                hde = HDEMethod(observations=data, bin_width=bin_width, anchor=np.min(data))
                estimated_densities_hdem = [hde.final_density_estimate(x) for x in np.linspace(np.min(data), np.max(data), num_bins)]
                errors_hdem = compute_errors(true_counts, estimated_densities_hdem, bin_width, sample_size)

                # Regular Histogram
                hist_counts, _ = compute_regular_histogram(data, num_bins)
                errors_hist = compute_errors(true_counts, hist_counts, bin_width, sample_size)

                result_entry = {
                    'Distribution': dist_name,
                    'Sample Size': sample_size,
                    'Bin Count': num_bins
                }
                result_entry.update({f'HDEM_{k}': v for k, v in errors_hdem.items()})
                result_entry.update({f'Hist_{k}': v for k, v in errors_hist.items()})

                results.append(result_entry)
                save_density_plots(data, estimated_densities_hdem, hist_counts, dist_name, sample_size, num_bins, iteration)
                iteration += 1

    df_results = pd.DataFrame(results)
    df_results.to_csv('experiment_results.csv', index=False)

    print("Experiment completed and results saved.")
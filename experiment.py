import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import textwrap
from hdem import HDEMethod
from scipy.stats import alpha, beta, gamma, weibull_min, norm, expon, chi2, lognorm, t, f, poisson, binom
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import sem, t

# Constants
MAX_SAMPLE_SIZE = 100000
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
bin_counts = [10, 15, 20, 25, 30, 50, 100]
distributions = ['beta', 'gamma', 'weibull', 'expon', 'chi2', 'poisson', 'binom', 'combined']
num_iterations = 10  # Number of iterations for each configuration

# Create directories
estimates_dir = 'estimates'
path = os.path.join('tests', 'densities')
for directory in [estimates_dir, path]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Functions
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

def compute_true_density(data, num_bins):
    counts, _ = np.histogram(data, bins=num_bins, density=True)
    return counts

def compute_errors_and_plot(true_density, hist_density, hde_density, dist_name, sample_size, num_bins, iteration, path):
    # Calculate error measures and plot
    # Convert densities to probability distributions
    true_prob_dist = np.array(true_density) / np.sum(true_density)
    hist_prob_dist = np.array(hist_density) / np.sum(hist_density)
    hde_prob_dist = hde_density # Already a density function

    # Calculate error measures
    mse_hist = mean_squared_error(true_prob_dist, hist_prob_dist)
    mse_hde = mean_squared_error(true_prob_dist, hde_prob_dist)
    mae_hist = mean_absolute_error(true_prob_dist, hist_prob_dist)
    mae_hde = mean_absolute_error(true_prob_dist, hde_prob_dist)

    # Normalized errors
    mise_hist = np.mean((true_prob_dist - hist_prob_dist) ** 2)
    mise_hde = np.mean((true_prob_dist - hde_prob_dist) ** 2)
    miae_hist = np.mean(np.abs(true_prob_dist - hist_prob_dist))
    miae_hde = np.mean(np.abs(true_prob_dist - hde_prob_dist))
    amise_hist = mise_hist / np.sqrt(sample_size)
    amise_hde = mise_hde / np.sqrt(sample_size)

    # Plotting the densities
    x_range = np.linspace(0, len(true_density), len(true_density))
    plt.figure()
    plt.plot(x_range, true_prob_dist, color='red', label='True Density')
    plt.plot(x_range, hist_prob_dist, color='blue', label='Histogram Estimated Density')
    plt.plot(x_range, hde_prob_dist, color='green', label='HDEM Estimated Density')
    plt.xlabel('Bins')
    plt.ylabel('Density')
    title = '\n'.join(textwrap.wrap(f'Density Comparison [Distribution: {dist_name}, Sample Size: {sample_size}, Bins: {num_bins}]', 60))
    plt.title(title)
    plt.legend()
    plt.grid('on')

    # Save the figure
    filename = os.path.join(path, f'density_plot_{dist_name}_n{sample_size}_bins{num_bins}_iter{iteration}.png')
    plt.savefig(filename)
    plt.close()

    return {
        'Hist_MSE': mse_hist,
        'HDE_MSE': mse_hde,
        'Hist_MAE': mae_hist,
        'HDE_MAE': mae_hde,
        'Hist_Normalized_MSE': mse_hist / sample_size,
        'HDE_Normalized_MSE': mse_hde / sample_size,
        'Hist_Normalized_MAE': mae_hist / sample_size,
        'HDE_Normalized_MAE': mae_hde / sample_size,
        'Hist_IMSE': mise_hist,
        'HDE_IMSE': mise_hde,
        'Hist_MIAE': miae_hist,
        'HDE_MIAE': miae_hde,
        'Hist_AMISE': amise_hist,
        'HDE_AMISE': amise_hde
    }

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)
    margin = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin

def accumulate_errors(all_errors, error_key, new_error):
    if error_key not in all_errors:
        all_errors[error_key] = []
    # Handle both single values and iterables
    if isinstance(new_error, (list, np.ndarray)):
        clean_error = [e if np.isfinite(e) else np.nan for e in new_error]
    else:  # Single value
        clean_error = [new_error] if np.isfinite(new_error) else [np.nan]
    all_errors[error_key].extend(clean_error)



def add_confidence_intervals(result_entry, all_errors):
    for error_key, errors in all_errors.items():
        mean, ci_lower, ci_upper = compute_confidence_interval(errors)
        result_entry[f'CI_{error_key}_Mean'] = mean
        result_entry[f'CI_{error_key}_Lower'] = ci_lower
        result_entry[f'CI_{error_key}_Upper'] = ci_upper

def calculate_percentage_difference(all_errors):
    percent_diff = {}
    for key in all_errors:
        if key.startswith('Hist_') and key.replace('Hist_', 'HDE_') in all_errors:
            hde_key = key.replace('Hist_', 'HDE_')
            hist_errors = np.array(all_errors[key])
            hde_errors = np.array(all_errors[hde_key])
            # Handle division by zero or NaN values
            valid_indices = np.where(hist_errors != 0)
            hist_errors_nonzero = hist_errors[valid_indices]
            hde_errors_nonzero = hde_errors[valid_indices]
            percent_diff[key.replace('Hist_', 'Percent_Diff_')] = 100 * (hist_errors_nonzero - hde_errors_nonzero) / hist_errors_nonzero
    return percent_diff

def main():
    results = []

    for dist_name in distributions:
        full_sample = generate_distribution(dist_name, MAX_SAMPLE_SIZE)

        for sample_size in sample_sizes[:-3]:
            for num_bins in bin_counts:
                true_density = compute_true_density(full_sample, num_bins)

                all_errors = {}
                for iteration in range(num_iterations):
                    subsample = np.random.choice(full_sample, sample_size, replace=False)

                    hist_density = compute_true_density(subsample, num_bins)
                    hde = HDEMethod(observations=subsample, bin_width=(np.max(subsample) - np.min(subsample)) / num_bins, anchor=np.min(subsample))
                    hde_density = [hde.final_density_estimate(x) for x in np.linspace(np.min(subsample), np.max(subsample), num_bins)]

                    errors = compute_errors_and_plot(true_density, hist_density, hde_density, dist_name, sample_size, num_bins, iteration, path)
                    for key, value in errors.items():
                        accumulate_errors(all_errors, key, value)

                # Inside main function, after accumulating errors:
                percent_diffs = calculate_percentage_difference(all_errors)
                for key, value in percent_diffs.items():
                    accumulate_errors(all_errors, key, value)  # Accumulate percentage differences

                # Add confidence intervals and percentage differences to result_entry
                result_entry = {'Distribution': dist_name, 'Sample Size': sample_size, 'Bin Count': num_bins}
                add_confidence_intervals(result_entry, all_errors)
                for key in percent_diffs.keys():
                    result_entry[key] = np.nanmean(all_errors[key])  # Store the mean of the percent differences
                results.append(result_entry)


    df_results = pd.DataFrame(results)
    df_results.to_csv('experiment_results.csv', index=False)
    print("Experiment 4 completed and results saved.")

if __name__ == "__main__":
    main()

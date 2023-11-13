import numpy as np
import pandas as pd
from hdem import HDEMethod
from scipy.stats import alpha, beta, gamma, weibull_min, norm, expon, chi2, lognorm, t, f, poisson, binom
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define utility functions
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

def compute_errors(true_counts, estimated_densities, bin_width, total_samples, data, dist_name, num_bins):
    # Convert estimated densities to a NumPy array if it's not already
    estimated_densities = np.array(estimated_densities)
    estimated_densities = np.nan_to_num(estimated_densities)

    # Adjust estimated densities to match the scale of true counts
    estimated_counts = estimated_densities * bin_width * total_samples

    # CAUTION: Clip true_counts to avoid division by very small numbers (may help in some cases)
    # clipped_true_counts = np.clip(true_counts, 1e-6, None)

    # Generate x_values for true density
    x_values = np.linspace(np.min(data), np.max(data), num_bins)

    # Compute true density for IMSE calculation
    true_density = generate_distribution(dist_name, len(x_values))

    # Ensure true_density and estimated_densities are of the same length
    true_density = np.interp(x_values, np.linspace(np.min(data), np.max(data), len(true_density)), true_density)

    imse = np.mean((true_density - estimated_densities)**2)

    return {
        'MSE': mean_squared_error(true_counts, estimated_counts),
        'MAE': mean_absolute_error(true_counts, estimated_counts),
        'IMSE': imse
    }

# Experiment parameters
sample_sizes = [10, 100, 1000, 10000, 100000]
bin_counts = [10, 15, 20, 25, 30, 50]
distributions = ['alpha', 'beta', 'gamma', 'weibull', 'norm', 'expon', 'chi2', 'lognorm', 'poisson', 'binom', 'combined']

# Conduct the experiment
results = []

for dist_name in distributions:
    for sample_size in sample_sizes:
        data = generate_distribution(dist_name, sample_size)
        for num_bins in bin_counts:
            # Compute bin width from the range of data and the number of bins
            bin_width = (np.max(data) - np.min(data)) / num_bins

            # Initialize HDEMethod and estimate densities
            hde = HDEMethod(observations=data, bin_width=bin_width, anchor=np.min(data))
            estimated_densities = [hde.final_density_estimate(x) for x in np.linspace(np.min(data), np.max(data), num_bins)]

            # Compute true counts for error calculation
            true_counts, _ = np.histogram(data, bins=num_bins, density=False)

            # Compute errors, now passing 'data' and 'dist_name' as additional arguments
            errors = compute_errors(true_counts, estimated_densities, bin_width, sample_size, data, dist_name, num_bins)

            # Store results
            results.append({
                'Distribution': dist_name,
                'Sample Size': sample_size,
                'Bin Count': num_bins,
                'Errors': errors
            })

# Convert results to DataFrame and save as CSV
df_results = pd.DataFrame(results)
df_results.to_csv('experiment_results.csv', index=False)

print("Experiment completed and results saved.")

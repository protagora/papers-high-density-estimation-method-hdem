import numpy as np
import pandas as pd
from hdem import HDEMethod
import shutil
from scipy.stats import alpha, beta, gamma, weibull_min, norm, expon, chi2, lognorm, t, f, poisson, binom
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import entropy

# Create (or empty) the 'estimates' directory
estimates_dir = 'estimates'
if os.path.exists(estimates_dir):
    shutil.rmtree(estimates_dir)
os.makedirs(estimates_dir)

def save_density_plots(data, estimated_densities, dist_name, sample_size, num_bins, iteration):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=num_bins, density=True, alpha=0.6, label='Sample Data')
    x_values = np.linspace(np.min(data), np.max(data), num_bins)
    plt.plot(x_values, estimated_densities, label='HDEM Estimate', color='red')
    plt.title(f'Histogram and HDEM Estimate for {dist_name} (n={sample_size}, bins={num_bins})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    filename = f'estimates/plot_{dist_name}_n{sample_size}_bins{num_bins}_iter{iteration}.png'
    plt.savefig(filename)
    plt.close()

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

# def compute_errors(true_counts, estimated_densities, bin_width, total_samples, data, dist_name, num_bins):
#     estimated_densities = np.array(estimated_densities)
#     estimated_densities = np.nan_to_num(estimated_densities)

#     estimated_counts = estimated_densities * bin_width * total_samples

#     x_values = np.linspace(np.min(data), np.max(data), num_bins)
#     true_density = generate_distribution(dist_name, len(x_values))
#     true_density = np.interp(x_values, np.linspace(np.min(data), np.max(data), len(true_density)), true_density)

#     mise = np.mean((true_density - estimated_densities)**2) * bin_width
#     miae = np.mean(np.abs(true_density - estimated_densities)) * bin_width
#     amise = mise / np.sqrt(total_samples)  # Approximate AMISE

#     return {
#         'MSE': mean_squared_error(true_counts, estimated_counts),
#         'MAE': mean_absolute_error(true_counts, estimated_counts),
#         'Normalized MSE': mean_squared_error(true_counts, estimated_counts) / total_samples,
#         'Normalized MAE': mean_absolute_error(true_counts, estimated_counts) / total_samples,
#         'IMSE': np.mean((true_density - estimated_densities)**2),
#         'MISE': mise,
#         'MIAE': miae,
#         'AMISE': amise
#     }

def compute_errors(true_counts, estimated_densities, bin_width, total_samples, data, dist_name, num_bins):
    estimated_densities = np.array(estimated_densities)
    estimated_densities = np.nan_to_num(estimated_densities)

    estimated_counts = estimated_densities * bin_width * total_samples

    # Calculate the true and estimated probability distributions
    true_prob_dist = true_counts / total_samples
    estimated_prob_dist = estimated_counts / total_samples

    # Compute KL Divergence
    kl_divergence = entropy(true_prob_dist, estimated_prob_dist)

    # Compute Relative Errors
    relative_mse = mean_squared_error(true_counts, estimated_counts) / np.mean(true_counts)
    relative_mae = mean_absolute_error(true_counts, estimated_counts) / np.mean(true_counts)

    mise = np.mean((true_prob_dist - estimated_prob_dist) ** 2) * bin_width
    miae = np.mean(np.abs(true_prob_dist - estimated_prob_dist)) * bin_width
    amise = mise / np.sqrt(total_samples)  # Approximate AMISE

    return {
        'MSE': mean_squared_error(true_counts, estimated_counts),
        'MAE': mean_absolute_error(true_counts, estimated_counts),
        'Relative MSE': relative_mse,
        'Relative MAE': relative_mae,
        'Normalized MSE': mean_squared_error(true_counts, estimated_counts) / total_samples,
        'Normalized MAE': mean_absolute_error(true_counts, estimated_counts) / total_samples,
        'IMSE': np.mean((true_prob_dist - estimated_prob_dist) ** 2),
        'MISE': mise,
        'MIAE': miae,
        'AMISE': amise,
        'KL Divergence': kl_divergence
    }

# Experiment parameters
sample_sizes = [10, 100, 1000, 10000, 100000]
bin_counts = [10, 15, 20, 25, 30, 50, 100]
distributions = ['alpha', 'beta', 'gamma', 'weibull', 'norm', 'expon', 'chi2', 'lognorm', 'poisson', 'binom', 'combined']

# Conduct the experiment
results = []

iteration = 0
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

            # Compute errors
            errors = compute_errors(true_counts, estimated_densities, bin_width, sample_size, data, dist_name, num_bins)

            # Merge the 'errors' dictionary with other information and store results
            result_entry = {
                'Distribution': dist_name,
                'Sample Size': sample_size,
                'Bin Count': num_bins
            }
            result_entry.update(errors)  # Merge the errors dictionary

            # Append the updated dictionary to the results list
            results.append(result_entry)

            # Save the plot for this iteration
            save_density_plots(data, estimated_densities, dist_name, sample_size, num_bins, iteration)
            iteration += 1


# Convert results to DataFrame and save as CSV
df_results = pd.DataFrame(results)
df_results.to_csv('experiment_results.csv', index=False)

print("Experiment completed and results saved.")

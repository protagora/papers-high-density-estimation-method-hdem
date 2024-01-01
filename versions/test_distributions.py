import numpy as np
import matplotlib.pyplot as plt
from phd.HDEM.versions.experiment2 import generate_distribution
import os

def plot_distribution(dist_name, sample_size, destination='tests'):
    if not os.path.exists(destination) or not os.path.isdir(destination):
        os.makedirs(destination)
    
    data = generate_distribution(dist_name, sample_size)
    path = os.path.join(destination, f"{dist_name}_distribution.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram
    ax[0].hist(data, bins=30, alpha=0.7, color='blue')
    ax[0].set_title(f'Histogram of {dist_name}')
    ax[0].set_xlabel('Values')
    ax[0].set_ylabel('Frequency')

    # Scatter Plot
    ax[1].scatter(range(len(data)), data, alpha=0.7, color='red')
    ax[1].set_title(f'Scatter Plot of {dist_name}')
    ax[1].set_xlabel('Sample Number')
    ax[1].set_ylabel('Values')

    plt.tight_layout()
    plt.savefig(f'{path}')
    plt.close()

distributions = ['beta', 'gamma', 'weibull', 'expon', 'chi2', 'lognorm', 'poisson', 'binom', 'combined']

for dist in distributions:
    print(dist)
    destination = os.path.join('tests', 'distributions')
    plot_distribution(dist, 10000, destination=destination)

print("Plots generated for all distributions.")

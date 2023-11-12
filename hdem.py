# Let's define the HDEMethod class according to the description provided.
import numpy as np
from scipy.stats import rv_continuous
from typing import List, Tuple, Callable

class HDEMethod:
    """
    Class implementing the Histogram Density Estimation Method (HDEM) as described.
    """
    
    def __init__(self, observations: List[float], bin_width: float, anchor: float):
        """
        Initializes the HDEMethod with a set of observations, bin width, and anchor position.
        
        :param observations: A list of observations from the random variable X.
        :param bin_width: The width of each bin in the histogram.
        :param anchor: The anchor position a_0 that determines the start of the first bin.
        """
        self.observations = np.array(observations)
        self.bin_width = bin_width
        self.anchor = anchor
        self.bins = self.create_bins()
        self.histogram = self.calculate_histogram()

    @staticmethod
    def normalize_histogram(histogram: np.ndarray) -> np.ndarray:
        """
        Normalizes the histogram by dividing each bin count by the total number of observations.
        
        :param histogram: An array containing bin counts.
        :return: Normalized histogram.
        """
        total_observations = np.sum(histogram)
        return histogram / total_observations if total_observations > 0 else histogram

    def create_bins(self) -> np.ndarray:
        """
        Creates bins for the histogram based on the observations, bin width, and anchor.
        
        :return: An array representing the bin edges.
        """
        min_edge = self.anchor
        max_edge = self.anchor + np.ceil((self.observations.max() - self.anchor) / self.bin_width) * self.bin_width
        return np.arange(min_edge, max_edge, self.bin_width)

    def calculate_histogram(self) -> np.ndarray:
        """
        Calculates the histogram for the observations.
        
        :return: An array representing the bin counts.
        """
        counts, _ = np.histogram(self.observations, bins=self.bins)
        return self.normalize_histogram(counts)

    def estimate_density(self, x: float, omega: int) -> float:
        """
        Estimates the density at a point x using a weighted average of histogram densities
        over an interval of bin widths determined by omega.
        
        :param x: The point at which to estimate the density.
        :param omega: The number of bin intervals to consider in the estimation.
        :return: The estimated density at x.
        """
        # Find the indices of the bins that x falls into
        k = np.digitize(x, self.bins) - 1
        r = max(k - omega + 1, 0)
        s = min(k + omega, len(self.histogram))

        # Calculate the average density values for the range of intervals
        densities = [self.histogram[i:i + omega].mean() for i in range(r, s - omega + 1)]
        expected_density = np.mean(densities)

        return expected_density

    def find_optimal_omega(self, x: float, weighting_function: Callable[[int], float]) -> int:
        """
        Finds the optimal omega that maximizes the weighted expected average density.
        
        :param x: The point at which to find the optimal omega.
        :param weighting_function: A function that assigns weights to different omegas.
        :return: The optimal omega value.
        """
        omegas = range(1, len(self.histogram) // 3 + 1)
        densities = [self.estimate_density(x, omega) for omega in omegas]
        weighted_densities = [weighting_function(omega) * density for omega, density in zip(omegas, densities)]
        optimal_omega = omegas[np.argmax(weighted_densities)]

        return optimal_omega

    def final_density_estimate(self, x: float) -> float:
        """
        Calculates the final density estimate for a given point x.
        
        :param x: The point at which to calculate the density estimate.
        :return: The final density estimate at x.
        """
        # Define the weighting function W(omega)
        N = len(self.histogram)
        weighting_function = lambda omega: 1 - (omega - 1) / (N / 3)

        # Find the optimal omega
        omega = self.find_optimal_omega(x, weighting_function)

        # Calculate the normalizing constant C
        C = 1 / np.sum([self.estimate_density(x, omega) for x in self.bins[:-1]])

        # Calculate the final density estimate
        final_density = C * weighting_function(omega) * self.estimate_density(x, omega)

        return final_density

# Generate a sample distribution (e.g., a beta distribution)
np.random.seed(42)  # for reproducibility
sample_data = np.random.beta(a=2, b=5, size=1000)

# Define bin width and anchor position
bin_width = 0.05
anchor = 0

# Initialize HDEMethod
hde_method = HDEMethod(observations=sample_data, bin_width=bin_width, anchor=anchor)

# Estimate density for a specific value
x_value = 0.5  # Example point
density_estimate = hde_method.final_density_estimate(x_value)

print(f"Density Estimate at x = {x_value}: {density_estimate}")

# PLOT

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# Generate a range of x-values for density estimation
x_values = np.linspace(sample_data.min(), sample_data.max(), 500)
density_estimates = [hde_method.final_density_estimate(x) for x in x_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.hist(sample_data, bins=40, density=True, alpha=0.6, label='Sample Distribution')
plt.plot(x_values, density_estimates, label='Density Estimates', color='red')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Sample Distribution vs. Density Estimates')
plt.legend()
plt.show()

# Compute MAPE and MSE
# For this, we need to compare the histogram counts with the estimated densities
counts, bin_edges = np.histogram(sample_data, bins=40, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
estimated_densities = [hde_method.final_density_estimate(x) for x in bin_centers]

# Compute errors
mse = mean_squared_error(counts, estimated_densities)
mape = mean_absolute_percentage_error(counts, estimated_densities)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('experiment_results.csv')

# Function to calculate confidence interval from sample data
def compute_confidence_interval_from_sample(data, confidence=0.95):
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard Error of the Mean
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean, mean - interval, mean + interval

# Grouping the data and calculating statistics
grouped = df.groupby(['Distribution', 'Sample Size', 'Bin Count'])

# Summary table
summary_table = []

# Define all metrics (assuming these are the columns in your data)
metrics = df.columns.difference(['Distribution', 'Sample Size', 'Bin Count'])

for name, group in grouped:
    distribution, sample_size, bin_count = name
    summary_row = [distribution, sample_size, bin_count]
    
    for metric in metrics:
        if metric in group.columns:
            mean, ci_low, ci_high = compute_confidence_interval_from_sample(group[metric])
            summary_row.extend([mean, ci_low, ci_high])
        else:
            summary_row.extend([np.nan, np.nan, np.nan])  # Append NaN if column doesn't exist

    summary_table.append(summary_row)

# Define column names
column_names = ['Distribution', 'Sample Size', 'Bin Count']
for metric in metrics:
    column_names.extend([f'{metric} Mean', f'{metric} CI Lower', f'{metric} CI Upper'])

summary_df = pd.DataFrame(summary_table, columns=column_names)
summary_df.to_csv('summary.csv', index=False)

# Print column names to identify the correct metric
print(summary_df.columns)

# Choose an appropriate metric from your summary_df
# Replace 'Metric1 Mean' with the actual metric name after identifying it from the printed column names
metric_for_heatmap = 'Metric1 Mean'  # Replace with actual metric name

# Pivot the DataFrame for the heatmap
heatmap_data = summary_df.pivot_table(index='Distribution', columns=['Sample Size', 'Bin Count'], values=metric_for_heatmap)

# Plotting the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap='viridis')
plt.title(f'{metric_for_heatmap} Across Distributions, Sample Sizes, and Bin Counts')
plt.ylabel('Distribution')
plt.xlabel('(Sample Size, Bin Count)')
plt.show()

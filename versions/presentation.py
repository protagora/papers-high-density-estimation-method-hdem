import pandas as pd
import numpy as np
from scipy import stats

# Function to read your data
def read_data(file_path):
    return pd.read_csv(file_path)

# Function to calculate confidence interval from sample data
def compute_confidence_interval_from_sample(data, confidence=0.95):
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard Error of the Mean
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean, mean - interval, mean + interval

# Function to process the data and create summary tables
def process_data(df):
    grouped = df.groupby(['Distribution', 'Sample Size', 'Bin Count'])
    metrics = df.columns.difference(['Distribution', 'Sample Size', 'Bin Count'])
    summary_tables = {metric: [] for metric in metrics}
    
    for name, group in grouped:
        distribution, sample_size, bin_count = name
        for metric in metrics:
            if metric in group.columns:
                mean, ci_low, ci_high = compute_confidence_interval_from_sample(group[metric])
                summary_tables[metric].append([distribution, sample_size, bin_count, mean, ci_low, ci_high])

    return summary_tables

# Function to print tables in a format suitable for publication
def print_tables_for_publication(summary_tables):
    for metric, data in summary_tables.items():
        print(f"\n#### Table: Summary Statistics for {metric}\n")
        print("| Distribution | Sample Size | Bin Count | Mean | CI Lower | CI Upper |")
        print("|--------------|-------------|-----------|------|----------|----------|")
        for row in data:
            print("| " + " | ".join(map(str, row)) + " |")

# Main execution
if __name__ == "__main__":
    file_path = 'experiment_results.csv'  # Change to your file path
    df = read_data(file_path)
    summary_tables = process_data(df)
    print_tables_for_publication(summary_tables)
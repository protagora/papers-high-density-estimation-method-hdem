import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_errors_by_sample_size(df, error_type, path='tests'):
    plt.figure(figsize=(12, 8))
    # Plotting for HDEM errors
    sns.lineplot(data=df, x='Bin Count', y=f'HDEM_{error_type}', hue='Sample Size', style='Sample Size', markers=True, palette="deep")
    # Plotting for Histogram errors
    sns.lineplot(data=df, x='Bin Count', y=f'Hist_{error_type}', hue='Sample Size', style='Sample Size', markers=True, palette="pastel")
    plt.title(f'{error_type} vs Bin Count for Different Sample Sizes')
    plt.xlabel('Bin Count')
    plt.ylabel(error_type)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(path, f"{error_type}_vs_Sample_size.png")
    plt.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()
    plt.close()

def plot_errors(df, error_type, path='tests'):
    plt.figure(figsize=(12, 8))
    # Plotting for HDEM errors
    sns.lineplot(data=df, x='Sample Size', y=f'HDEM_{error_type}', hue='Distribution', style='Bin Count', markers=True, palette="deep")
    # Plotting for Histogram errors
    sns.lineplot(data=df, x='Sample Size', y=f'Hist_{error_type}', hue='Distribution', style='Bin Count', markers=True, palette="pastel")
    plt.title(f'{error_type} vs Sample Size for Different Distributions and Bin Counts')
    plt.xscale('log') 
    plt.yscale('log' if df[f'HDEM_{error_type}'].min() > 0 and df[f'Hist_{error_type}'].min() > 0 else 'linear')
    plt.xlabel('Sample Size')
    plt.ylabel(error_type)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(path, f"{error_type}_vs_Bin_Count.png")
    plt.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()
    plt.close()

# Load the results
df = pd.read_csv('experiment_results.csv')
path = os.path.join('tests', 'errors')
errors = ['MSE', 'MAE', 'Normalized MSE', 'Normalized MAE', 'IMSE', 'MISE', 'MIAE', 'AMISE']

if not os.path.exists(path) or not os.path.isdir(path):
    os.makedirs(path)

# Plot each error type by sample size first
for error in errors:
    plot_errors(df, error, path=path)

# Then plot each error type by bin count
for error in errors:
    plot_errors_by_sample_size(df, error, path=path)

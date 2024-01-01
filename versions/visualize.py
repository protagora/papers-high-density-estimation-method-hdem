import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_errors_by_sample_size(df, error_type):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Bin Count', y=error_type, hue='Sample Size', style='Sample Size', markers=True)
    plt.title(f'Total {error_type} vs Bin Count for Different Sample Sizes')
    plt.xlabel('Bin Count')
    plt.ylabel(f'Total {error_type}')
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Total_{error_type}_vs_Bin_Count.png", bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()
    plt.close()

def plot_errors(df, error_type):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Sample Size', y=error_type, hue='Distribution', style='Bin Count', markers=True)
    plt.title(f'{error_type} vs Sample Size for Different Distributions and Bin Counts')
    plt.xscale('log') 
    plt.yscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel(error_type)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{error_type}_plot.png", bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()
    plt.close()

# Load the results
df = pd.read_csv('experiment_results.csv')

# Plot each error type by sample size first
for error in ['MSE', 'MAE', 'Normalized MSE', 'Normalized MAE', 'IMSE', 'MISE', 'MIAE', 'AMISE']:
    plot_errors(df, error)

# Then plot each error type by bin count
for error in ['MSE', 'MAE', 'Normalized MSE', 'Normalized MAE', 'IMSE', 'MISE', 'MIAE', 'AMISE']:
    plot_errors_by_sample_size(df, error)

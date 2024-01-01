import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_errors_by_sample_size(df, error_type, path='tests'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    if f'CI_HDE_{error_type}_Mean' in df.columns and f'CI_Hist_{error_type}_Mean' in df.columns:
        # HDEM errors
        sns.lineplot(ax=axes[0], data=df, x='Bin Count', y=f'CI_HDE_{error_type}_Mean', hue='Sample Size', style='Sample Size', markers=True, palette="deep")
        axes[0].set_title(f'HDEM {error_type}')
        axes[0].set_xlabel('Bin Count')
        axes[0].set_ylabel(f'{error_type}')
        axes[0].grid(True)

        # Histogram errors
        sns.lineplot(ax=axes[1], data=df, x='Bin Count', y=f'CI_Hist_{error_type}_Mean', hue='Sample Size', style='Sample Size', markers=True, palette="pastel")
        axes[1].set_title(f'Histogram {error_type}')
        axes[1].set_xlabel('Bin Count')
        axes[1].set_ylabel(f'{error_type}')
        axes[1].grid(True)

        # Difference
        df_diff = df.copy()
        df_diff[f'Diff_{error_type}'] = df_diff[f'CI_HDE_{error_type}_Mean'] - df_diff[f'CI_Hist_{error_type}_Mean']
        sns.lineplot(ax=axes[2], data=df_diff, x='Bin Count', y=f'Diff_{error_type}', hue='Sample Size', style='Sample Size', markers=True, palette="coolwarm")
        axes[2].set_title(f'Difference {error_type}')
        axes[2].set_xlabel('Bin Count')
        axes[2].set_ylabel(f'Difference {error_type}')
        axes[2].grid(True)

        # Move legend to the side
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))

        plt.tight_layout()
        filename = os.path.join(path, f"CI_{error_type}_vs_Sample_size_comparison.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        print(f"Column(s) for {error_type} not found in DataFrame.")

def plot_errors(df, error_type, path='tests'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    if f'CI_HDE_{error_type}_Mean' in df.columns and f'CI_Hist_{error_type}_Mean' in df.columns:
        # HDEM errors
        sns.lineplot(ax=axes[0], data=df, x='Sample Size', y=f'CI_HDE_{error_type}_Mean', hue='Distribution', style='Bin Count', markers=True, palette="deep")
        axes[0].set_title(f'HDEM {error_type}')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Sample Size')
        axes[0].set_ylabel(f'{error_type}')
        axes[0].grid(True)

        # Histogram errors
        sns.lineplot(ax=axes[1], data=df, x='Sample Size', y=f'CI_Hist_{error_type}_Mean', hue='Distribution', style='Bin Count', markers=True, palette="pastel")
        axes[1].set_title(f'Histogram {error_type}')
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Sample Size')
        axes[1].set_ylabel(f'{error_type}')
        axes[1].grid(True)

        # Difference
        df_diff = df.copy()
        df_diff[f'Diff_{error_type}'] = df_diff[f'CI_HDE_{error_type}_Mean'] - df_diff[f'CI_Hist_{error_type}_Mean']
        sns.lineplot(ax=axes[2], data=df_diff, x='Sample Size', y=f'Diff_{error_type}', hue='Distribution', style='Bin Count', markers=True, palette="coolwarm")
        axes[2].set_title(f'Difference {error_type}')
        axes[2].set_xscale('log')
        axes[2].set_xlabel('Sample Size')
        axes[2].set_ylabel(f'Difference {error_type}')
        axes[2].grid(True)

        # Move legend to the side
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))

        plt.tight_layout()
        filename = os.path.join(path, f"{error_type}_vs_Sample_Size_comparison.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        print(f"Column(s) for {error_type} not found in DataFrame.")

# Load the results
df = pd.read_csv('experiment4_results.csv')
path = os.path.join('tests', 'errors')
errors = ['MSE', 'MAE', 'Normalized_MSE', 'Normalized_MAE', 'IMSE', 'MIAE', 'AMISE']

if not os.path.exists(path) or not os.path.isdir(path):
    os.makedirs(path)

# Plot each error type by sample size first
for error in errors:
    plot_errors(df, error, path=path)

# Then plot each error type by bin count
for error in errors:
    plot_errors_by_sample_size(df, error, path=path)

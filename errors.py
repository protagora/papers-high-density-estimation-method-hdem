import pandas as pd

def calculate_average_percent_diff(df, distribution):
    # Filter the DataFrame for the given distribution
    dist_df = df[df['Distribution'] == distribution]

    # List of error types
    error_types = ['MSE', 'MAE', 'Normalized_MSE', 'Normalized_MAE', 'IMSE', 'MIAE', 'AMISE']

    # Calculate average percent difference for each error type
    avg_percent_diff = {}
    for error in error_types:
        col_name = f'Percent_Diff_{error}'
        if col_name in dist_df.columns:
            avg_percent_diff[error] = dist_df[col_name].mean()

    return avg_percent_diff

def main():
    # Load the data
    df = pd.read_csv('experiment_results.csv')

    # Unique distributions
    distributions = df['Distribution'].unique()

    # Calculate and display average percent differences for each distribution
    for dist in distributions:
        avg_diff = calculate_average_percent_diff(df, dist)
        print(f"Average Percent Difference for {dist}:")
        for error, value in avg_diff.items():
            print(f"  {error}: {value:.2f}%")
        print()

    # Calculate overall average percent difference
    overall_avg_diff = {}
    error_types = ['MSE', 'MAE', 'Normalized_MSE', 'Normalized_MAE', 'IMSE', 'MIAE', 'AMISE']
    for error in error_types:
        col_name = f'Percent_Diff_{error}'
        if col_name in df.columns:
            overall_avg_diff[error] = df[col_name].mean()

    print("Overall Average Percent Difference:")
    for error, value in overall_avg_diff.items():
        print(f"  {error}: {value:.2f}%")

if __name__ == "__main__":
    main()

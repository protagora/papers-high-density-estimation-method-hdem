import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
df = pd.read_csv('experiment_results.csv')

# Convert the 'Errors' column from string to dictionary
import ast
df['Errors'] = df['Errors'].apply(ast.literal_eval)

# Unpack the errors into separate columns
df_errors = pd.json_normalize(df['Errors'])
df = pd.concat([df.drop(columns=['Errors']), df_errors], axis=1)

# Plotting function
def plot_errors(df, error_type):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Sample Size', y=error_type, hue='Distribution', style='Bin Count', markers=True)
    plt.title(f'{error_type} vs Sample Size for Different Distributions and Bin Counts')
    plt.xscale('log') 
    plt.yscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel(error_type)
    # Place the legend outside of the figure/plot
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(True)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{error_type}_plot.png", bbox_extra_artists=(legend,), bbox_inches='tight')
    
    # Show the plot
    plt.show()
    plt.close()

# Plot each error type
for error in ['MSE', 'MAE', 'IMSE']:
    plot_errors(df, error)

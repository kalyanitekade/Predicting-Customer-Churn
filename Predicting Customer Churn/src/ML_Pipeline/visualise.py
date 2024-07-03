import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 

output = "./output"

def plot_scatter(df, target_col, col_to_exclude):
    cols = df.select_dtypes(include=np.number).columns.to_list()  # Finding all the numerical columns from the dataframe
    X = df[cols]  # Creating a dataframe only with the numerical columns
    X = X[X.columns.difference(col_to_exclude)]  # Columns to exclude
    target_col_list = [target_col]  # Ensure target_col is a list
    for col in X.columns.difference(target_col_list):
        g = sns.FacetGrid(df)
        g.map(sns.scatterplot, col, target_col)
        g.add_legend()
        plt.title(f'Scatter plot of {col} vs {target_col}')
        plt.xlabel(col)
        plt.ylabel(target_col)
        # Save the plot
        file_path = os.path.join(output, f'{col}_vs_{target_col}.png')
        plt.savefig(file_path)
        # Show the plot
        plt.show()
        
        
def full_diagnostic(df, target_col):
    sns.pairplot(df, hue = target_col)
    plt.suptitle("Full Diagnostic plot")
    file_path = os.path.join(output, 'full_diagnostic.png')
    plt.savefig(file_path)
    plt.show()
    
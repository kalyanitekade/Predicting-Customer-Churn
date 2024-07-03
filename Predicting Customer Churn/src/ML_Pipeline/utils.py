import pandas as pd 
import numpy as np 

#Function to read the data
def read_data(path, **kwargs):
    raw_data = pd.read_csv(path)
    return raw_data

#Function to remove cols 
def inspection(dataframe):
    print("Checking the datatype of the columns:")
    print(dataframe.dtypes)
    
    print("Total missing values per column:")
    print(dataframe.isnull().sum())
    
#Function to remove nulls 
def null_values(dataframe):
    dataframe = dataframe.dropna()
    return dataframe

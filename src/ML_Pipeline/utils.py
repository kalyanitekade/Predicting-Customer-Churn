import pandas as pd 
import numpy as np 


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

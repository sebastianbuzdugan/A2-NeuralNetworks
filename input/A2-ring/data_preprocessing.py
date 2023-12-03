# read th information of a CSV and load into a dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the csv file
name='A2-ring-merged'

df= pd.read_csv(name+'.txt',sep='\t')
print(df.head())

print(df.describe())

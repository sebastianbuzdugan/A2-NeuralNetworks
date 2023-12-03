# read th information of a CSV and load into a dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the csv file
name='bank-additional'

df= pd.read_csv(name+'.csv',sep=';')
print(df.head())

print(df.describe())

df_processed = pd.DataFrame()

# Replace 'unknown' with the mode in each column
for column in df.columns:
    mode_value = df[column].mode()[0]
    df_processed[column] = df[column].replace('unknown', mode_value)
    is_yes_no = df_processed[column].isin(['yes', 'no']).all()
    if is_yes_no:
        df_processed[column] = df_processed[column].replace({'yes': 1, 'no': 0}).astype(int)
    else:
        is_numeric = pd.to_numeric(df_processed[column], errors='coerce').notnull().all()
        if not is_numeric:
            unique=df_processed[column].unique()
            unique={k:v for v,k in enumerate(unique)}
            print(unique)
            df_processed[column]=df_processed[column].map(unique)   
        print(column, is_numeric,is_yes_no)

df_processed=df_processed.iloc[1:]

df_processed=df_processed.iloc[:, [0,10,11,12,13,15,16,17,18, 19, -1]]

# Data normalization
# Min-Max Scaling
df_processed.iloc[:, 0]= (df_processed.iloc[:,0] - df_processed.iloc[:,0].min()) / (df_processed.iloc[:,0].max() - df_processed.iloc[:,0].min())
df_processed.iloc[:, 1]= (df_processed.iloc[:,1] - df_processed.iloc[:,1].min()) / (df_processed.iloc[:,1].max() - df_processed.iloc[:,1].min())
df_processed.iloc[:, 2]= (df_processed.iloc[:,2] - df_processed.iloc[:,2].min()) / (df_processed.iloc[:,2].max() - df_processed.iloc[:,2].min())
df_processed.iloc[:, 3]= (df_processed.iloc[:,3] - df_processed.iloc[:,3].min()) / (df_processed.iloc[:,3].max() - df_processed.iloc[:,3].min())
df_processed.iloc[:, 4]= (df_processed.iloc[:,4] - df_processed.iloc[:,4].min()) / (df_processed.iloc[:,4].max() - df_processed.iloc[:,4].min())
df_processed.iloc[:, 5]= (df_processed.iloc[:,5] - df_processed.iloc[:,5].min()) / (df_processed.iloc[:,5].max() - df_processed.iloc[:,5].min())
df_processed.iloc[:, 6]= (df_processed.iloc[:,6] - df_processed.iloc[:,6].min()) / (df_processed.iloc[:,6].max() - df_processed.iloc[:,6].min())
df_processed.iloc[:, 7]= (df_processed.iloc[:,7] - df_processed.iloc[:,7].min()) / (df_processed.iloc[:,7].max() - df_processed.iloc[:,7].min())
df_processed.iloc[:, 8]= (df_processed.iloc[:,8] - df_processed.iloc[:,8].min()) / (df_processed.iloc[:,8].max() - df_processed.iloc[:,8].min())
df_processed.iloc[:, 9]= (df_processed.iloc[:,9] - df_processed.iloc[:,9].min()) / (df_processed.iloc[:,9].max() - df_processed.iloc[:,9].min())

print(df_processed.describe())

print(df_processed.head())

output_file_name=name+'-processed.csv'
df_processed.to_csv(output_file_name,sep='\t', index=False,header=None)
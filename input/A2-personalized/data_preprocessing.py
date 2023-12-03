# read th information of a CSV and load into a dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the csv file
name='data_banknote_authentication'

df= pd.read_csv(name+'.txt',sep=',')
print(df.head())

print(df.describe())

df_processed = pd.DataFrame()

# Replace 'unknown' with the mode in each column
for column in df.columns:
    mode_value = df[column].mode()[0]
    df_processed[column] = df[column].replace('unknown', mode_value)
    is_yes_no = df_processed[column].isin(['Kecimen', 'Besni']).all()
    if is_yes_no:
        df_processed[column] = df_processed[column].replace({'Kecimen': 1, 'Besni': 0}).astype(int)
    else:
        is_numeric = pd.to_numeric(df_processed[column], errors='coerce').notnull().all()
        if not is_numeric:
            unique=df_processed[column].unique()
            unique={k:v for v,k in enumerate(unique)}
            print(unique)
            df_processed[column]=df_processed[column].map(unique)

        print(column, is_numeric,is_yes_no)


# Data normalization
# Min-Max Scaling
df_processed.iloc[:, 0]= (df_processed.iloc[:,0] - df_processed.iloc[:,0].min()) / (df_processed.iloc[:,0].max() - df_processed.iloc[:,0].min())
df_processed.iloc[:, 1]= (df_processed.iloc[:,1] - df_processed.iloc[:,1].min()) / (df_processed.iloc[:,1].max() - df_processed.iloc[:,1].min())
# Standardization
df_processed.iloc[:, 2]= (df_processed.iloc[:,2] - df_processed.iloc[:,2].mean()) / df_processed.iloc[:,2].std()
df_processed.iloc[:, 3]= (df_processed.iloc[:,2] - df_processed.iloc[:,3].mean()) / df_processed.iloc[:,3].std()

df_processed = df_processed.sample(frac=1, random_state=42)

print(df_processed.head())

output_file_name=name+'-processed.txt'
df_processed.to_csv(output_file_name,sep='\t', index=False,header=None)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path, sep=";")

def preprocess_data(df):
    # replace 'unknown' with the most frequent value
    for column in df.columns:
        most_frequent = df[column].mode()[0]
        df[column] = df[column].replace('unknown', most_frequent)
    
    # convert categorical features into numerical form using label encoding
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    return df

# to split the dataset into training and testing sets and save them
def split_and_save(df, target_column, test_size=0.2, file_path_prefix=""):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    X_train.to_csv(f"input/A2-bank/{file_path_prefix}X_train.csv", index=False)
    X_test.to_csv(f"input/A2-bank/{file_path_prefix}X_test.csv", index=False)
    y_train.to_csv(f"input/A2-bank/{file_path_prefix}y_train.csv", index=False)
    y_test.to_csv(f"input/A2-bank/{file_path_prefix}y_test.csv", index=False)

def main():
    file_path = "input/A2-bank/bank-additional.csv"  
    bank_dataset = load_data(file_path)
    preprocessed_data = preprocess_data(bank_dataset)
    split_and_save(preprocessed_data, 'y', file_path_prefix="bank_")

if __name__ == "__main__":
    main()

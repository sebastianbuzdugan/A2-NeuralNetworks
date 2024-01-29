import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    column_names = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]
    return pd.read_csv(file_path, header=None, names=column_names)

# to split the dataset into training and testing sets and save them
def split_and_save(df, target_column, test_size=0.2, file_path_prefix=""):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
    
    X_train.to_csv(f"input/A2-personalized/{file_path_prefix}X_train.csv", index=False)
    X_test.to_csv(f"input/A2-personalized/{file_path_prefix}X_test.csv", index=False)
    y_train.to_csv(f"input/A2-personalized/{file_path_prefix}y_train.csv", index=False)
    y_test.to_csv(f"input/A2-personalized/{file_path_prefix}y_test.csv", index=False)

def main():
    file_path = "input/A2-personalized/data_banknote_authentication.txt"
    banknote_data = load_data(file_path)
    split_and_save(banknote_data, 'Class', file_path_prefix="banknote_")

if __name__ == "__main__":
    main()

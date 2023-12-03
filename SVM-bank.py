from SVM_Method import SVM_Class  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Kernel list
#svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
svm_kernels = [ 'poly']
svm_constant_min=1
svm_constant_max=1
svm_constant_step=1
svm_degree_min=4
svm_degree_max=4
svm_degree_step=1

min_error_aux=1

# Load the datasets
file_path = 'input/A2-bank/bank-additional-processed.csv'

# Load data into DataFrames
df_dataset = pd.read_csv(file_path, delimiter='\t', header=1)

# Display the first few rows of each dataset
#print("Separable Dataset:")
print(df_dataset.head())

# Convert the last column to integers
df_dataset[df_dataset.columns[-1]] = df_dataset[df_dataset.columns[-1]].astype(int)

#print(df_dataset)

# Extract features and labels
X_train, X_test, y_train, y_test  = train_test_split(
    df_dataset.iloc[:, :-1].values,
    df_dataset.iloc[:, -1].values,
    test_size=0.2,
    random_state=42
)

svmclass = SVM_Class()

svmclass.printPlots(df_dataset,'Bank')


try:
    for kernel in svm_kernels:
        for constant in range(svm_constant_min, svm_constant_max +1 , svm_constant_step):
            for degree in range(svm_degree_min, svm_degree_max +1 , svm_degree_step):
                print(f"PARAM> kernel={kernel} C={constant}")
                svmclass.svm_classification(X_train,X_test,y_train, y_test,kernel,constant,degree,'Bank',True)
    print(svmclass.best_kernel,svmclass.best_constant,svmclass.best_degree, svmclass.min_error)
except ValueError as e:
    # Code to handle the exception
    print(f"Error: {e}")
except Exception as e:
    # Code to handle other exceptions
    print(f"Unexpected error: {e}")

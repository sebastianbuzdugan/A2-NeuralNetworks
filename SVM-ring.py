from SVM_Method import SVM_Class  
import pandas as pd
import numpy as np

#Kernel list
#svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
svm_kernels = ['rbf']
svm_constant_min=701
svm_constant_max=701
svm_constant_step=1

min_error_aux=1

# Load the datasets
file_path_separable = 'input/A2-ring/A2-ring-separable.txt'
file_path_merged = 'input/A2-ring/A2-ring-merged.txt'
file_path_test = 'input/A2-ring/A2-ring-test.txt'

# Load data into DataFrames
df_separable = pd.read_csv(file_path_separable, delimiter='\t', header=None)
df_merged = pd.read_csv(file_path_merged, delimiter='\t', header=None)
df_test = pd.read_csv(file_path_test, delimiter='\t', header=None)

# Display the first few rows of each dataset
#print("Separable Dataset:")
#print(df_separable.head())

#print("\nMerged Dataset:")
#print(df_merged.head())

#print("\nTest Dataset:")
#print(df_test.head())

# Extract features and labels
print(df_separable)
X_train_separable = df_separable.iloc[:, :-1].values
y_train_separable = df_separable.iloc[:, -1].values

print(X_train_separable)
print(y_train_separable)

X_train_merged = df_merged.iloc[:, :-1].values
y_train_merged = df_merged.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

svmclass = SVM_Class()

svmclass.printPlots(df_separable,'Separable')
svmclass.printPlots(df_merged,'Merged')

try:
    for kernel in svm_kernels:
        for constant in range(svm_constant_min, svm_constant_max +1 , svm_constant_step):
            if constant > 0 :
                print(f"PARAM> kernel={kernel} C={constant} ")
                print(f"Separable dataset")
                svmclass.svm_classification(X_train_separable,X_test,y_train_separable, y_test,kernel,constant,1,'Separable',True)
                print(f"Merged dataset")
                svmclass.svm_classification(X_train_merged,X_test,y_train_merged, y_test,kernel,constant,1,'Merged',True)
    print(svmclass.best_kernel,svmclass.best_constant,svmclass.best_degree, svmclass.min_error)
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")


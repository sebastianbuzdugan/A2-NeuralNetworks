import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score

class SVM_Class:

    def __init__(self):
        self.min_error=1
        self.best_constant=0
        self.best_kernel=''
        self.best_degree=0

    def printPlots(self,df_plot,label):
        # Visualize the data for the separable dataset
        plt.scatter(
            df_plot[df_plot.iloc[:, -1] == 0].iloc[:, 0],
            df_plot[df_plot.iloc[:, -1] == 0].iloc[:, 1],
            color='blue',
            label='Class 0',
            s=3  # Adjust the point size as needed
        )
        plt.scatter(
            df_plot[df_plot.iloc[:, -1] == 1].iloc[:, 0],
            df_plot[df_plot.iloc[:, -1] == 1].iloc[:, 1],
            color='red',
            label='Class 1',
            s=3  # Adjust the point size as needed
        )
        plt.title(f'Scatter Plot of {label} Dataset')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
    
    def printPredictionPlots(self,X_test1,y_test1,y_pred1,label1):
        # Visualize the predictions for the test set

        plt.scatter(
            X_test1[y_test1 == 0][:, 0],
            X_test1[y_test1 == 0][:, 1],
            color='blue',
            label='True Class 0',
            s=3  # Adjust the point size as needed
        )
        plt.scatter(
            X_test1[y_test1 == 1][:, 0],
            X_test1[y_test1 == 1][:, 1],
            color='red',
            label='True Class 1',
            s=3  # Adjust the point size as needed
        )
        plt.scatter(
            X_test1[y_pred1 == 0][:, 0],
            X_test1[y_pred1 == 0][:, 1],
            facecolors='none',
            edgecolors='blue',
            label=f'Predicted Class 0 ({label1}])',
            s=30  # Adjust the point size as needed
        )
        plt.scatter(
            X_test1[y_pred1 == 1][:, 0],
            X_test1[y_pred1 == 1][:, 1],
            facecolors='none',
            edgecolors='red',
            label=f'Predicted Class 1 ({label1})',
            s=30  # Adjust the point size as needed
        )
        plt.title(f'Scatter Plot of Test Set Predictions ({label1}])')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

    def svm_classification(self, X_train,X_test,y_train,y_test,classification_kernel,constant,degree,label,plot):
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train SVM classifiers
        svm_classifier = SVC(kernel=classification_kernel, C=constant, gamma='scale', random_state=42,degree=degree)
        
        # Perform cross-validation
        cross_val_scores = cross_val_score(svm_classifier, X_train, y_train, cv=5)

        # Print the cross-validation scores
        print("Cross-Validation Scores:", cross_val_scores)

        # Calculate and print the mean cross-validation score
        mean_cv_score = cross_val_scores.mean()
        print("Mean Cross-Validation Score:", mean_cv_score)

        svm_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = svm_classifier.predict(X_test)

        # Evaluate the models
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy on the test set for dataset: {accuracy:.2f}")

        # Compute classification error
        error = 1 - accuracy
        if error < self.min_error:
            self.min_error=error
            self.best_constant=constant
            self.best_kernel=classification_kernel
            self.best_degree=degree

        print(f"Classification error on the test set for dataset: {error:.2%}")

        if plot:

             # Compute confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)

            print(f"\nConfusion Matrix for {label} Dataset:")
            print(conf_matrix)

            #Print classification report for both datasets
            print(f"\nClassification Report for {label} Dataset:")
            print(classification_report(y_test, y_pred))

            self.printPredictionPlots(X_test,y_test,y_pred,label)
        
            # Compute ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            

            #if plot:
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()

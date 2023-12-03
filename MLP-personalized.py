import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score

file_path = 'input/A2-personalized/data_banknote_authentication-processed.txt'

df_dataset = pd.read_csv(file_path, delimiter='\t', header=1)

# Display the first few rows of each dataset
#print("Separable Dataset:")
print(df_dataset.head())

# Convert the last column to integers
df_dataset[df_dataset.columns[-1]] = df_dataset[df_dataset.columns[-1]].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test  = train_test_split(
    df_dataset.iloc[:, :-1].values,
    df_dataset.iloc[:, -1].values,
    test_size=0.2,
    random_state=42
)

# Create and train an MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)


# Perform cross-validation
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", cross_val_scores)

# Calculate and print the mean cross-validation score
mean_cv_score = cross_val_scores.mean()
print("Mean Cross-Validation Score:", mean_cv_score)

model.fit(X_train, y_train.ravel())

# Make predictions on the test set
y_pred = model.predict(X_test)

print(y_pred)

threshold = 0.5  # Adjust the threshold as needed
y_pred_class = (y_pred > threshold).astype(int)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_class)
print(f'Mean Squared Error: {mse}')

# Evaluate the models
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy on the test set for dataset: {accuracy:.2f}")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(f"\nConfusion Matrix for Dataset:")
print(conf_matrix)

#Print classification report for both datasets
print(f"\nClassification Report for Dataset:")
print(classification_report(y_test, y_pred_class))


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
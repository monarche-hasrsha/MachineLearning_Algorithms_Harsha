# -*- coding: utf-8 -*-
"""random_forest_wine_dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Y2jI2BhDVLv1HfJNXLH0dWvf41dtt0PA
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

"""## Load and explore data

### Subtask:
Load the Wine dataset, create a pandas DataFrame, and perform initial data exploration (shape, head, class distribution).

"""

wine = load_wine()
df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df_wine['target'] = wine.target

print("Shape of the DataFrame:")
display(df_wine.shape)

print("\nFirst 5 rows of the DataFrame:")
display(df_wine.head())

print("\nClass distribution of the target variable:")
display(df_wine['target'].value_counts())

"""## Preprocess data

### Subtask:
Separate features and target, split data into training and testing sets, and scale the features using `StandardScaler`.

"""

X = df_wine.drop('target', axis=1)
y = df_wine['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

"""## Train random forest model

### Subtask:
Initialize and train the `RandomForestClassifier` on the training data.

"""

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

"""## Evaluate model

### Subtask:
Predict on the test set and evaluate the model's performance using accuracy, classification report, and confusion matrix.

"""

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""## Visualize with pca (optional)

### Subtask:
Apply PCA to reduce dimensionality and visualize the data in 2D, colored by actual and predicted classes.

"""

# Initialize PCA
pca = PCA(n_components=2, random_state=42)

# Fit PCA on scaled training data and transform training and testing data
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot PCA visualization colored by actual classes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis')
plt.title('PCA Visualization (Test Data) - Actual Classes')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter1, label='Actual Class')
plt.grid(True)

# Plot PCA visualization colored by predicted classes
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='viridis')
plt.title('PCA Visualization (Test Data) - Predicted Classes')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter2, label='Predicted Class')
plt.grid(True)

plt.tight_layout()
plt.show()

"""## Hyperparameter tuning (optional)

### Subtask:
Experiment with different hyperparameters for the Random Forest model to potentially improve performance.

"""

# parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

#  GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5, # Using 5-fold cross-validation
                           n_jobs=-1) # Use all available cores

# Fit GridSearchCV to the scaled training data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

"""**Reasoning**:
Train a new RandomForestClassifier using the best parameters found by GridSearchCV and evaluate its performance on the test set.


"""

# Train a new model with the best parameters
best_model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                   max_depth=grid_search.best_params_['max_depth'],
                                   min_samples_split=grid_search.best_params_['min_samples_split'],
                                   random_state=42)

best_model.fit(X_train, y_train)

# Evaluate the performance of the best model on the test set
y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"\nAccuracy with best parameters: {accuracy_best:.4f}")

print("\nClassification Report with best parameters:")
print(classification_report(y_test, y_pred_best))

conf_matrix_best = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix with Best Parameters')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""## Summary:

### Data Analysis Key Findings

*   The Wine dataset was successfully loaded and explored, revealing a shape of (178, 14) and a class distribution of 71 instances for class 1, 59 for class 0, and 48 for class 2.
*   The data was successfully preprocessed by separating features and target, splitting into training (142 samples) and testing (36 samples) sets, and scaling the features.
*   A Random Forest Classifier was trained on the scaled training data.
*   The initial Random Forest model achieved perfect accuracy (1.0000) on the test set, with perfect precision, recall, and F1-score (1.00) for all classes.
*   PCA was successfully applied to reduce the dimensionality to 2 components for visualization.
*   Hyperparameter tuning using `GridSearchCV` identified `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}` as the best parameters.
*   Training and evaluating a Random Forest model with the best parameters also resulted in perfect accuracy (1.0000) on the test set.

### Insights or Next Steps

*   The Wine dataset appears to be highly separable, as evidenced by the perfect classification performance of the default Random Forest model on the test set.
*   While hyperparameter tuning confirmed perfect performance, it might be beneficial to test the model on a different split or a larger dataset if available to ensure robustness.

"""
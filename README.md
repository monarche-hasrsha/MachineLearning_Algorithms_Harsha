# MachineLearning_Algorithms_Harsha

This repository contains three Jupyter notebooks showcasing supervised classification using three powerful machine learning algorithms on standard datasets from scikit-learn:

K-Nearest Neighbors (KNN) on the Digits dataset

Support Vector Machine (SVM) on the Breast Cancer dataset

Random Forest on the Wine dataset

These notebooks demonstrate the full workflow: loading data, preprocessing, training models, making predictions, and evaluating performance.

📘 1. KNN on Digits Dataset
Dataset:
From sklearn.datasets

1797 grayscale images of handwritten digits (0–9)

Each image is 8×8 pixels (64 features)

Steps:
Import Libraries: numpy, matplotlib, seaborn, sklearn

Load Dataset: load_digits()

Visualize Samples: Display example digit images using imshow()

Flatten Images: Convert 8×8 matrix to 1D array (64 pixels)

Train/Test Split

Model Training: Use KNeighborsClassifier with chosen k

Predict & Evaluate: Confusion matrix, accuracy, classification report

📗 2. SVM on Breast Cancer Dataset
Dataset:
Binary classification: malignant vs. benign

30 features extracted from digitized images of breast mass

From load_breast_cancer() in sklearn.datasets

Steps:
Load Data: Convert into pandas DataFrame

Explore Dataset: Check features, class balance

Visualize: Correlation matrix, count plots (optional)

Split Data

Model Training: SVC() with linear or RBF kernel

Prediction

Evaluation: Accuracy, confusion matrix, classification report

🍷 3. Random Forest on Wine Dataset
Dataset:
Multiclass classification of 3 types of wine

13 chemical features per wine sample

From load_wine() in sklearn.datasets

Steps:
Import Libraries: pandas, seaborn, sklearn

Load Dataset: Store as a DataFrame

Explore & Visualize: Use head(), info(), and visualizations like heatmaps

Split Data: train_test_split()

Model Training: Use RandomForestClassifier(n_estimators=100)

Prediction

Evaluation: Accuracy, confusion matrix, classification report

Feature Importance (Optional): Plot most important features
🚢 Decision Tree Classifier on Titanic Dataset
This Jupyter Notebook demonstrates how to build and evaluate a Decision Tree Classification model on the classic Titanic dataset using scikit-learn. The goal is to predict whether a passenger survived or not based on features like age, sex, class, etc.

📁 Dataset
Source: Kaggle Titanic Dataset (or via seaborn or other CSV imports)

Target variable: Survived (0 = No, 1 = Yes)

🧠 Steps Involved
1. Import Libraries
pandas, numpy for data manipulation

seaborn, matplotlib for visualization

sklearn for model building and evaluation

2. Load Dataset
Load the Titanic dataset into a Pandas DataFrame using read_csv().

Use df.head(), df.info() and df.describe() to understand the dataset structure.

3. Data Preprocessing
Handle missing values (e.g., fill missing Age with mean or drop rows).

Convert categorical variables (e.g., Sex, Embarked) into numeric form using LabelEncoder or pd.get_dummies.

4. Feature Selection
Choose relevant features (e.g., Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) for prediction.

Drop unnecessary columns like PassengerId, Name, Ticket, Cabin.

5. Train-Test Split
Use train_test_split() from sklearn.model_selection to divide the dataset into training and testing sets (e.g., 70/30 split).

6. Train Decision Tree Model
Initialize DecisionTreeClassifier() from sklearn.tree.

Fit the model using model.fit(X_train, y_train).

7. Model Evaluation
Predict on the test set using model.predict().

Evaluate using:

Accuracy Score

Confusion Matrix

Classification Report (precision, recall, F1-score)

8. (Optional) Visualize Decision Tree
Use plot_tree() from sklearn.tree or export as .dot file to visualize the tree structure.




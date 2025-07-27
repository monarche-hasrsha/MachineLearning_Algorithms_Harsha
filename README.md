Github rep:- https://github.com/monarche-hasrsha/MachineLearning_Algorithms_Harsha
Supervised Machine Learning Classifiers with scikit-learn
This repository contains four Jupyter Notebooks showcasing powerful supervised classification algorithms applied to popular datasets from scikit-learn. These notebooks demonstrate a complete machine learning workflow — from loading data to evaluation and visualization.

📂 Contents
📘 K-Nearest Neighbors (KNN) – Digits Dataset

📗 Support Vector Machine (SVM) – Breast Cancer Dataset

🍷 Random Forest – Wine Dataset

🚢 Decision Tree – Titanic Dataset

📘 1. KNN on Digits Dataset
📊 Dataset:
Source: sklearn.datasets.load_digits()

Samples: 1797 grayscale images of handwritten digits (0–9)

Features: Each image is 8×8 pixels → 64 numeric features

🔧 Steps:
Import libraries: numpy, matplotlib, seaborn, sklearn

Load dataset

Visualize samples using imshow()

Flatten 8×8 images into 1D vectors

Train-test split

Train model using KNeighborsClassifier

Predict & evaluate:

Accuracy score

Confusion matrix

Classification report

📗 2. SVM on Breast Cancer Dataset
📊 Dataset:
Source: sklearn.datasets.load_breast_cancer()

Type: Binary classification (Malignant vs. Benign)

Features: 30 numerical attributes from breast mass images

🔧 Steps:
Load data and convert to Pandas DataFrame

Explore dataset: info(), describe(), class balance

(Optional) Visualize with heatmaps or count plots

Train-test split

Train model using SVC() (linear/RBF kernel)

Predict & evaluate:

Accuracy

Confusion matrix

Classification report

🍷 3. Random Forest on Wine Dataset
📊 Dataset:
Source: sklearn.datasets.load_wine()

Type: Multiclass classification (3 wine types)

Features: 13 chemical properties per sample

🔧 Steps:
Import data into Pandas DataFrame

Explore with .head(), .info(), heatmaps

Train-test split

Train RandomForestClassifier(n_estimators=100)

Predict & evaluate:

Accuracy

Confusion matrix

Classification report

(Optional) Plot feature importance

🚢 4. Decision Tree on Titanic Dataset
📊 Dataset:
Source: Kaggle Titanic Dataset

Type: Binary classification (Survived = 1, Not Survived = 0)

🔧 Steps:
Load dataset using read_csv()

Inspect data using head(), info(), describe()

Preprocess:

Fill missing values (e.g., Age)

Encode categorical columns (Sex, Embarked) using LabelEncoder or pd.get_dummies

Select features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

Drop unused columns: PassengerId, Name, Ticket, Cabin

Train-test split (e.g., 70/30)

Train model using DecisionTreeClassifier()

Predict & evaluate:

Accuracy

Confusion matrix

Classification report

(Optional) Visualize tree using plot_tree()

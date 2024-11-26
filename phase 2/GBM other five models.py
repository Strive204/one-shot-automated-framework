import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load the dataset
data_path = "ceshi.csv"
data = pd.read_csv(data_path)

# Prepare data
X = data.iloc[:, :-1]  # All columns except the last
y = data.iloc[:, -1]  # The last column as the target

# Encode target labels if necessary
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Models
models = {
    "Naive Bayes Classifier": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Support Vector Classifier": SVC(),
    "K-Nearest Neighbors Classifier": KNeighborsClassifier()
}

# To store results
classification_results = []

# Training and Evaluating traditional models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Save predictions and actual values to CSV
    train_results = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
    test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
    train_results.to_csv(f'{name}_train_results.csv', index=False)
    test_results.to_csv(f'{name}_test_results.csv', index=False)

    # Save precision, recall, and F1-score results to CSV
    model_results = pd.DataFrame({
        'Metric': ['Training Precision', 'Training Recall', 'Training F1', 'Test Precision', 'Test Recall', 'Test F1'],
        'Value': [train_precision, train_recall, train_f1, test_precision, test_recall, test_f1]
    })
    model_results.to_csv(f'{name}_metrics_results.csv', index=False)

    classification_results.append({
        'Model': name,
        'Training Precision': train_precision,
        'Training Recall': train_recall,
        'Training F1': train_f1,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1': test_f1
    })

# Displaying results
classification_results_df = pd.DataFrame(classification_results)
print(classification_results_df)

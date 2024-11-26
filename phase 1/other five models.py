import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
data_path = "扩展后的数据集.csv"
data = pd.read_csv(data_path)

# Prepare data
X = data.iloc[:, :-1]  # All columns except the last
y = data.iloc[:, -1]  # The last column as the target

# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest Regressor": RandomForestRegressor(),
    "Lasso Regression": Lasso(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Extra Trees Regressor": ExtraTreesRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "XGBoost Regressor": XGBRegressor()
}

# To store results
regression_results = []

# Training and Evaluating traditional models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Save predictions and actual values to CSV
    train_results = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
    test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
    train_results.to_csv(f'{name}_train_results.csv', index=False)
    test_results.to_csv(f'{name}_test_results.csv', index=False)

    # Save R² and RMSE results to CSV
    model_results = pd.DataFrame({
        'Metric': ['Training R²', 'Training RMSE', 'Test R²', 'Test RMSE'],
        'Value': [train_r2, train_rmse, test_r2, test_rmse]
    })
    model_results.to_csv(f'{name}_metrics_results.csv', index=False)

    regression_results.append({
        'Model': name,
        'Training R²': train_r2,
        'Training RMSE': train_rmse,
        'Test R²': test_r2,
        'Test RMSE': test_rmse
    })

# Displaying results
regression_results_df = pd.DataFrame(regression_results)
print(regression_results_df)

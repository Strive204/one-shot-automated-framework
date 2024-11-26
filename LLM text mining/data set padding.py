# Import libraries
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split

# Initialize H2O
h2o.init()

# Read data
df_all = pd.read_csv("第一个总数据集.csv", encoding='gbk')

# 处理可能存在的字符串 '<0.01'，并转换为合适的数值
df_all = df_all.replace('<0.01', 0.01)


# Function to impute features using H2O AutoML
def feature_impute_h2o(df_all, target):
    '''
    Uses H2O AutoML to predict and impute null values.
    H2O will automatically find the best model for the imputation task.
    Args:
        df_all: DataFrame
        target: feature name to be imputed

    Returns:
        DataFrame with imputed data
    '''
    # Prepare data for H2O
    df_mv = df_all[["Surface Area", target]].copy()
    df_mv = df_mv.dropna()

    # Split into features (x) and target (y)
    x = np.array(df_mv["Surface Area"]).reshape(-1, 1)
    y = np.array(df_mv[target]).reshape(-1, 1)

    # Convert to H2O frames
    df_h2o = h2o.H2OFrame(df_mv)

    # Define features and target for H2O
    features = ["Surface Area"]
    target = target

    # Split the data into training and test sets
    train, test = df_h2o.split_frame(ratios=[.8])

    # Run H2O AutoML for 20 base models
    aml = H2OAutoML(max_models=20, seed=42)
    aml.train(x=features, y=target, training_frame=train)

    # Get the leader model
    leaderboard = aml.leaderboard
    print(leaderboard)

    # Predict using the leader model
    preds = aml.leader.predict(test)
    print(f"Test R2 for {target}: {aml.leader.r2()}")

    # Impute missing values
    missing_data = h2o.H2OFrame(df_all[["Surface Area"]])
    imputed = aml.leader.predict(missing_data)

    # Convert the H2O frame back to pandas
    df_all[target + "_imputed"] = h2o.as_list(imputed)

    return df_all


# Impute the features using H2O AutoML
# df_all = feature_impute_h2o(df_all, "Total Pore Volume")
# df_all = feature_impute_h2o(df_all, "Micropore Volume")
# df_all = feature_impute_h2o(df_all, "Carbon Content")
# df_all = feature_impute_h2o(df_all, "Hydrogen Content")
# df_all = feature_impute_h2o(df_all, "Nitrogen Content")
df_all = feature_impute_h2o(df_all, "Oxygen Content")

# Fill missing values
# df_all["Total Pore Volume"] = df_all["Total Pore Volume"].fillna(df_all["Total Pore Volume_imputed"])
# df_all["Micropore Volume"] = df_all["Micropore Volume"].fillna(df_all["Micropore Volume_imputed"])
# df_all["Carbon Content"] = df_all["Carbon Content"].fillna(df_all["Carbon Content_imputed"])
# df_all["Hydrogen Content"] = df_all["Hydrogen Content"].fillna(df_all["Hydrogen Content_imputed"])
# df_all["Nitrogen Content"] = df_all["Nitrogen Content"].fillna(df_all["Nitrogen Content_imputed"])
df_all["Oxygen Content"] = df_all["Oxygen Content"].fillna(df_all["Oxygen Content_imputed"])

# Save the processed dataset to an Excel file
df_all.to_excel("1.xlsx", index=False)

# Processed dataset
df_all.head()

# Shutdown H2O
h2o.shutdown()

import h2o
import pandas as pd
import matplotlib.pyplot as plt
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

h2o.init()

data = pd.read_csv("initial.csv")

df_h2o = h2o.H2OFrame(data)

X = df_h2o.columns[:-1]
y = df_h2o.columns[-1]

train, test = df_h2o.split_frame(ratios=[0.8])

aml = H2OAutoML(max_runtime_secs=900, seed=1234)

aml.train(x=X, y=y, training_frame=train)

best_model = aml.leader

train_predictions = best_model.predict(train).as_data_frame(use_pandas=True)
test_predictions = best_model.predict(test).as_data_frame(use_pandas=True)

train_df = train.as_data_frame(use_pandas=True)
test_df = test.as_data_frame(use_pandas=True)

train_r2 = r2_score(train_df[y], train_predictions['predict'])
train_rmse = mean_squared_error(train_df[y], train_predictions['predict'], squared=False)

test_r2 = r2_score(test_df[y], test_predictions['predict'])
test_rmse = mean_squared_error(test_df[y], test_predictions['predict'], squared=False)

print(f"Training R²: {train_r2}")
print(f"Training RMSE: {train_rmse}")
print(f"Test R²: {test_r2}")
print(f"Test RMSE: {test_rmse}")

plt.figure(figsize=(10, 6))
plt.scatter(test_df[y], test_predictions['predict'], alpha=0.5)
plt.plot([test_df[y].min(), test_df[y].max()], [test_df[y].min(), test_df[y].max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values for Test Set')
plt.show()

test_results_df = pd.DataFrame({
    'Actual': test_df[y],
    'Predicted': test_predictions['predict']
})
test_results_df.to_csv('test_actual_vs_predicted.csv', index=False)

train_results_df = pd.DataFrame({
    'Actual': train_df[y],
    'Predicted': train_predictions['predict']
})
train_results_df.to_csv('train_actual_vs_predicted.csv', index=False)

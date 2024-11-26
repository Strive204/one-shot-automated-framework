import h2o
import pandas as pd
import numpy as np
from h2o.automl import H2OAutoML
from sklearn.metrics import r2_score, mean_squared_error

h2o.init()

data = pd.read_csv("../Third_Backward/initial.csv")

data['S (%)'] = pd.to_numeric(data['S (%)'], errors='coerce')

df_h2o = h2o.H2OFrame(data)

df_h2o['S (%)'] = df_h2o['S (%)'].asnumeric()

X = df_h2o.columns[:-1]
y = df_h2o.columns[-1]

aml = H2OAutoML(max_runtime_secs=60, seed=1234)

aml.train(x=X, y=y, training_frame=df_h2o)

best_model = aml.leader

df_train = df_h2o.as_data_frame()
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

h2o_X_train = h2o.H2OFrame(X_train)
preds = best_model.predict(h2o_X_train).as_data_frame().values.flatten()

r2 = r2_score(y_train, preds)
mse = mean_squared_error(y_train, preds)

print(f"Best model R2: {r2}")
print(f"Best model MSE: {mse}")

num_samples = 30000
num_features = len(X_train.columns)
np.random.seed(1234)
random_data = np.random.rand(num_samples, num_features) * (X_train.max().values - X_train.min().values) + X_train.min().values

h2o_random_data = h2o.H2OFrame(random_data, column_names=X_train.columns.tolist())

predictions = best_model.predict(h2o_random_data).as_data_frame().values.flatten()

h2o_random_data_validation = h2o.H2OFrame(random_data, column_names=X_train.columns.tolist())
validation_predictions = best_model.predict(h2o_random_data_validation).as_data_frame().values.flatten()

random_data_df = pd.DataFrame(random_data, columns=X_train.columns)
random_data_df['Predicted CO2 Capture Efficiency'] = predictions
random_data_df['Validation Predicted CO2 Capture Efficiency'] = validation_predictions
random_data_df.to_csv('data_predictions.csv', index=False)

print("generated 30000 samples and predictions saved.")

h2o.shutdown()

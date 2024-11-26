import pandas as pd
import h2o
from h2o.automl import H2OAutoML

h2o.init()

data_path = "ceshi.csv"
data = pd.read_csv(data_path)

X = data.drop('Preparation Method', axis=1)
y = data['Preparation Method']

h2o_data = h2o.H2OFrame(pd.concat([X, y], axis=1))

h2o_data['Preparation Method'] = h2o_data['Preparation Method'].asfactor()

x = list(X.columns)
y = 'Preparation Method'

train, test = h2o_data.split_frame(ratios=[0.8], seed=42)

aml = H2OAutoML(max_runtime_secs=60, seed=42)
aml.train(x=x, y=y, training_frame=train)

best_model = aml.leader

def predict_new_data(new_data):
    h2o_new_data = h2o.H2OFrame(new_data)
    prediction = best_model.predict(h2o_new_data)
    return prediction.as_data_frame()['predict'].values

new_data = pd.DataFrame({
    'Ingredients': [37],
    'Total Pore Volume ': [36.05639995],
    'Carbon Content': [33.47520429],
    'Hydrogen Content': [0],
    'Nitrogen Content': [0],
    'Temperature': [0],
    'Pressure ': [61.47991866],
    'Surface Area': [2001.077046],
    'Micropore Volume': [26.63015203],
    'CO2 Capture Efficiency': [9.168700459]
})
prediction = predict_new_data(new_data)
print(f"Predicted Preparation Method: {prediction}")

h2o.shutdown(prompt=False)

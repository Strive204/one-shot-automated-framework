import h2o
import pandas as pd
import random
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings

h2o.init()

data = pd.read_csv("extended_dataset.csv")

df_h2o = h2o.H2OFrame(data)

X = df_h2o.columns[:-1]
y = df_h2o.columns[-1]

train, test = df_h2o.split_frame(ratios=[0.8])

aml = H2OAutoML(max_runtime_secs=300, seed=1234)

aml.train(x=X, y=y, training_frame=train)

best_model = aml.leader

target_y = float(input("Please enter the target variable value:"))

n_solutions = 50
X_ranges = {
    "Carbon Content": (0, 99.99),
    "Hydrogen Content": (0, 99.99),
    "Nitrogen Content": (0, 99.99),
    "Micropore Volume": (0, 1),
    "Total Pore Volume": (0, 1),
    "Surface Area": (0, 1000),
    "Pressure": (0, 50),
    "Temperature": (0, 2000)
}
X_names = data.columns[:-1].tolist()

solutions = pd.DataFrame()

while len(solutions) < n_solutions:
    current_solution = data.iloc[random.randint(0, len(data)-1)].copy()
    for name in X_names:
        value_range = X_ranges[name]
        direction = random.choice([-1, 1])
        step_size = random.uniform(0, (value_range[1] - value_range[0]) / 2)
        new_value = max(value_range[0], min(value_range[1], current_solution[name] + direction * step_size))
        current_solution[name] = new_value

    sum_CHN = sum(current_solution[col] for col in ["Carbon Content", "Hydrogen Content", "Nitrogen Content"])
    if sum_CHN > 100:
        factor = 100 / sum_CHN
        for col in ["Carbon Content", "Hydrogen Content", "Nitrogen Content"]:
            current_solution[col] *= factor

    current_solution_h2o = h2o.H2OFrame([current_solution.tolist()], column_names=X_names)
    with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
        pred_y = best_model.predict(current_solution_h2o).as_data_frame().iloc[0, 0]

    if abs(pred_y - target_y) < 0.5:
        current_solution['Predicted Value'] = pred_y
        solutions = pd.concat([solutions, current_solution.to_frame().T], ignore_index=True)

    print(f"Currently collected {len(solutions)} solutions.")

if solutions.empty:
    print("No solutions found!")
else:
    solutions.to_csv('solutions.csv', index=False)
    print("Solutions and their predicted values have been saved to 'solutions.csv'.")

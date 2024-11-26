import h2o
import pandas as pd
import random
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings

h2o.init()

data = pd.read_csv("../第三次反向/扩展后数据.csv")

df_h2o = h2o.H2OFrame(data)

X = df_h2o.columns[:-1]
y = df_h2o.columns[-1]

train, test = df_h2o.split_frame(ratios=[0.8])

aml = H2OAutoML(max_runtime_secs=300, seed=1234)

aml.train(x=X, y=y, training_frame=train)

best_model = aml.leader

target_y = float(input("Enter the target variable value:"))

n_solutions = 20
X_ranges = {
    "C (%)": (0, 99.9),
    "H (%)": (0, 99.9),
    "O (%)": (0, 99.9),
    "N (%)": (0, 99.9),
    "S (%)": (0, 99.9),
    "Ash (%)": (0, 99.9),
    "H/C": (0, 2),
    "O/C": (0, 1),
    "T (K)": (0, 2000)
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

    sum_CHN = sum(current_solution[col] for col in ["C (%)", "H (%)", "O (%)", "N (%)", "S (%)"])
    if sum_CHN > 100:
        factor = 100 / sum_CHN
        for col in ["C (%)", "H (%)", "O (%)", "N (%)", "S (%)"]:
            current_solution[col] *= factor

    current_solution_h2o = h2o.H2OFrame([current_solution.tolist()], column_names=X_names)
    with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
        pred_y = best_model.predict(current_solution_h2o).as_data_frame().iloc[0, 0]

    if abs(pred_y - target_y) < 0.05:
        current_solution['Predicted Value'] = pred_y
        solutions = pd.concat([solutions, current_solution.to_frame().T], ignore_index=True)

    print(f"{len(solutions)} solutions collected so far.")

if solutions.empty:
    print("No solutions found!")
else:
    solutions.to_csv('solutions.csv', index=False)
    print("Solutions and their predicted values have been saved to 'solutions.csv'.")

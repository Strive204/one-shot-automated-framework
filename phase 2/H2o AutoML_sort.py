import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score

h2o.init()

data_path = ("ceshi.csv")
data_h2o = h2o.upload_file(data_path)

predictors = data_h2o.columns[:-1]
target = data_h2o.columns[-1]

data_h2o[target] = data_h2o[target].asfactor()

train, test = data_h2o.split_frame(ratios=[0.8], seed=123)

aml = H2OAutoML(max_runtime_secs=1500)
aml.train(x=predictors, y=target, training_frame=train, validation_frame=test)

print(aml.leaderboard)

preds_train = aml.predict(train)
preds_test = aml.predict(test)

y_train = h2o.as_list(train[target], use_pandas=True)
y_test = h2o.as_list(test[target], use_pandas=True)
preds_train = h2o.as_list(preds_train['predict'], use_pandas=True)
preds_test = h2o.as_list(preds_test['predict'], use_pandas=True)

y_train_series = y_train.squeeze()
y_test_series = y_test.squeeze()
preds_train_series = preds_train.squeeze()
preds_test_series = preds_test.squeeze()

accuracy_train = accuracy_score(y_train_series, preds_train_series)
recall_train = recall_score(y_train_series, preds_train_series, average='macro')
f1_train = f1_score(y_train_series, preds_train_series, average='macro')

accuracy_test = accuracy_score(y_test_series, preds_test_series)
recall_test = recall_score(y_test_series, preds_test_series, average='macro')
f1_test = f1_score(y_test_series, preds_test_series, average='macro')

print(f'''
Training accuracy: {accuracy_train}
Training recall: {recall_train}
Training F1 score: {f1_train}
Test accuracy: {accuracy_test}
Test recall: {recall_test}
Test F1 score: {f1_test}
''')

results_df = pd.DataFrame({
    'Actual values': y_test_series,
    'Predicted values': preds_test_series
})

results_filename = 'test_predictions.csv'
results_df.to_csv(results_filename, index=False)
print(f"Results saved to {results_filename}")

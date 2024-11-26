import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingRegressor

data = pd.read_csv('extended_dataset.csv')

y = data.iloc[:, -1]
X = data.iloc[:, :-1]

model = GradientBoostingRegressor()
model.fit(X, y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.gcf().set_size_inches(10, 6)
plt.gcf().set_dpi(100)
plt.xlabel(plt.gca().get_xlabel(), fontsize=18, fontname='Times New Roman')
plt.ylabel(plt.gca().get_ylabel(), fontsize=18, fontname='Times New Roman')
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.savefig('shap_bar_plot.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.gcf().set_size_inches(10, 6)
plt.gcf().set_dpi(100)
plt.xlabel(plt.gca().get_xlabel(), fontsize=18, fontname='Times New Roman')
plt.ylabel(plt.gca().get_ylabel(), fontsize=18, fontname='Times New Roman')
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.savefig('shap_beeswarm_plot.png', bbox_inches='tight')
plt.show()

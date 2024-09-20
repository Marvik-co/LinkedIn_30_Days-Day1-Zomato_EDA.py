import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from pycebox.ice import ice, ice_plot

# Step 1: Load California Housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Fit a model (e.g., DecisionTreeRegressor)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 4: Define a prediction function
def predict_fn(X):
    return model.predict(X)

# Step 5: Generate ICE data for a specific feature (e.g., 'MedInc' - Median Income)
feature = 'MedInc'
ice_data = ice(X_test, feature, predict_fn, num_grid_points=50)

# Randomly sample 100 of the ICE curves to plot
sampled_ice_data = ice_data.iloc[:, :100]

# Step 7: Plot ICE curves
fig, ax = plt.subplots(figsize=(10, 6))
ice_plot(sampled_ice_data, frac_to_plot=1.0, ax=ax) 
plt.title(f"ICE Plot for feature '{feature}'")
plt.xlabel(feature)
plt.ylabel('Predicted House Value')

# Adjust line width for all lines in the plot
for line in ax.get_lines():
    line.set_linewidth(0.5)  # Set line width to 0.5 for finer lines

plt.show()

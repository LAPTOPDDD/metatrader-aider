import numpy as np
import pandas as pd
from tabpfn import TabFMRegressor

# Choose your backend:

# OPTION A: JAX Backend
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
model = tabfm_v1_0_0.load(model_type="regression")

# OPTION B: PyTorch Backend
# from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
# model = tabfm_v1_0_0.load(model_type="regression")

# Initialize scikit-learn compatible regressor (works with either backend model)
reg = TabFMRegressor(model=model)

# Prepare your dataset
X_train = pd.DataFrame({
    "sqft": [1200, 2500, 1500, 3000],
    "neighborhood": ["A", "B", "A", "C"]
})
y_train = np.array([250000, 550000, 310000, 620000])

X_test = pd.DataFrame({
    "sqft": [1800, 2800],
    "neighborhood": ["A", "B"]
})

# Fit and Predict
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

print("Predicted Prices:", predictions)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_parquet("~/pro_v2/output/wesad_eda_features_phase3.parquet")

# Define feature columns (exclude metadata)
feature_cols = [c for c in df.columns if c not in ["subject", "window_id", "binary_label"]]

# Exclude subject S4 from the scaler fitting
train_df = df[df["subject"] != "S10"]
X_train = train_df[feature_cols].values

# Fit scaler only on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Print the 15 means and 15 standard deviations
print("MEAN =", scaler.mean_.tolist())
print("STD  =", scaler.scale_.tolist())
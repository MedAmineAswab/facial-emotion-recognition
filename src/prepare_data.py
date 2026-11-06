
import sys
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical

# ======================
# Configuration
# ======================
base_path = "/content/drive/MyDrive/emotion-detection"

# Ajouter src au path 
sys.path.append(os.path.join(base_path, "src"))

# ======================
# Importer les modules 
# ======================
from data.load_data import load_data
from data.preprocess import preprocess_data
from data.split import split_data

# ======================
# Load data
# ======================
csv_path = os.path.join(base_path, "data", "raw", "fer2013.csv")

df = load_data(csv_path)

# ======================
# Extraction
# ======================
images = df["pixels"]
labels = df["emotion"]

# ======================
# Preprocess
# ======================
X = preprocess_data(images)
print("X shape:", X.shape)

# ======================
# Encoding labels
# ======================
Y = to_categorical(labels, num_classes=7)
print("Y shape:", Y.shape)

# ======================
# Split
# ======================
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X, Y, df)

print("=== Split shapes ===")
print("Train:", X_train.shape, Y_train.shape)
print("Val  :", X_val.shape, Y_val.shape)
print("Test :", X_test.shape, Y_test.shape)

# ======================
# Save data
# ======================
processed_data_path = os.path.join(base_path, "data", "processed")
os.makedirs(processed_data_path, exist_ok=True)

pd.to_pickle(X_train, os.path.join(processed_data_path, "X_train.pkl"))
pd.to_pickle(Y_train, os.path.join(processed_data_path, "Y_train.pkl"))

pd.to_pickle(X_val, os.path.join(processed_data_path, "X_val.pkl"))
pd.to_pickle(Y_val, os.path.join(processed_data_path, "Y_val.pkl"))

pd.to_pickle(X_test, os.path.join(processed_data_path, "X_test.pkl"))
pd.to_pickle(Y_test, os.path.join(processed_data_path, "Y_test.pkl"))

print(" Data saved successfully in:", processed_data_path)
  
  
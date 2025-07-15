# Import necessary libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from joblib import parallel_backend
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from torch.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('data.csv')
print("Missing values before handling:\n", df.isnull().sum())

# Fill missing values (median for numerical, mode for categorical)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)
print("\nMissing values after handling:\n", df.isnull().sum())

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
print("\nCategorical columns encoded successfully.")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Normalize features
X = StandardScaler().fit_transform(X)

# Apply SMOTE to balance dataset
X, y = SMOTE(random_state=42).fit_resample(X, y)
print(f"\nClass distribution after SMOTE:\n{pd.Series(y).value_counts()}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {X_train.shape[0]}\nTesting samples: {X_test.shape[0]}")

# ==== Serial Random Forest ====
start_serial = time.time()
rf_serial = RandomForestClassifier(n_estimators=100, random_state=42)
rf_serial.fit(X_train, y_train)
y_pred_serial = rf_serial.predict(X_test)
time_serial = time.time() - start_serial
acc_serial = accuracy_score(y_test, y_pred_serial)
f1_serial = f1_score(y_test, y_pred_serial)
print("\n=== Serial RF ===")
print(f"Accuracy: {acc_serial:.4f}, F1: {f1_serial:.4f}, Time: {time_serial:.2f}s")
print(confusion_matrix(y_test, y_pred_serial))
print(classification_report(y_test, y_pred_serial))

# ==== Parallel Random Forest ====
start_parallel = time.time()
with parallel_backend('threading', n_jobs=-1):
    rf_parallel = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_parallel.fit(X_train, y_train)
y_pred_parallel = rf_parallel.predict(X_test)
time_parallel = time.time() - start_parallel
acc_parallel = accuracy_score(y_test, y_pred_parallel)
f1_parallel = f1_score(y_test, y_pred_parallel)
print("\n=== Parallel RF ===")
print(f"Accuracy: {acc_parallel:.4f}, F1: {f1_parallel:.4f}, Time: {time_parallel:.2f}s")
print(confusion_matrix(y_test, y_pred_parallel))
print(classification_report(y_test, y_pred_parallel))

# ==== PyTorch Model with GPU Optimization ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

# Define dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Define binary classification model
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

model = BinaryClassifier(X_train.shape[1]).to(device)
pos_weight = torch.tensor([sum(y_train==0)/sum(y_train==1)], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

train_losses, best_f1 = [], 0
start_dl = time.time()

# Train the model
for epoch in range(25):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    model.eval()
    with torch.no_grad():
        val_logits = model(X_test_tensor)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs > 0.5).cpu().numpy().astype(int)
        f1_val = f1_score(y_test, val_preds)
        if f1_val > best_f1:
            best_f1 = f1_val
        elif epoch > 3 and f1_val < best_f1 - 0.02:
            print(f"Early stopping at epoch {epoch+1}")
            break
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, F1: {f1_val:.4f}")

dl_time = time.time() - start_dl
acc_dl = accuracy_score(y_test, val_preds)
f1_dl = f1_score(y_test, val_preds)
print("\n=== PyTorch ===")
print(f"Accuracy: {acc_dl:.4f}, F1: {f1_dl:.4f}, Time: {dl_time:.2f}s")
print(confusion_matrix(y_test, val_preds))
print(classification_report(y_test, val_preds))

# ==== XGBoost ====
start_xgb = time.time()
xgb = XGBClassifier(tree_method="hist", device="cuda")
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
time_xgb = time.time() - start_xgb
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
print("\n=== XGBoost ===")
print(f"Accuracy: {acc_xgb:.4f}, F1: {f1_xgb:.4f}, Time: {time_xgb:.2f}s")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# ==== CatBoost ====
start_cat = time.time()
cat = CatBoostClassifier(verbose=0)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
time_cat = time.time() - start_cat
acc_cat = accuracy_score(y_test, y_pred_cat)
f1_cat = f1_score(y_test, y_pred_cat)
print("\n=== CatBoost ===")
print(f"Accuracy: {acc_cat:.4f}, F1: {f1_cat:.4f}, Time: {time_cat:.2f}s")
print(confusion_matrix(y_test, y_pred_cat))
print(classification_report(y_test, y_pred_cat))

# ==== Speedup Summary ====
speedup_dl = max(0.0, round((time_serial - dl_time) / time_serial * 100, 2))
speedup_xgb = max(0.0, round((time_serial - time_xgb) / time_serial * 100, 2))
speedup_cat = max(0.0, round((time_serial - time_cat) / time_serial * 100, 2))

cpu_total_time = time_serial + time_parallel + time_xgb + time_cat
gpu_total_time = dl_time
gpu_speedup_over_cpu = max(0.0, round((cpu_total_time - gpu_total_time) / cpu_total_time * 100, 2))

# Display time/speedup summary
print(f"\nPyTorch Speedup: {speedup_dl}%")
print(f"XGBoost Speedup: {speedup_xgb}%")
print(f"CatBoost Speedup: {speedup_cat}%")
print(f"\nTotal CPU Models Time: {cpu_total_time:.2f}s")
print(f"Total GPU Model Time (PyTorch): {gpu_total_time:.2f}s")
print(f"GPU Speedup over CPU models: {gpu_speedup_over_cpu}%")

# CUDA info
print(f"\nPyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available, running on CPU")

# ==== Best Model Summary ====
model_scores = {
    "Serial RF": (acc_serial + f1_serial) / 2,
    "Parallel RF": (acc_parallel + f1_parallel) / 2,
    "PyTorch": (acc_dl + f1_dl) / 2,
    "XGBoost": (acc_xgb + f1_xgb) / 2,
    "CatBoost": (acc_cat + f1_cat) / 2
}

best_model = max(model_scores, key=model_scores.get)

# Display best model summary
print("\n=== Best Performing Model Summary ===")
if best_model == "Serial RF":
    print(f"Model: {best_model}\nAccuracy: {acc_serial:.4f}\nF1 Score: {f1_serial:.4f}\nTraining Time: {time_serial:.2f}s")
elif best_model == "Parallel RF":
    print(f"Model: {best_model}\nAccuracy: {acc_parallel:.4f}\nF1 Score: {f1_parallel:.4f}\nTraining Time: {time_parallel:.2f}s")
elif best_model == "PyTorch":
    print(f"Model: {best_model}\nAccuracy: {acc_dl:.4f}\nF1 Score: {f1_dl:.4f}\nTraining Time: {dl_time:.2f}s")
elif best_model == "XGBoost":
    print(f"Model: {best_model}\nAccuracy: {acc_xgb:.4f}\nF1 Score: {f1_xgb:.4f}\nTraining Time: {time_xgb:.2f}s")
else:
    print(f"Model: {best_model}\nAccuracy: {acc_cat:.4f}\nF1 Score: {f1_cat:.4f}\nTraining Time: {time_cat:.2f}s")

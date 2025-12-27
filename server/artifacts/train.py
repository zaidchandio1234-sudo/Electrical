import os
# --- THE FIX FOR OMP ERROR #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib

# --- 1. CONFIGURATION ---
# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

SEQ_LENGTH = 24  # Use past 24 hours to predict next hour
BATCH_SIZE = 64
EPOCHS = 50      # Increased epochs since you have a fast GPU!
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# --- 2. DATA LOADING & PREPROCESSING ---
print("Loading data...")
try:
    df = pd.read_csv('electricity_usage.csv')
except FileNotFoundError:
    print("Error: 'electricity_usage.csv' not found. Please make sure the file is in the same folder.")
    exit()

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

# We use Usage + Weather as features
# Ensure these columns exist in your CSV
features = ['Usage_kW', 'temperature', 'pressure', 'windspeed']
data = df[features].values

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler for your future App
joblib.dump(scaler, 'gru_scaler.pkl')

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0] # Predict Usage_kW (column 0)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

print("Creating sequences...")
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split Train/Test (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to Tensors and move to GPU
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# Create DataLoader
train_data = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

# --- 3. DEFINE GRU MODEL ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = GRUModel(input_size=len(features), hidden_size=HIDDEN_SIZE, output_size=1, num_layers=NUM_LAYERS).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. TRAINING LOOP ---
print("Starting training on GPU...")
train_losses = []

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.5f}")

# Save the model
torch.save(model.state_dict(), 'gru_model.pth')
print("Model saved as 'gru_model.pth'")

# --- 5. EVALUATION ---
print("Evaluating...")
model.eval()
with torch.no_grad():
    predictions = model(X_test_t).cpu().numpy()
    y_actual = y_test_t.cpu().numpy()

# Inverse Transform
preds_extended = np.zeros((len(predictions), len(features)))
preds_extended[:, 0] = predictions.flatten()
preds_real = scaler.inverse_transform(preds_extended)[:, 0]

y_extended = np.zeros((len(y_actual), len(features)))
y_extended[:, 0] = y_actual
y_real = scaler.inverse_transform(y_extended)[:, 0]

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(y_real[:168], label='Actual Usage (kW)', color='blue')
plt.plot(preds_real[:168], label='Predicted Usage (GRU)', color='red', linestyle='--')
plt.title('GRU Model Prediction vs Actual (First 7 Days of Test Set)')
plt.xlabel('Hours')
plt.ylabel('Power Usage (kW)')
plt.legend()
plt.savefig('gru_results.png')
print("Graph saved as 'gru_results.png'")
plt.show()

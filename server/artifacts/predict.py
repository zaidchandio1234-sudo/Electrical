import os
# --- THE FIX FOR OMP ERROR #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LENGTH = 24
FUTURE_STEPS = 168  # 7 days * 24 hours
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FEATURES = ['Usage_kW', 'temperature', 'pressure', 'windspeed']

# --- 2. DEFINE THE MODEL ARCHITECTURE ---
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

# --- 3. LOAD RESOURCES ---
print("Loading model and data...")

# Load Scaler
scaler = joblib.load('gru_scaler.pkl')

# Load Model
model = GRUModel(len(FEATURES), HIDDEN_SIZE, 1, NUM_LAYERS).to(device)
model.load_state_dict(torch.load('gru_model.pth'))
model.eval()

# Load Data
try:
    df = pd.read_csv('electricity_usage.csv')
except FileNotFoundError:
    print("Error: 'electricity_usage.csv' not found.")
    exit()

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

# Prepare the last 24 hours of data
last_sequence_data = df[FEATURES].iloc[-SEQ_LENGTH:].values
last_sequence_scaled = scaler.transform(last_sequence_data)

# Convert to Tensor [1, 24, 4]
current_seq = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)

# --- 4. PREDICT FUTURE LOOP ---
print(f"Generating forecast for the next {FUTURE_STEPS} hours (7 Days)...")

future_predictions = []

# Using last known weather for simplicity
last_weather = last_sequence_scaled[-1, 1:] 

with torch.no_grad():
    for _ in range(FUTURE_STEPS):
        # 1. Predict
        pred_scaled = model(current_seq)
        
        # 2. Create new row
        new_row = np.concatenate([pred_scaled.cpu().numpy(), [last_weather]], axis=1)
        new_row_tensor = torch.FloatTensor(new_row).unsqueeze(0).to(device)
        
        # 3. Store
        future_predictions.append(new_row[0])
        
        # 4. Update Sequence (Remove first, Add new)
        current_seq = torch.cat((current_seq[:, 1:, :], new_row_tensor), dim=1)

# --- 5. SAVE RESULTS ---
future_predictions = np.array(future_predictions)
future_predictions_real = scaler.inverse_transform(future_predictions)
predicted_usage = future_predictions_real[:, 0]

# Create DataFrame
last_date = df['datetime'].iloc[-1]
future_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(FUTURE_STEPS)]

forecast_df = pd.DataFrame({
    'datetime': future_dates,
    'Predicted_Usage_kW': predicted_usage
})

forecast_df.to_csv('7_day_forecast.csv', index=False)
print("Success! Saved '7_day_forecast.csv'")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['datetime'], forecast_df['Predicted_Usage_kW'], color='green', label='Future Prediction')
plt.title('7-Day Energy Usage Forecast')
plt.xlabel('Date')
plt.ylabel('Usage (kW)')
plt.grid(True)
plt.legend()
plt.savefig('forecast_plot.png')
print("Graph saved as 'forecast_plot.png'")

# --- 6. PRINT SUMMARY ---
print("\n--- SUMMARY FOR LLM ADVISOR ---")
daily_avg = forecast_df.resample('D', on='datetime')['Predicted_Usage_kW'].mean()
print(daily_avg)

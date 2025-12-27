import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class GRUModel(nn.Module):
    """GRU Model matching your train.py architecture"""
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.load_model()
        
    def load_model(self):
        """Load the trained GRU model and scaler"""
        try:
            # Load scaler with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scaler = joblib.load(self.config.SCALER_PATH)
            
            # Initialize model
            self.model = GRUModel(
                input_size=len(self.config.FEATURES),
                hidden_size=self.config.HIDDEN_SIZE,
                output_size=1,
                num_layers=self.config.NUM_LAYERS
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.device))
            self.model.eval()
            print(f"✓ GRU Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            raise Exception(f"Failed to load model: {str(e)}")
    
    def create_sample_sequence(self):
        """Create a sample sequence when no data is available"""
        # Create sample data based on your model's expected features
        seq_length = self.config.SEQ_LENGTH
        n_features = len(self.config.FEATURES)
        
        # Create realistic sample data
        np.random.seed(42)
        sample_data = np.zeros((seq_length, n_features))
        
        # Usage_kW: 1.0 - 3.0 kW range
        sample_data[:, 0] = 1.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, seq_length)) + np.random.normal(0, 0.1, seq_length)
        
        # Temperature: 20-30°C
        sample_data[:, 1] = 25 + 5 * np.sin(np.linspace(0, 2*np.pi, seq_length))
        
        # Pressure: 1000-1020 hPa
        sample_data[:, 2] = 1010 + 10 * np.sin(np.linspace(0, np.pi, seq_length))
        
        # Windspeed: 0-10 m/s
        sample_data[:, 3] = 5 + 3 * np.sin(np.linspace(0, 3*np.pi, seq_length))
        
        return sample_data
    
    def load_or_create_data(self):
        """Load data from file or create sample data"""
        try:
            if os.path.exists(self.config.DATA_PATH):
                df = pd.read_csv(self.config.DATA_PATH)
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.sort_values('datetime')
                
                # Check if required features exist
                missing_features = [f for f in self.config.FEATURES if f not in df.columns]
                if missing_features:
                    print(f"⚠️  Missing features in dataset: {missing_features}")
                    return self.create_sample_sequence()
                
                # Get last sequence
                last_sequence = df[self.config.FEATURES].iloc[-self.config.SEQ_LENGTH:].values
                return last_sequence
            else:
                print("⚠️  Dataset not found, using sample data")
                return self.create_sample_sequence()
                
        except Exception as e:
            print(f"⚠️  Error loading data: {e}, using sample data")
            return self.create_sample_sequence()
    
    def predict(self, sequence_data):
        """
        Predict energy consumption for a sequence of data
        sequence_data: numpy array of shape (n_samples, SEQ_LENGTH, n_features)
        """
        if self.model is None or self.scaler is None:
            raise Exception("Model not loaded")
        
        with torch.no_grad():
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence_data).to(self.device)
            
            # Predict
            predictions_scaled = self.model(sequence_tensor)
            
            # Inverse transform
            predictions_scaled_np = predictions_scaled.cpu().numpy()
            predictions_full = np.zeros((len(predictions_scaled_np), len(self.config.FEATURES)))
            predictions_full[:, 0] = predictions_scaled_np.flatten()
            
            # Fill other features with their last known values
            if sequence_data.shape[0] > 0:
                last_weather = sequence_data[:, -1, 1:]
                predictions_full[:, 1:] = last_weather
            
            predictions_real = self.scaler.inverse_transform(predictions_full)[:, 0]
            
            return predictions_real
    
    def generate_7day_forecast(self):
        """Generate 7-day forecast (168 hours)"""
        try:
            # Load or create data
            last_sequence_data = self.load_or_create_data()
            
            # Scale the data
            last_sequence_scaled = self.scaler.transform(last_sequence_data)
            
            # Convert to Tensor [1, 24, 4]
            current_seq = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
            
            # Generate 7-day forecast (168 hours)
            future_predictions = []
            last_weather = last_sequence_scaled[-1, 1:]
            
            print(f"🔮 Generating 168-hour forecast...")
            with torch.no_grad():
                for i in range(168):  # 7 days * 24 hours
                    # Predict
                    pred_scaled = self.model(current_seq)
                    
                    # Create new row
                    new_row = np.concatenate([pred_scaled.cpu().numpy(), [last_weather]], axis=1)
                    new_row_tensor = torch.FloatTensor(new_row).unsqueeze(0).to(self.device)
                    
                    # Store
                    future_predictions.append(new_row[0])
                    
                    # Update Sequence (Remove first, Add new)
                    current_seq = torch.cat((current_seq[:, 1:, :], new_row_tensor), dim=1)
                    
                    # Show progress
                    if (i + 1) % 24 == 0:
                        print(f"   Day {(i + 1) // 24}/7 complete")
            
            # Convert to real values
            future_predictions = np.array(future_predictions)
            future_predictions_real = self.scaler.inverse_transform(future_predictions)
            predicted_usage = future_predictions_real[:, 0]
            
            # Create timestamps
            last_date = datetime.now() - timedelta(hours=24)
            future_dates = [last_date + timedelta(hours=i+1) for i in range(168)]
            
            # Prepare forecast data
            forecast_data = []
            for i in range(168):
                forecast_data.append({
                    'timestamp': future_dates[i].isoformat(),
                    'hour': future_dates[i].hour,
                    'predicted_kw': float(predicted_usage[i]),
                    'date': future_dates[i].date().isoformat(),
                    'day_name': future_dates[i].strftime('%A')
                })
            
            print(f"✅ Forecast generated: {len(forecast_data)} hours")
            return forecast_data
            
        except Exception as e:
            print(f"❌ Forecast generation error: {e}")
            raise Exception(f"Failed to generate forecast: {str(e)}")
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "model_type": "GRU",
            "input_features": self.config.FEATURES,
            "target": self.config.TARGET,
            "sequence_length": self.config.SEQ_LENGTH,
            "hidden_size": self.config.HIDDEN_SIZE,
            "num_layers": self.config.NUM_LAYERS,
            "device": str(self.device),
            "data_available": os.path.exists(self.config.DATA_PATH)
        }
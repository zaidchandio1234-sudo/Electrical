import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    """Configuration settings for the application"""
    
    def __init__(self):
        # Get the project root directory
        try:
            self.BASE_DIR = Path(__file__).parent.parent
        except:
            self.BASE_DIR = Path.cwd()
        
        # API Configuration
        self.API_HOST = "0.0.0.0"
        self.API_PORT = 8000
        self.DEBUG = True
        
        # Model Configuration
        self.MODEL_PATH = self.get_path("server/artifacts/gru_model.pth")
        self.SCALER_PATH = self.get_path("server/artifacts/gru_scaler.pkl")
        
        # Dataset Configuration
        self.DATA_PATH = self.get_path("dataset/electricity_usage.csv")
        self.FORECAST_DATA_PATH = self.get_path("dataset/7_day_forecast.csv")
        
        # LLM Configuration
        self.LLM_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
        self.LLM_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        self.LLM_MAX_LENGTH = 500
        
        # Model Parameters
        self.SEQ_LENGTH = 24
        self.HIDDEN_SIZE = 64
        self.NUM_LAYERS = 2
        self.FEATURES = ['Usage_kW', 'temperature', 'pressure', 'windspeed']
        self.TARGET = 'Usage_kW'
        
        # Authentication
        self.SECRET_KEY = os.getenv("SECRET_KEY", "energywise-secret-key-2024")
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 1440
        
        # Database
        self.DATABASE_URL = "sqlite:///./energywise.db"
        
        # Verify paths
        self.verify_paths()
    
    def get_path(self, relative_path):
        """Get absolute path, trying multiple locations"""
        # Try in current directory
        current_dir = Path.cwd()
        path = current_dir / relative_path
        
        if path.exists():
            return str(path)
        
        # Try in server directory
        server_dir = current_dir / "server"
        path = server_dir / relative_path
        
        if path.exists():
            return str(path)
        
        # Try in parent directory
        parent_dir = current_dir.parent
        path = parent_dir / relative_path
        
        if path.exists():
            return str(path)
        
        # Return relative path as fallback
        return relative_path
    
    def verify_paths(self):
        """Verify that required files exist"""
        print("🔍 Verifying paths...")
        
        paths_to_check = [
            (self.MODEL_PATH, "Model file"),
            (self.SCALER_PATH, "Scaler file"),
            (self.DATA_PATH, "Dataset file")
        ]
        
        for path, description in paths_to_check:
            if os.path.exists(path):
                print(f"  ✓ {description}: {path}")
            else:
                print(f"  ⚠️  {description} not found: {path}")
                print(f"     Will use sample data")
    
    def create_sample_dataset(self):
        """Create a sample dataset if it doesn't exist"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        if not os.path.exists(self.DATA_PATH):
            print(f"📝 Creating sample dataset at {self.DATA_PATH}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.DATA_PATH), exist_ok=True)
            
            # Generate sample data
            hours = 24 * 30  # 30 days
            dates = [datetime.now() - timedelta(hours=i) for i in range(hours)][::-1]
            
            df = pd.DataFrame({
                'datetime': dates,
                'Usage_kW': 1.5 + 0.5 * np.sin(np.arange(hours) * 2 * np.pi / 24) + np.random.normal(0, 0.1, hours),
                'temperature': 25 + 5 * np.sin(np.arange(hours) * 2 * np.pi / (24*7)),
                'pressure': 1010 + 5 * np.random.randn(hours),
                'windspeed': 5 + 2 * np.sin(np.arange(hours) * 2 * np.pi / 12)
            })
            
            df.to_csv(self.DATA_PATH, index=False)
            print(f"  ✓ Created dataset with {len(df)} rows")
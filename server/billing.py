from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel
import sqlite3
import json

class BillingPeriod(BaseModel):
    period_start: str
    period_end: str
    total_kwh: float
    total_cost: float
    rate_per_kwh: float = 0.15  # Default rate: $0.15/kWh
    status: str = "pending"
    breakdown: Optional[Dict] = None

class BillingService:
    def __init__(self, config):
        self.config = config
        self.base_rate = 0.15  # $0.15 per kWh
        self.init_database()
    
    def init_database(self):
        """Initialize billing database"""
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        
        # Create billing_history table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS billing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                billing_date DATE,
                period_start DATE,
                period_end DATE,
                total_kwh REAL,
                peak_kwh REAL,
                offpeak_kwh REAL,
                total_cost REAL,
                rate_per_kwh REAL DEFAULT 0.15,
                status TEXT DEFAULT 'paid',
                payment_method TEXT,
                paid_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_bill(self, user_id: int, usage_data: List[Dict]) -> Dict:
        """Calculate bill based on usage data"""
        total_kwh = sum(hour['usage_kw'] for hour in usage_data) / 1000  # Convert to kWh
        peak_hours = [hour for hour in usage_data if 14 <= hour['hour'] <= 20]
        offpeak_hours = [hour for hour in usage_data if hour not in peak_hours]
        
        peak_kwh = sum(hour['usage_kw'] for hour in peak_hours) / 1000
        offpeak_kwh = sum(hour['usage_kw'] for hour in offpeak_hours) / 1000
        
        # Time-of-use pricing
        peak_rate = self.base_rate * 1.5  # 50% higher during peak
        offpeak_rate = self.base_rate * 0.7  # 30% lower during off-peak
        
        peak_cost = peak_kwh * peak_rate
        offpeak_cost = offpeak_kwh * offpeak_rate
        total_cost = peak_cost + offpeak_cost
        
        # Apply discounts for efficient usage
        if total_kwh < 300:  # Less than 300 kWh
            discount = total_cost * 0.05  # 5% discount
            total_cost -= discount
        
        return {
            "total_kwh": round(total_kwh, 2),
            "peak_kwh": round(peak_kwh, 2),
            "offpeak_kwh": round(offpeak_kwh, 2),
            "peak_cost": round(peak_cost, 2),
            "offpeak_cost": round(offpeak_cost, 2),
            "total_cost": round(total_cost, 2),
            "average_daily": round(total_kwh / 30, 2),
            "carbon_footprint": round(total_kwh * 0.92, 2),  # kg CO2
            "breakdown": {
                "peak_rate": peak_rate,
                "offpeak_rate": offpeak_rate,
                "discount_applied": total_kwh < 300
            }
        }
    
    def generate_monthly_bill(self, user_id: int, month: int, year: int) -> Dict:
        """Generate monthly bill"""
        # Simulate usage data for the month
        import random
        import numpy as np
        
        days_in_month = 30
        hourly_usage = []
        
        for day in range(days_in_month):
            for hour in range(24):
                # Base pattern + randomness
                base_usage = 1.5 + 0.5 * np.sin(hour/24 * 2*np.pi)
                if 14 <= hour <= 20:  # Peak hours
                    base_usage *= 1.3
                usage = base_usage + random.uniform(-0.2, 0.2)
                
                hourly_usage.append({
                    "hour": hour,
                    "usage_kw": max(0.5, usage),
                    "date": f"{year}-{month:02d}-{day+1:02d}"
                })
        
        bill_details = self.calculate_bill(user_id, hourly_usage)
        
        # Save to database
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO billing_history 
            (user_id, billing_date, period_start, period_end, total_kwh, peak_kwh, 
             offpeak_kwh, total_cost, rate_per_kwh, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            f"{year}-{month:02d}-01",
            f"{year}-{month-1:02d}-15" if month > 1 else f"{year-1}-12-15",
            f"{year}-{month:02d}-14",
            bill_details["total_kwh"],
            bill_details["peak_kwh"],
            bill_details["offpeak_kwh"],
            bill_details["total_cost"],
            self.base_rate,
            "pending"
        ))
        
        bill_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        bill_details["bill_id"] = bill_id
        bill_details["billing_period"] = f"{year}-{month:02d}"
        bill_details["due_date"] = f"{year}-{month:02d}-28"
        
        return bill_details
    
    def get_billing_history(self, user_id: int, limit: int = 12) -> List[Dict]:
        """Get billing history for user"""
        conn = sqlite3.connect("users.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM billing_history 
            WHERE user_id = ? 
            ORDER BY billing_date DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        bills = []
        for row in rows:
            bills.append(dict(row))
        
        return bills
    
    def get_current_bill(self, user_id: int) -> Optional[Dict]:
        """Get current/pending bill"""
        conn = sqlite3.connect("users.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM billing_history 
            WHERE user_id = ? AND status = 'pending'
            ORDER BY billing_date DESC 
            LIMIT 1
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def process_payment(self, user_id: int, bill_id: int, payment_method: str = "credit_card") -> bool:
        """Process payment for a bill"""
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE billing_history 
            SET status = 'paid', 
                payment_method = ?,
                paid_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
        ''', (payment_method, bill_id, user_id))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
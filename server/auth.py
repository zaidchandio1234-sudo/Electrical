from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional
import jwt # type: ignore
import bcrypt # type: ignore
from pydantic import BaseModel
import sqlite3
import os

# Security
security = HTTPBearer()

# Models
class User(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    address: Optional[str] = None
    meter_id: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    email: str

class AuthService:
    def __init__(self, config):
        self.config = config
        self.secret_key = config.SECRET_KEY
        self.algorithm = config.ALGORITHM
        self.access_token_expire_minutes = config.ACCESS_TOKEN_EXPIRE_MINUTES
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for users"""
        db_path = "users.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                address TEXT,
                meter_id TEXT,
                account_type TEXT DEFAULT 'residential',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Create user_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER,
                token TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create billing table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS billing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                period_start DATE,
                period_end DATE,
                total_kwh REAL,
                total_cost REAL,
                status TEXT DEFAULT 'pending',
                paid_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create demo users if table is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            self.create_demo_users(cursor)
        
        conn.commit()
        conn.close()
    
    def create_demo_users(self, cursor):
        """Create demo users for testing"""
        demo_users = [
            {
                'email': 'customer@energywise.com',
                'password': 'EnergyWise2024!',
                'name': 'John Doe',
                'address': '123 Green Street, Eco City',
                'meter_id': 'MTR-001-2024',
                'account_type': 'premium'
            },
            {
                'email': 'business@energywise.com',
                'password': 'Business2024!',
                'name': 'Jane Smith',
                'address': '456 Business Ave, Tech Park',
                'meter_id': 'MTR-002-2024',
                'account_type': 'commercial'
            }
        ]
        
        for user in demo_users:
            password_hash = bcrypt.hashpw(user['password'].encode('utf-8'), bcrypt.gensalt())
            cursor.execute('''
                INSERT INTO users (email, password_hash, name, address, meter_id, account_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user['email'], password_hash, user['name'], user['address'], 
                  user['meter_id'], user['account_type']))
    
    def hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def authenticate_user(self, email: str, password: str):
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, email, password_hash, name FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return None
        
        user_id, user_email, password_hash, name = user
        
        if not self.verify_password(password, password_hash):
            return None
        
        # Update last login
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        
        return {
            "user_id": user_id,
            "email": user_email,
            "name": name
        }
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        payload = self.verify_token(token)
        
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, name, account_type FROM users WHERE email = ?", (payload.get("sub"),))
        user = cursor.fetchone()
        conn.close()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "user_id": user[0],
            "email": user[1],
            "name": user[2],
            "account_type": user[3]
        }
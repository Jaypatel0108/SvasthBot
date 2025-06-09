import sqlite3
import bcrypt
import jwt
import os
import datetime
import re
import logging
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# 🔹 Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 🔹 Set up database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "health_chatbot.db")

# 🔹 Load secret key from environment variable
SECRET_KEY = os.getenv("SECRET_KEY", "your_fallback_secret_key_here")

def get_user_id_from_session():
    if st.session_state.auth_token:
        # Assuming you have a way to decode the token and get the user ID
        # This is a placeholder; implement your logic to extract user ID from the token
        return decode_token_and_get_user_id(st.session_state.auth_token)  # Implement this function as needed
    return None

def decode_token_and_get_user_id(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])  # Use your secret key
        return payload.get("user_id")  # Adjust based on your token structure
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    
# ✅ Initialize the database connection with context manager
def open_db_connection():
    try:
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")  # Ensure foreign keys are enforced
        return conn
    except sqlite3.Error as e:
        logger.error(f"❌ Database connection error: {e}")
        return None

# ✅ Secure Password Hashing
def hash_password(password):
    # Increased work factor for bcrypt
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(12)).decode("utf-8")

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

# ✅ Check password security
def is_secure_password(password: str) -> bool:
    return bool(re.match(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$", password))

# ✅ Register a new user
def register_user(username, password):
    username = username.strip()
    password = password.strip()

    if not username or not password:
        return {"success": False, "message": "Username and password must not be empty or whitespace-only."}

    if not is_secure_password(password):
        return {"success": False, "message": "Password must be at least 8 characters long, include a number and a special character."}

    hashed_password = hash_password(password)  # Hash the password
    with open_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return {"success": True, "message": "User registered successfully."}

# ✅ Authenticate user and generate JWT token
def authenticate_user(username, password):
    username = username.strip()
    password = password.strip()

    if not username or not password:
        return {"success": False, "message": "Username and password must not be empty or whitespace-only."}

    with open_db_connection() as conn:
        if not conn:
            return {"success": False, "message": "Database connection failed."}

        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username=?", (username,))
        user = cursor.fetchone()

        if user:
            user_id, stored_hashed_password = user
            if verify_password(password, stored_hashed_password):
                try:
                    token = jwt.encode(
                        {
                            "user_id": user_id,
                            "username": username,
                            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
                        },
                        SECRET_KEY,
                        algorithm="HS256"
                    )
                    return {"success": True, "token": token}
                except jwt.PyJWTError as e:
                    logger.error(f"❌ JWT Error: {e}")
                    return {"success": False, "message": "Token generation failed."}

    return {"success": False, "message": "Invalid username or password"}

# ✅ Function to Save Chat History
def save_chat(user_id, query, response):
    try:
        with open_db_connection() as conn:
            if not conn:
                return False

            conn.execute(
                "INSERT INTO chat_history (user_id, query, response) VALUES (?, ?, ?)", 
                (user_id, query, response)
            )
            conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"❌ Database Error: {e}")
        return False

# ✅ Function to Retrieve Chat History
def get_chat_history(user_id):
    with open_db_connection() as conn:
        if not conn:
            return []

        cursor = conn.cursor()
        cursor.execute("SELECT query, response, timestamp FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
        return [{"query": q, "response": r, "timestamp": t} for q, r, t in cursor.fetchall()]

# ✅ Function to Clear Chat History
def clear_chat_history(user_id):
    with open_db_connection() as conn:
        if not conn:
            return False

        conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.commit()
        return True

# ✅ Initialize database tables
def init_db():
    with open_db_connection() as conn:
        if not conn:
            logger.error("❌ Database connection failed during initialization.")
            return

        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                username TEXT UNIQUE NOT NULL, 
                password TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                user_id INTEGER NOT NULL, 
                query TEXT NOT NULL, 
                response TEXT NOT NULL, 
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
        logger.info("✅ Database initialized successfully.")

# ✅ Run database initialization
init_db()

def store_chat_history():
    try:
        with open_db_connection() as conn:
            if not conn:
                return False

            user_id = get_user_id_from_session()  # Get the user ID from the session
            if user_id is None:
                logger.error("User ID not found in session.")
                return False

            # Store the current messages in the database
            for message in st.session_state.messages:
                if message['role'] == 'user':
                    query = message['content']
                    response = None  # No response yet for user messages
                elif message['role'] == 'assistant':
                    query = None  # No query for assistant messages
                    response = message['content']
                
                # Insert into the database only if we have a query or response
                if query or response:
                    conn.execute(
                        "INSERT INTO chat_history (user_id, query, response) VALUES (?, ?, ?)", 
                        (user_id, query, response)
                    )
            conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"❌ Database Error: {e}")
        return False

class ChatHistoryData(BaseModel):
    messages: List[Dict[str, str]]
    timestamp: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "What are symptoms of the flu?"},
                    {"role": "assistant", "content": "Flu symptoms include fever, cough, sore throat..."}
                ],
                "timestamp": "2023-06-15 14:30:00"
            }
        }

def get_user_by_username(username):
    with open_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()  # Returns user data if found

def remove_all_users():
    try:
        with open_db_connection() as conn:
            if not conn:
                return False
            
            conn.execute("DELETE FROM users")  # This will remove all users
            conn.commit()
        logger.info("✅ All users have been removed from the database.")
        return True
    except sqlite3.Error as e:
        logger.error(f"❌ Database Error: {e}")
        return False

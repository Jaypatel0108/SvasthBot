from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
import sqlite3
import jwt, sys, os
import datetime
import logging
import bcrypt
from typing import List, Dict, Optional
from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load AI Models Locally
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Models.medbert import medbert  # Ensure correct import path
from Models.biogpt import biogpt  # Ensure correct import path

DATABASE = "health_chatbot.db"
SECRET_KEY = os.getenv("SECRET_KEY")
if SECRET_KEY is None:
    raise RuntimeError("SECRET_KEY environment variable is not set.")
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ✅ Securely initialize SQLite DB
def get_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    return conn, cursor

# ✅ Create tables if they don't exist
def init_db():
    conn, cursor = get_db()
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
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ✅ Token generation function
def create_access_token(username: str) -> str:
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    payload = {"sub": username, "exp": expiration}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# ✅ Token validation function
def verify_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ✅ Password verification function
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# ✅ Register new user
@app.post("/auth/register")
async def register_user(user: LoginRequest):
    username = user.username
    password = user.password

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode("utf-8")

    conn, cursor = get_db()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        conn.close()

# ✅ Login and return JWT token
@app.post("/auth/login")
async def login_user(request: LoginRequest):
    """Authenticate user and return JWT token"""
    logger.info(f"Login request received: {request}")

    # ✅ Convert Pydantic object to dictionary
    user_data = request.dict()

    conn, cursor = get_db()

    # ✅ Fetch user from database
    cursor.execute("SELECT * FROM users WHERE username = ?", (user_data["username"],))
    user = cursor.fetchone()

    if not user or not verify_password(user_data["password"], user[2]):  # user[2] is hashed_password
        logger.error("Invalid username or password")
        raise HTTPException(status_code=422, detail="Invalid username or password")

    # ✅ Generate JWT token
    access_token = create_access_token(username=user_data["username"])
    conn.close()

    return {"token": access_token}

# ✅ Store user queries and bot responses
@app.post("/healthbot")
async def healthbot_response(request: Request, data: Dict[str, str]):
    # Check for a guest token by inspecting the request headers
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer guest"):
        username = "Guest"
    else:
        # Verify token for registered users
        token = await oauth2_scheme(request)
        username = verify_token(token)

    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    # Process input with MedBERT
    classification_result = medbert.classify_text(query)

    # Generate response using BioGPT
    bot_response = biogpt.generate_response(query)

    # Store chat history for registered users
    if username != "Guest":
        conn, cursor = get_db()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user:
            user_id = user[0]
            cursor.execute("INSERT INTO chat_history (user_id, query, response) VALUES (?, ?, ?)", 
                           (user_id, query, bot_response))
            conn.commit()
        conn.close()

    return {"classification": classification_result.tolist(), "response": bot_response}

# ✅ Retrieve all chat history for the logged-in user
@app.get("/chat_history/")
async def get_chat_history(token: str = Depends(oauth2_scheme)):
    username = verify_token(token)

    conn, cursor = get_db()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    user_id = user[0]
    cursor.execute("SELECT id, query, response, timestamp FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    chat_records = cursor.fetchall()
    conn.close()

    history_by_date = {}
    for record in chat_records:
        record_id, query, response, timestamp = record
        date = timestamp.split(" ")[0]
        if date not in history_by_date:
            history_by_date[date] = []
        history_by_date[date].append({
            "id": record_id,
            "query": query,
            "response": response,
            "timestamp": timestamp
        })

    return {"history": history_by_date}

# ✅ Retrieve a specific chat history
@app.get("/chat_history/{chat_id}")
async def get_chat_detail(chat_id: int, token: str = Depends(oauth2_scheme)):
    username = verify_token(token)

    conn, cursor = get_db()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    user_id = user[0]
    cursor.execute("SELECT query, response, timestamp FROM chat_history WHERE id = ? AND user_id = ?", (chat_id, user_id))
    chat_record = cursor.fetchone()
    conn.close()

    if not chat_record:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {
        "query": chat_record[0],
        "response": chat_record[1],
        "timestamp": chat_record[2]
    }

# ✅ Delete chat history
@app.delete("/chat_history/delete/")
async def delete_chat_history(token: str = Depends(oauth2_scheme)):
    username = verify_token(token)

    conn, cursor = get_db()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    user_id = user[0]
    cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    return {"message": "Chat history deleted successfully"}
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
import sqlite3
from datetime import datetime, timedelta
import logging
import bcrypt
from typing import Dict, Optional, List, Any, Tuple
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
import os, sys
from database import verify_password, hash_password, open_db_connection
from cryptography.fernet import Fernet
import asyncio
from utils import translator,translate_to_english, translate_from_english, performance_monitor, monitor_performance
from googletrans import Translator  # Add missing import

# Import AI models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Models.medbert import medbert
from Models.biogpt import biogpt

# App Configuration
app = FastAPI(title="Health Chatbot API", 
              description="API for health-related chatbot interactions",
              version="1.0.0")

# CORS configuration with more specific origins for production
app.add_middleware(
    CORSMiddleware, 
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True, 
    allow_methods=["GET", "POST", "PUT", "DELETE"], 
    allow_headers=["Authorization", "Content-Type"]
)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the correct URL and endpoint for the Gemini API
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Constants
DATABASE = os.path.join(os.path.dirname(__file__), 'health_chatbot.db')
SECRET_KEY = os.getenv("SECRET_KEY", "your_default_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Encryption Setup with better error handling
try:
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
    if not ENCRYPTION_KEY:
        ENCRYPTION_KEY = Fernet.generate_key()
        logger.warning("ENCRYPTION_KEY not found in environment variables. Generated a new one.")
    cipher = Fernet(ENCRYPTION_KEY)
except Exception as e:
    logger.error(f"Error setting up encryption: {e}")
    # Generate a key but also set a flag to warn about this in API responses
    ENCRYPTION_KEY = Fernet.generate_key()
    cipher = Fernet(ENCRYPTION_KEY)
    ENCRYPTION_SETUP_FAILED = True
else:
    ENCRYPTION_SETUP_FAILED = False

# Translator
translator = Translator()

# Models with improved validation
class ChatHistoryData(BaseModel):
    messages: List[Dict[str, str]]
    timestamp: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "What are symptoms of the flu?"},
                    {"role": "assistant", "content": "Flu symptoms include fever, cough, sore throat..."}
                ],
                "timestamp": "2023-06-15 14:30:00"
            }
        }

class LoginRequest(BaseModel):
    username: str
    password: str
    
    class Config:
        schema_extra = {
            "example": {
                "username": "user123",
                "password": "securepassword123!"
            }
        }

class ChatHistoryRequest(BaseModel):
    messages: List[Dict[str, str]]
    timestamp: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "What are symptoms of the flu?"},
                    {"role": "assistant", "content": "Flu symptoms include fever, cough, sore throat..."}
                ],
                "timestamp": "2023-06-15T14:30:00"
            }
        }

class ConversationMetadata(BaseModel):
    last_active: datetime = Field(default_factory=datetime.utcnow)
    topic: Optional[str] = None
    message_count: int = 0
    language: str = "en"

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.metadata: ConversationMetadata = ConversationMetadata()
        self.last_context: Optional[str] = None

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.metadata.message_count += 1
        self.metadata.last_active = datetime.utcnow()
        
        # Update conversation topic based on user messages
        if role == "user" and not self.metadata.topic:
            self.metadata.topic = content[:50] + "..." if len(content) > 50 else content

    def get_history(self) -> List[str]:
        return [f"{'Bot' if msg['role'] == 'assistant' else 'User'}: {msg['content']}" 
                for msg in self.messages[-6:]]

    def clear(self):
        self.messages.clear()
        self.metadata = ConversationMetadata()
        self.last_context = None

# Add this to store active conversations
active_conversations: Dict[int, Conversation] = {}

# Database Functions
def get_db():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            user_id INTEGER NOT NULL, 
            query TEXT NOT NULL, 
            response TEXT NOT NULL, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)
init_db()

# Token Functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return jwt.encode({**data, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Sanitize and Validate User Input
def sanitize_credentials(username: str, password: str):
    username = username.strip()
    password = password.strip()
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password must not be empty.")

    return username, password

# User Authentication
@app.post("/auth/register")
async def register_user(user: LoginRequest):
    username, password = sanitize_credentials(user.username, user.password)
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8")
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/auth/login")
async def login_user(request: LoginRequest):
    username, password = sanitize_credentials(request.username, request.password)
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if not user or not verify_password(password, user[2]):
        logger.warning(f"Failed login attempt for user: {username}")
        raise HTTPException(status_code=422, detail="Invalid username or password")

    return {"token": create_access_token(data={"sub": username})}

# Add conversation cleanup task
async def cleanup_inactive_conversations():
    while True:
        try:
            current_time = datetime.utcnow()
            inactive_threshold = current_time - timedelta(hours=24)
            
            async with asyncio.Lock():  # Thread-safe cleanup
                to_remove = [chat_id for chat_id, conv in active_conversations.items() 
                           if conv.metadata.last_active < inactive_threshold]
                for chat_id in to_remove:
                    del active_conversations[chat_id]
                    
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute on error

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_inactive_conversations())

# Enhanced healthbot endpoint
@app.post("/healthbot")
@monitor_performance
async def healthbot_response(request: Request, data: Dict[str, str]):
    """
    Generate a response from the health chatbot.
    
    Args:
        request: The FastAPI request object
        data: The request data containing the user's query
        
    Returns:
        Dict: The chatbot response
    """
    try:
        query = data.get("query", "").strip()
        chat_id = data.get("chat_id")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        # Get or create conversation based on chat_id
        conversation = await get_or_create_conversation(chat_id)
        
        # Add user message to conversation
        conversation.add_message("user", query)
        
        # Generate response with retry logic
        try:
            response = await generate_responses_with_retry(query, conversation)
            
            # Add assistant response to conversation
            conversation.add_message("assistant", response)
            
            return {"response": response}
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in healthbot_response: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

async def get_or_create_conversation(chat_id: Optional[str] = None):
    try:
        if chat_id:
            conversation = active_conversations.get(int(chat_id))
            if not conversation:
                conversation = Conversation()
                active_conversations[int(chat_id)] = conversation
        else:
            conversation = Conversation()
            chat_id = len(active_conversations) + 1
            active_conversations[chat_id] = conversation
        return conversation
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat ID format")

@monitor_performance
async def generate_responses_with_retry(query: str, conversation, max_retries: int = 3) -> str:
    """
    Generate responses with retry logic for handling transient errors.
    
    Args:
        query: The user's query
        conversation: The conversation object containing history
        max_retries: Maximum number of retry attempts
        
    Returns:
        str: The generated response
    """
    retry_delay = 1  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            response = biogpt.generate_response_with_gemini(query, conversation.get_history())
            if response:
                return response
            else:
                logger.warning(f"Empty response received (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise Exception("Received empty response after all retry attempts")
        except Exception as e:
            logger.error(f"Error generating response (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                logger.error("All retry attempts failed")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to generate response after {max_retries} attempts: {str(e)}"
                )

# Retrieve Chat History
@app.get("/chat_history/")
async def get_chat_history(token: str = Depends(oauth2_scheme)):
    username = verify_token(token)
    try:
        with get_db() as conn:
            user_id = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if not user_id:
                raise HTTPException(status_code=401, detail="User not found")

            chat_records = conn.execute("SELECT id, query, response, timestamp FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC", (user_id[0],)).fetchall()
            logging.info(f"Retrieved chat records for user {username}: {chat_records}")

        if not chat_records:
            # Return empty history instead of raising an error
            return {"history": {}}

        # Return chat history grouped by date
        return {"history": {record[3].split(" ")[0]: [{"id": record[0], "query": record[1], "response": record[2], "timestamp": record[3]}] for record in chat_records}}
    except Exception as e:
        logging.error(f"Error retrieving chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving chat history")

# Retrieve Specific Chat
@app.get("/chat_history/{chat_id}")
async def get_chat_detail(chat_id: int, token: str = Depends(oauth2_scheme)):
    username = verify_token(token)
    try:
        with get_db() as conn:
            user_id = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            chat_record = conn.execute('''
                SELECT query, response, timestamp 
                FROM chat_history 
                WHERE id = ? AND user_id = ?''', (chat_id, user_id[0])).fetchone()

        if not chat_record:
            raise HTTPException(status_code=404, detail="Chat not found")

        # Initialize messages as an empty list
        messages = []

        # Create interleaved message sequence from query and response
        user_queries = [q.strip() for q in chat_record[0].split("\n") if q.strip()]
        assistant_responses = [r.strip() for r in chat_record[1].split("\n") if r.strip()]
        
        # Ensure we have equal number of queries and responses by padding if necessary
        max_len = max(len(user_queries), len(assistant_responses))
        user_queries = user_queries + [""] * (max_len - len(user_queries))
        assistant_responses = assistant_responses + [""] * (max_len - len(assistant_responses))
        
        # Interleave messages
        for i in range(max_len):
            if user_queries[i]:
                messages.append({"role": "user", "content": user_queries[i]})
            if assistant_responses[i]:
                messages.append({"role": "assistant", "content": assistant_responses[i]})

        logging.info(f"Processed messages for chat_id {chat_id}: {messages}")

        return {
            "query": chat_record[0],
            "response": chat_record[1],
            "timestamp": chat_record[2],
            "messages": messages
        }
    except Exception as e:
        logging.error(f"Error retrieving specific chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving specific chat")


# Delete Chat History
@app.delete("/chat_history/delete/")
async def delete_chat_history(token: str = Depends(oauth2_scheme)):
    username = verify_token(token)
    with get_db() as conn:
        user_id = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id[0],))

    return {"message": "Chat history deleted successfully"}

# Save Chat History
@app.post("/chat_history/")
async def save_chat_history(request: Request, data: ChatHistoryData, token: str = Depends(oauth2_scheme)):
    username = verify_token(token)
    try:
        with get_db() as conn:
            user_id = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if not user_id:
                raise HTTPException(status_code=401, detail="User not found")

            timestamp = data.timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            # Log all messages for debugging
            logging.info(f"Received messages for saving: {data.messages}")
            
            # Process messages and combine them
            user_messages = [msg["content"] for msg in data.messages if msg["role"] == "user"]
            assistant_messages = [msg["content"] for msg in data.messages if msg["role"] == "assistant"]
            
            logging.info(f"User messages: {user_messages}")
            logging.info(f"Assistant messages: {assistant_messages}")

            # Join messages with newlines
            combined_query = "\n".join(user_messages)
            combined_response = "\n".join(assistant_messages)

            # Store the chat history
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_history (user_id, query, response, timestamp) 
                VALUES (?, ?, ?, ?)''', 
                (user_id[0], combined_query, combined_response, timestamp))
            
            # Get the ID of the newly inserted chat
            new_chat_id = cursor.lastrowid
            conn.commit()

            logging.info(f"Stored chat history for user {username}: Query: {combined_query}, Response: {combined_response}, Chat ID: {new_chat_id}")
            return {"message": "Chat history saved successfully", "chat_id": new_chat_id}
    except Exception as e:
        logging.error(f"Error saving chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error saving chat history")

# Update Chat History
@app.put("/chat_history/{chat_id}")
async def update_chat_history(chat_id: int, data: ChatHistoryRequest, token: str = Depends(oauth2_scheme)):
    username = verify_token(token)
    try:
        with get_db() as conn:
            user_id = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if not user_id:
                raise HTTPException(status_code=401, detail="User not found")

            existing_chat = conn.execute('''
                SELECT id FROM chat_history 
                WHERE id = ? AND user_id = ?''', (chat_id, user_id[0])).fetchone()
            
            if not existing_chat:
                raise HTTPException(status_code=404, detail="Chat not found")

            # Process messages and combine them
            user_messages = [msg["content"] for msg in data.messages if msg["role"] == "user"]
            assistant_messages = [msg["content"] for msg in data.messages if msg["role"] == "assistant"]

            # Join messages with newlines
            combined_query = "\n".join(user_messages)
            combined_response = "\n".join(assistant_messages)

            # Update the chat history
            conn.execute('''
                UPDATE chat_history 
                SET query = ?, response = ?, timestamp = ? 
                WHERE id = ? AND user_id = ?''', 
                (combined_query, combined_response, data.timestamp, chat_id, user_id[0]))

            logging.info(f"Updated chat history for chat_id {chat_id} with new query and response")
            return {"message": "Chat history updated successfully"}
    except Exception as e:
        logging.error(f"Error updating chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating chat history")

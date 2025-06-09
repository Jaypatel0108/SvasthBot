import streamlit as st
import requests
from bs4 import BeautifulSoup
import pytz
import os, io, re
import sys
import datetime, time
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import pyaudio
import logging
import markdown
import json
import jwt  # Make sure to import the jwt library

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path to import utils and models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Models.biogpt import biogpt
from Backend.utils import translate_to_english, translate_from_english, clean_text

# Load environment variables from a .env file
load_dotenv()

# Initialize session state
def initialize_session_state():
    defaults = {
        "auth_token": None,
        "username": None,
        "messages": [],
        "chat_sessions": [],
        "selected_chat": None,
        "is_guest": True,
        "clear_chat_confirm": False,
        "clear_history_confirm": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

initialize_session_state()

# Use the clean_text function from utils.py instead of defining our own sanitize_text
def sanitize_text(text):
    return clean_text(text)

# Function to create audio in-memory
def create_audio(text):
    if not text:
        logging.warning("No text provided for audio generation.")
        return None
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak into the microphone")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized speech: {text}")
            return text
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError) as e:
            st.error(f"Error recognizing speech: {e}")
            return None

# FastAPI backend URLs
API_BASE_URL = "http://localhost:8000"  # Replace after deploying Backend
API_AUTH_URL = f"{API_BASE_URL}/auth"
API_CHAT_URL = f"{API_BASE_URL}/healthbot"
API_CHAT_HISTORY_URL = f"{API_BASE_URL}/chat_history"

# Set page config
st.set_page_config(page_title="SvasthBot - AI Health Assistant", page_icon="🩺", layout="wide")

# Disclaimer message
st.markdown(
    """
    **Disclaimer:** Please do not enter any sensitive or personal information into this application.
    The information provided here is for informational purposes only and is not a substitute for professional medical advice.
    """,
    unsafe_allow_html=True
)

# Function to send API requests
def api_request(endpoint, method="GET", data=None):
    headers = {"Authorization": f"Bearer {st.session_state.auth_token}"} if st.session_state.auth_token else {}
    url = f"{API_BASE_URL}{endpoint}"

    try:
        request_method = {
            "POST": requests.post,
            "DELETE": requests.delete,
            "GET": requests.get,
            "PUT": requests.put
        }.get(method, requests.get)
        response = request_method(url, json=data, headers=headers)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
        status_code = e.response.status_code
        error_msg = f"Server error (HTTP {status_code})"
        if status_code == 401:
            error_msg = "Authentication error. Please log in again."
            # Clear auth token to force re-login
            st.session_state.auth_token = None
        elif status_code == 404:
            error_msg = "Resource not found."
        elif status_code == 500:
            error_msg = "Internal server error. Please try again later."
        st.error(f"⚠️ {error_msg}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error: {e}")
        st.error("⚠️ Unable to connect to the server. Please check your internet connection.")
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout Error: {e}")
        st.error("⚠️ Request timed out. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Exception: {e}")
        st.error(f"⚠️ Unable to connect to the server. Error: {e}")
        return None

# Sidebar - Light/Dark Mode Toggle
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Theme selection with system default as an option
    theme_choice = st.radio(
        "Choose Theme",
        ["🌞 Light Mode", "🌙 Dark Mode", "🖥️ System Default"],
        index=2 if st.session_state.user_theme == "system" else 
              0 if st.session_state.user_theme == "light" else 1,
        key="theme"
    )

    # Update the session state with the chosen theme
    new_theme = (
        "dark" if theme_choice == "🌙 Dark Mode" else
        "light" if theme_choice == "🌞 Light Mode" else "system"
    )
    
    if new_theme != st.session_state.user_theme:
        st.session_state.user_theme = new_theme
        st.experimental_rerun() 

# Insert Custom Theme Styling Logic
if st.session_state.user_theme == "dark":
    st.markdown(
        """
        <style>
            body, .stApp, .st-b7,.st-fl {
                background-color: #121212 !important;
                color: white !important;
            }
            /* Explicit style for specific components */
            .st-emotion-cache-12fmjuu, .st-emotion-cache-128upt6, .st-bd,.st-fm {
                background-color: #121212 !important;
            }
            .stChatMessage, .chat-container, .stTextInput>div>div>input {
                background-color: #1E1E1E !important;
                color: white !important;
                border-radius: 8px;
            }
            .stTextInput>div>div>input {
                border: 1px solid #555 !important;
            }
            /* Fix Chat Input Box in Dark Mode */
            [data-testid="stChatInputContainer"] {
                background-color: #1E1E1E !important;
                color: #1E1E1E !important;
                border: 1px solid #555 !important;
                border-radius: 10px;
            }
            [data-testid="stChatInputContainer"] textarea,
            [data-testid="stChatInput"] textarea {
                background-color: #1E1E1E !important;
                color: white !important;
                border: 1px solid #555 !important;
            }
            [data-testid="stChatInputContainer"] textarea::placeholder,
            [data-testid="stChatInput"] textarea::placeholder {
                color: #CCCCCC !important;
                opacity: 1 !important;
                font-weight: bold !important;
            }
            .user-message {
                color: #FF4B4B !important;
                font-weight: bold;
                font-size: 18px;
                padding: 10px;
                display: block;
                text-align: left;
            }
            .bot-message {
                color: #4CAF50 !important;
                font-weight: bold;
                font-size: 18px;
                padding: 10px;
                display: block;
                text-align: left;
            }
            .stButton>button {
                background-color: #FF4B4B !important;
                color: white !important;
                border-radius: 8px;
            }
            [data-testid="stSidebar"] {
                background-color: #181818 !important;
                color: white !important;
            }
            [data-testid="stSidebar"] * {
                color: white !important;
            }
            /* Styles for dropdown selections and items */
            [data-baseweb="select"],
            [data-baseweb="select"] ul,
            [data-baseweb="select"] li {
                background-color: #1E1E1E !important;
                color: white !important;
                border: 1px solid #555 !important; /* Optional for dropdown borders */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif st.session_state.user_theme == "light":
    st.markdown("""<style>
        .st-b7 {
            background-color: #FFFFFF !important;
        }
        .st-bs,.st-fe {
            color: black !important;
        }
        .st-emotion-cache-hzygls {
            background-color: #FFFFFF !important;
        }
        .st-dy {
            color: red !important;
        }
        .st-cp, .st-co, .st-cn, .st-cm, .st-cl {
            background-color: #FFFFFF !important;
            border-color: #ccc !important;
        }
        body, .stApp, .block-container, .main, .chat-container, 
        .stChatMessage, .stTextInput>div>div>input, 
        [data-testid="stChatInputContainer"], 
        [data-testid="stTextArea"], [data-testid="stChatInput"] {
            background-color: #FFFFFF !important;
            color: black !important;
            border-radius: 8px;
        }
        [data-testid="stChatInputContainer"] {
            background-color: #FFFFFF !important;
            color: black !important;
            border: 1px solid #ccc !important;
            border-radius: 10px;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: black !important;
        }
        [data-testid="stChatInputContainer"] textarea::placeholder, 
        [data-testid="stChatInput"] textarea::placeholder {
            color: #555 !important;
            opacity: 1 !important;
            font-weight: bold !important;
        }
        input[type="password"] {
            background-color: white !important;
            color: black !important;
            border: 1px solid #ccc !important;
            border-radius: 8px;
        }
        input[type="password"]::-ms-reveal,
        input[type="password"]::-ms-clear,
        input[type="password"]::-webkit-reveal,
        input[type="password"]::-webkit-credentials-auto-fill-button,
        input[type="password"]::-webkit-contacts-auto-fill-button {
            background: transparent !important;
            border: none !important;
            color: #333 !important;
        }
        .st-ae.st-ay.st-d1.st-d2.st-cz.st-d0.st-dg.st-br.st-dh.st-bb.st-bs {
            background-color: transparent !important;
            border: none !important;
            cursor: pointer;
            display: flex;
            align-items: center;
            padding: 0;
        }
        .stButton>button {
            background-color: #1E88E5 !important;
            color: white !important;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 16px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1565C0 !important;
        }
        [data-testid="stSidebar"], .stSidebar {
            background-color: #F8F9FA !important;
            color: black !important;
        }
        [data-testid="stSidebar"] * {
            color: black !important;
        }
        footer, header {
            display: none !important;
        }
        ::-webkit-scrollbar {
            background: transparent;
            width: 6px;
        }
        ::-webkit-scrollbar-thumb {
            background: #ccc;
            border-radius: 10px;
        }
    </style>""",
    unsafe_allow_html=True)

# Initialize session states
for key in ["auth_token", "username", "messages", "chat_sessions", "selected_chat", "is_guest", "clear_chat_confirm", "clear_history_confirm"]:
    if key not in st.session_state:
        st.session_state[key] = None if "is" not in key else False

# Function to convert UTC timestamp to IST
def convert_utc_to_ist(utc_timestamp):
    if not utc_timestamp:
        return "Unknown Date"
    ist_tz = pytz.timezone("Asia/Kolkata")
    utc_time = datetime.datetime.strptime(utc_timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
    return utc_time.astimezone(ist_tz).strftime("%Y-%m-%d %I:%M %p")

def store_chat_history():
    if not st.session_state.auth_token or st.session_state.is_guest or not st.session_state.messages:
        logging.info(f"Not storing chat history: auth_token={bool(st.session_state.auth_token)}, is_guest={st.session_state.is_guest}, messages={len(st.session_state.messages) if st.session_state.messages else 0}")
        return
    
    # Check if we have both user and assistant messages
    has_user_message = any(msg["role"] == "user" for msg in st.session_state.messages)
    has_assistant_message = any(msg["role"] == "assistant" for msg in st.session_state.messages)
    
    if not (has_user_message and has_assistant_message):
        logging.info(f"Not storing chat history: missing user or assistant messages. User: {has_user_message}, Assistant: {has_assistant_message}")
        return
    
    logging.info(f"Storing chat history with {len(st.session_state.messages)} messages")
    logging.info(f"Selected chat: {st.session_state.selected_chat}")
    logging.info(f"Messages: {st.session_state.messages}")

    # If the user has loaded an existing chat, update that chat history
    if st.session_state.selected_chat:
        chat_id = st.session_state.selected_chat
        data = {
            "messages": st.session_state.messages,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        response = api_request(f"/chat_history/{chat_id}", method="PUT", data=data)
        logging.info(f"Update chat response: {response.status_code if response else 'No response'}")
    else:
        # Otherwise, create a new chat entry
        data = {
            "messages": st.session_state.messages,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        response = api_request("/chat_history/", method="POST", data=data)
        logging.info(f"New chat response: {response.status_code if response else 'No response'}")
        
        # After successfully saving a new chat, refresh the chat sessions and get the new chat ID
        if response and response.status_code == 200:
            try:
                new_chat_id = response.json().get("chat_id")
                if new_chat_id:
                    st.session_state.selected_chat = new_chat_id
                    logging.info(f"Set selected_chat to new chat ID: {new_chat_id}")
            except Exception as e:
                logging.error(f"Error getting new chat ID: {e}")
            
            # Refresh chat sessions
            load_chat_sessions()

    if response and response.status_code == 200:
        logging.info("✅ Chat history saved successfully.")
        return True
    else:
        logging.error("❌ Failed to save chat history.")
        return False

# Function to start a new conversation
def start_new_conversation():
    # Store the current chat history before starting a new one
    if st.session_state.messages:
        if not store_chat_history():
            st.error("Failed to save current chat history.")
    
    # Clear messages and reset selected chat
    st.session_state.messages = []
    st.session_state.selected_chat = None
    st.session_state.latest_responses = None
    logging.info("Started new conversation - cleared messages and selected_chat")

# Function to load chat history from API
def load_chat_sessions():
    if st.session_state.is_guest:
        return  # Don't load history for guest users

    response = api_request("/chat_history/")
    if response and response.status_code == 200:
        raw_history = response.json().get("history", {})
        logging.info(f"Loaded chat history: {raw_history}")  # Log loaded history
        chat_sessions = [
            {"id": chat["id"], "timestamp": convert_utc_to_ist(chat["timestamp"])}
            for date, chats in raw_history.items() for chat in chats
        ]
        st.session_state.chat_sessions = chat_sessions
    else:
        st.session_state.chat_sessions = []

# Function to load a selected chat
def load_selected_chat(chat_id):
    # Clear existing messages before loading
    st.session_state.messages = []
    
    response = api_request(f"/chat_history/{chat_id}")

    if response and response.status_code == 200:
        chat_data = response.json()
        logging.info(f"Loaded chat data: {chat_data}")  # Debugging

        # Ensure messages are loaded properly
        if chat_data.get("messages"):
            st.session_state.messages.extend(
                [{"role": msg["role"], "content": msg["content"]} for msg in chat_data["messages"]]
            )
        else:
            st.session_state.messages.append({"role": "assistant", "content": "No response available."})

        # Debugging to check if bot responses are actually retrieved
        logging.info(f"Updated session state messages: {st.session_state.messages}")
    else:
        st.error("⚠️ Failed to load chat messages.")

# Sidebar - User Authentication & Chat History
with st.sidebar:
    st.title("🔑 Login / Signup")

    if st.session_state.auth_token is None:
        auth_mode = st.radio("Choose an option:", ["Login", "Signup", "Guest Access"])

        if auth_mode != "Guest Access":
            username = st.text_input("👤 Username").strip()
            password = st.text_input("🔒 Password", type="password").strip()

        def handle_auth(response, username, is_login=True):
            if response.status_code == 200:
                st.session_state.auth_token = response.json().get("token")
                st.session_state.username = username
                st.session_state.is_guest = False
                st.session_state.messages = []  # Clear messages
                st.session_state.selected_chat = None  # Reset selected chat
                load_chat_sessions()
                st.rerun()
            else:
                error_detail = response.json().get("detail", "An error occurred during authentication.")
                st.error(f"❌ {error_detail}")

        if auth_mode == "Login" and st.button("🔓 Login"):
            try:
                response = requests.post(f"{API_AUTH_URL}/login", json={"username": username, "password": password})
                handle_auth(response, username)
            except requests.exceptions.RequestException:
                st.error("❌ Unable to connect to the server. Please check your connection.")

        elif auth_mode == "Signup" and st.button("📝 Signup"):
            try:
                response = requests.post(f"{API_AUTH_URL}/register", json={"username": username, "password": password})
                if response.status_code == 200:
                    st.success("✅ Signup successful! Please log in.")
                else:
                    error_detail = response.json().get("detail", "An unexpected error occurred during signup.")
                    st.error(f"❌ {error_detail}")
            except requests.exceptions.RequestException:
                st.error("❌ Unable to connect to the server. Please check your connection.")

        elif auth_mode == "Guest Access" and st.button("👤 Continue as Guest"):
            st.session_state.auth_token = "guest"
            st.session_state.username = "Guest"
            st.session_state.is_guest = True
            st.success("Logged in as guest!")
            st.rerun()

    else:
        st.markdown(f"👤 Logged in as **{st.session_state.username}**")

        if not st.session_state.is_guest:
            # Add a New Chat button
            if st.button("📝 New Chat"):
                start_new_conversation()
                st.rerun()
                
            load_chat_sessions()

            if st.session_state.chat_sessions:
                chat_options = {f"Chat on {chat['timestamp']}": chat["id"] for chat in st.session_state.chat_sessions}
                selected_chat_title = st.selectbox("📜 Chat History by Date", options=list(chat_options.keys()), key="selected_chat_title")

                if st.button("📂 Load Chat"):
                    st.session_state.selected_chat = chat_options.get(selected_chat_title)
                    load_selected_chat(st.session_state.selected_chat)
                    st.rerun()  # Rerun to reflect loaded chat

        # Move the delete chat history button here
        if not st.session_state.is_guest:
            if st.button("🗑️ Delete Chat History"):
                st.session_state.clear_history_confirm = True

            if st.session_state.clear_history_confirm:
                st.warning("Are you sure you want to delete all chat history?")
                if st.button("Yes, delete it"):
                    response = api_request("/chat_history/delete/", method="DELETE")
                    if response and response.status_code == 200:
                        st.session_state.messages.clear()
                        st.session_state.chat_sessions.clear()
                        st.success("✅ Chat history deleted successfully!")
                    else:
                        st.error("❌ Failed to delete chat history. Please try again.")
                    
                    st.session_state.clear_history_confirm = False
                    st.rerun()  # Rerun to reflect changes
                if st.button("Cancel"):
                    st.session_state.clear_history_confirm = False

# Initialize user_query to None
user_query = None

# Main App - Chat Interface
if 'messages' not in st.session_state or st.session_state.messages is None:
    st.session_state.messages = []

if 'latest_responses' not in st.session_state:
    st.session_state.latest_responses = None

if st.session_state.auth_token:
    st.markdown("<h1 style='text-align: center;'>🩺 Health Assistant Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>💡 Ask me about medical symptoms, treatments, and general health advice!</p>", unsafe_allow_html=True)

    # Iterate through messages to display and add TTS functionality
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            color = "red" if message["role"] == "user" else "green"
            role_icon = "🧑‍💬" if message["role"] == "user" else "🤖"
            st.markdown(f"<div style='color:{color}; font-weight:bold; font-size:18px;'>"
                        f"{role_icon} <b>{message['role'].capitalize()}:</b> {message['content']}</div>",
                        unsafe_allow_html=True)
            if message["role"] == "assistant":
                audio_fp = create_audio(message['content'])
                if st.button("🔊 Listen", key=f"listen_{idx}"):
                    st.audio(audio_fp, format="audio/mp3", start_time=0)

    # Capture Speech Button
    if st.button("🎙️ Speak"):
        user_query = recognize_speech()
        if user_query:
            user_query = user_query.lower().strip()
            logging.debug(f"Recognized voice input: {user_query}")
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(f"<p style='color:red; font-weight:bold; font-size:18px;'>🧑‍💬 <b>You:</b> {user_query}</p>", unsafe_allow_html=True)

            # Generate a single response from the Gemini API
            with st.spinner("🤖 Thinking..."):
                response = biogpt.generate_response_with_gemini(user_query)

            # Append the response to the chat
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Display the response in normal format
            with st.chat_message("assistant"):
                st.markdown(f"<p style='color:green; font-weight:bold; font-size:18px;'>🤖 <b>SvasthBot:</b> {response}</p>", unsafe_allow_html=True)

            # Store chat history after both user and assistant messages are added
            if not store_chat_history():
                st.error("Failed to save chat history.")

    # User Input Textbox
    query = st.chat_input("💬 Type your message here...")
    if query:
        query = query.strip()
        logging.debug(f"Text input captured: {query}")
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(f"<p style='color:red; font-weight:bold; font-size:18px;'>🧑‍💬 <b>You:</b> {query}</p>", unsafe_allow_html=True)

        # Generate a single response from the Gemini API
        with st.spinner("🤖 Thinking..."):
            response = biogpt.generate_response_with_gemini(query)

        # Append the response to the chat
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the response in normal format
        with st.chat_message("assistant"):
            st.markdown(f"<p style='color:green; font-weight:bold; font-size:18px;'>🤖 <b>SvasthBot:</b> {response}</p>", unsafe_allow_html=True)

        # Store chat history after both user and assistant messages are added
        if not store_chat_history():
            st.error("Failed to save chat history.")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.clear_chat_confirm = True

    if st.session_state.clear_chat_confirm:
        st.warning("Are you sure you want to clear the chat?")
        if st.button("Yes, clear it"):
            start_new_conversation()
            st.session_state.clear_chat_confirm = False
            st.success("Chat cleared successfully.")
            st.rerun()
        if st.button("Cancel"):
            st.session_state.clear_chat_confirm = False
else:
    st.markdown("<h2 style='text-align: center;'>🔒 Please log in to access the chatbot.</h2>", unsafe_allow_html=True)



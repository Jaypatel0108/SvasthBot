import requests
import os
import sqlite3
import logging
from dotenv import load_dotenv
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'feedback_data.db')

def init_db():
    """Initialize the database and create the user_feedback table if it doesn't exist."""
    try:
        if not os.path.exists(DB_PATH):
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    gemini_response TEXT NOT NULL,
                    preferred TEXT NOT NULL
                );
            """)
            conn.commit()
            logging.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")

init_db()

class BioGPT:
    def __init__(self):
        """Initialize the BioGPT class with API keys and URLs."""
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"  # Updated API endpoint

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.gemini_api_key:
            logging.error("API Key missing! Check environment variables.")
            raise ValueError("API Key missing!")

    def generate_response(self, prompt):
        """Generate a response using the Gemini API."""
        return self.generate_response_with_gemini(prompt)  # Make sure we're using the correct function

    def generate_response_with_gemini(self, prompt, conversation_history=None):
        """Generate two responses using the Gemini API with health assistant and general context."""
        if not prompt:
            logging.error("Prompt is None or empty.")
            return "⚠️ Error: Prompt cannot be empty."

        # Create a dynamic system context for both medical and general questions
        system_context = '''You are SvasthBot, a professional medical AI assistant and a friendly conversational partner. 
        You can provide medical information and advice, as well as engage in general discussions. 
        Be informative, interactive, and approachable in your responses.'''

        # Combine conversation history with current prompt if available
        full_prompt = system_context
        if conversation_history:
            full_prompt += "\n\nPrevious conversation:\n" + "\n".join(conversation_history)
        full_prompt += "\n\nCurrent question: " + prompt

        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }]
        }

        try:
            # Generate first response
            logging.info("Generating first response")
            response1 = requests.post(
                f"{self.gemini_api_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            response1.raise_for_status()
            result1 = response1.json()

            # Generate second response with slightly different context
            system_context_2 = system_context + "\n\nProvide an alternative perspective or approach while following the same formatting rules."
            full_prompt_2 = system_context_2
            if conversation_history:
                full_prompt_2 += "\n\nPrevious conversation:\n" + "\n".join(conversation_history)
            full_prompt_2 += "\n\nCurrent question: " + prompt

            data["contents"][0]["parts"][0]["text"] = full_prompt_2
            
            logging.info("Generating second response")
            response2 = requests.post(
                f"{self.gemini_api_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            response2.raise_for_status()
            result2 = response2.json()

            # Extract and clean up responses
            response1_text = result1["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in result1 and result1["candidates"] else ""
            response2_text = result2["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in result2 and result2["candidates"] else ""

            # Clean up responses - remove any remaining markdown and ensure proper spacing
            def clean_response(text):
                # Remove markdown formatting
                text = text.replace('*', '').replace('**', '').replace('***', '')
                # Ensure proper spacing around lists
                text = re.sub(r'\n\s*-', '\n-', text)
                # Ensure proper spacing around paragraphs
                text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
                # Ensure proper spacing around disclaimer
                text = re.sub(r'\(Disclaimer:', '\n\n(Disclaimer:', text)
                return text.strip()

            response1_text = clean_response(response1_text)
            response2_text = clean_response(response2_text)

            logging.info(f"Generated two responses successfully")
            return response1_text, response2_text

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}")
            return "⚠️ Network error occurred.", "⚠️ Network error occurred."
        except (KeyError, IndexError, ValueError) as e:
            logging.error(f"Error processing Gemini API response: {e}")
            return "⚠️ Error: Unexpected response from Gemini API.", "⚠️ Error: Unexpected response from Gemini API."

    def collect_user_feedback(self, prompt, gemini_response, preferred):
        """Collect user feedback on the generated response."""
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Gemini Response: {gemini_response}")
        logging.info(f"Preferred: {preferred}")
        
        # Ensure gemini_response is a string
        if isinstance(gemini_response, tuple):
            gemini_response = ' '.join(gemini_response)  # Join tuple elements into a single string
        
        feedback_data = (prompt, gemini_response, preferred)  # Collecting user feedback data

        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_feedback (prompt, gemini_response, preferred)
                    VALUES (?, ?, ?)
                """, feedback_data)
                conn.commit()
                logging.info("User Feedback Recorded: %s", feedback_data)  # Logging feedback data

        except sqlite3.Error as e:
            logging.error(f"Error recording user feedback: {e}")

biogpt = BioGPT()

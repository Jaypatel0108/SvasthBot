import logging
import os
import re
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MedBERT:
    """
    A class for medical text understanding using transformer-based language models
    specialized for healthcare applications.
    """
    
    def __init__(self):
        # Initialize with a clinical language model if available
        self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.max_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"MedBERT model loaded successfully on {self.device}")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load MedBERT model: {str(e)}")
            self.model_loaded = False
            
            # Set up for API fallback - ensure this matches BioGPT's configuration
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not self.gemini_api_key:
                logger.warning("GEMINI_API_KEY not found in environment variables")
            
            # Gemini API URL - matched with BioGPT's URL for consistency
            self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    def extract_medical_terms(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical terms and concepts from text.
        
        Args:
            text: Input medical text
            
        Returns:
            List of extracted medical terms with metadata
        """
        # First clean and normalize the text
        cleaned_text = self._preprocess_text(text)
        
        if self.model_loaded:
            return self._extract_terms_with_model(cleaned_text)
        else:
            return self._extract_terms_with_api(cleaned_text)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess and normalize medical text.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert common abbreviations to standard form
        abbreviations = {
            "pt": "patient",
            "pts": "patients",
            "dx": "diagnosis",
            "sx": "symptoms",
            "hx": "history",
            "fx": "fracture",
            "tx": "treatment",
            "rxn": "reaction",
            "abd": "abdominal",
            "hr": "heart rate",
            "bp": "blood pressure",
            "temp": "temperature",
            "w/": "with",
            "w/o": "without",
            "yo": "year old",
            "y/o": "year old",
        }
        
        # Replace abbreviations
        for abbr, full in abbreviations.items():
            # Only replace when it's a whole word
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_terms_with_model(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical terms using the local transformer model.
        
        Args:
            text: Preprocessed input text
            
        Returns:
            List of extracted medical terms with metadata
        """
        extracted_terms = []
        
        try:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", 
                                    max_length=self.max_length, 
                                    truncation=True, 
                                    padding=True).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get the embeddings
            embeddings = outputs.last_hidden_state
            
            # Convert tokens back to words
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Simple medical term detection based on token properties
            medical_tokens = []
            current_term = []
            medical_prefixes = ["anti", "hyper", "hypo", "brady", "tachy", "cardio", 
                               "gastro", "neuro", "onco", "dermato", "osteo", "nephro"]
            medical_suffixes = ["itis", "emia", "osis", "pathy", "algia", "ectomy", 
                               "opsy", "oma", "logy", "itis", "gram", "ase"]
            
            for i, token in enumerate(tokens):
                if token.startswith("##"):
                    # Continue previous token
                    if current_term:
                        current_term.append(token[2:])
                else:
                    # Save previous term if it exists
                    if current_term:
                        term = "".join(current_term)
                        if len(term) > 3:  
                            # Check if it might be medical
                            if (any(term.lower().startswith(prefix) for prefix in medical_prefixes) or
                                any(term.lower().endswith(suffix) for suffix in medical_suffixes)):
                                medical_tokens.append(term)
                        current_term = []
                    
                    # Start new token
                    if not token.startswith("[") and not token.endswith("]"):
                        current_term.append(token)
            
            # Add any remaining term
            if current_term:
                term = "".join(current_term)
                if len(term) > 3:
                    if (any(term.lower().startswith(prefix) for prefix in medical_prefixes) or
                        any(term.lower().endswith(suffix) for suffix in medical_suffixes)):
                        medical_tokens.append(term)
            
            # Process the found tokens
            for term in medical_tokens:
                extracted_terms.append({
                    "term": term,
                    "type": "unknown",  # We'd need a classifier to determine type
                    "confidence": 0.7  # Placeholder confidence
                })
            
            return extracted_terms
            
        except Exception as e:
            logger.error(f"Error in transformer-based extraction: {str(e)}")
            # Fall back to API-based extraction
            return self._extract_terms_with_api(text)
    
    def _extract_terms_with_api(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical terms using the Gemini API.
        
        Args:
            text: Preprocessed input text
            
        Returns:
            List of extracted medical terms with metadata
        """
        try:
            # Prepare the prompt for structured extraction
            prompt = f"""
            You are a medical NLP system. Extract all medical terms from the text below and categorize them.
            For each term, identify:
            1. The term itself
            2. Type (condition, symptom, medication, procedure, anatomy, lab value, etc.)
            3. Confidence level (0.0-1.0)
            
            Format the response as a JSON array of objects with fields: "term", "type", and "confidence".
            
            Text to analyze: "{text}"
            
            Response (JSON only):
            """
            
            # Prepare API request - use consistent structure with BioGPT
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            # Call the Gemini API
            response = requests.post(
                f"{self.gemini_api_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text_response = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Extract JSON from response (handle potential text wrapping)
                json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        extracted_terms = json.loads(json_str)
                        return extracted_terms
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from API response")
                
                # If we couldn't extract valid JSON, do basic extraction
                return self._basic_term_extraction(text)
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return self._basic_term_extraction(text)
                
        except Exception as e:
            logger.error(f"Error in API-based extraction: {str(e)}")
            return self._basic_term_extraction(text)
    
    def _basic_term_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        Basic rule-based medical term extraction.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted terms
        """
        # Very basic medical term list for fallback
        medical_terms = [
            "fracture", "lesion", "inflammation", "tumor", "carcinoma", "melanoma",
            "infection", "pneumonia", "dermatitis", "fibrosis", "edema", "effusion",
            "necrosis", "atrophy", "hyperplasia", "dysplasia", "cyst", "abscess",
            "hernia", "stenosis", "sclerosis", "ischemia", "infarct", "embolism",
            "thrombosis", "hemorrhage", "aneurysm", "malformation", "hypertension",
            "hypotension", "tachycardia", "bradycardia", "arrhythmia", "murmur",
            "hyponatremia", "hyperkalemia", "anemia", "leukemia", "lymphoma", "syndrome"
        ]
        
        extracted = []
        
        # Simple pattern matching
        for term in medical_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                extracted.append({
                    "term": term,
                    "type": "medical",
                    "confidence": 0.6
                })
        
        return extracted
    
    def analyze_medical_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of medical text.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            Dict with analysis results
        """
        # Extract medical terms
        terms = self.extract_medical_terms(text)
        
        # Use Gemini API for deeper analysis
        try:
            # Prepare prompt for analysis
            prompt = f"""
            As a medical AI assistant, analyze the following medical text:
            
            "{text}"
            
            Provide:
            1. A summary of key medical findings
            2. Potential diagnoses suggested by the text
            3. Any critical values or concerning findings that need attention
            4. Relevant medical context for these findings
            
            Format your response as structured sections.
            """
            
            # Prepare API request
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            # Call the Gemini API
            response = requests.post(
                f"{self.gemini_api_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Structure the response
                return {
                    "extracted_terms": terms,
                    "analysis": analysis_text,
                    "has_critical_findings": self._check_for_critical_findings(terms, analysis_text)
                }
            else:
                logger.error(f"API error in analysis: {response.status_code} - {response.text}")
                return {
                    "extracted_terms": terms,
                    "analysis": "Failed to generate detailed analysis.",
                    "has_critical_findings": self._check_for_critical_findings(terms, "")
                }
                
        except Exception as e:
            logger.error(f"Error in medical text analysis: {str(e)}")
            return {
                "extracted_terms": terms,
                "analysis": "Failed to generate detailed analysis due to an error.",
                "has_critical_findings": self._check_for_critical_findings(terms, "")
            }
    
    def _check_for_critical_findings(self, terms: List[Dict[str, Any]], analysis_text: str) -> bool:
        """
        Check if there are any critical medical findings that need urgent attention.
        
        Args:
            terms: Extracted medical terms
            analysis_text: Generated analysis text
            
        Returns:
            bool: True if critical findings detected
        """
        # Critical keywords to look for
        critical_terms = [
            "emergency", "urgent", "critical", "severe", "immediately", "life-threatening",
            "stroke", "hemorrhage", "cardiac arrest", "respiratory failure", "sepsis",
            "anaphylaxis", "pulmonary embolism", "meningitis", "appendicitis", "myocardial infarction"
        ]
        
        # Check extracted terms
        for term in terms:
            if any(critical.lower() in term["term"].lower() for critical in critical_terms):
                return True
        
        # Check analysis text
        if any(critical.lower() in analysis_text.lower() for critical in critical_terms):
            return True
            
        # Check for urgency language
        urgency_patterns = [
            r"requires?\s+immediate",
            r"emergency\s+(care|attention|treatment)",
            r"urgent\s+(care|attention|treatment)",
            r"without\s+delay",
            r"life[\-\s]threatening",
            r"critical\s+(condition|situation)",
        ]
        
        for pattern in urgency_patterns:
            if re.search(pattern, analysis_text, re.IGNORECASE):
                return True
        
        return False
    
    def combine_image_and_text_analysis(
        self, 
        image_analysis: Dict[str, Any], 
        text_description: str
    ) -> Dict[str, Any]:
        """
        Combine image analysis results with text description for multi-modal analysis.
        
        Args:
            image_analysis: Results from image analysis
            text_description: Patient's description or clinical notes
            
        Returns:
            Dict: Combined analysis
        """
        # Extract medical terms from text
        extracted_terms = self.extract_medical_terms(text_description)
        
        # Get image labels if available
        image_labels = []
        if "analysis" in image_analysis and "categories" in image_analysis["analysis"]:
            image_labels = image_analysis["analysis"]["categories"]
        
        # Get conditions from image analysis
        image_conditions = []
        if "conditions" in image_analysis["analysis"]:
            image_conditions = image_analysis["analysis"]["conditions"]
        
        # Combine via Gemini API
        try:
            # Create a detailed prompt for multi-modal analysis
            prompt = f"""
            You are a medical AI performing multi-modal analysis of both image findings and text descriptions.
            
            IMAGE FINDINGS:
            - Detected features: {', '.join(image_labels) if image_labels else 'None specified'}
            - Potential conditions: {', '.join(image_conditions) if image_conditions else 'None specified'}
            
            PATIENT TEXT:
            "{text_description}"
            
            Extracted medical terms from text: {', '.join([term['term'] for term in extracted_terms]) if extracted_terms else 'None'}
            
            Provide a comprehensive analysis that:
            1. Identifies correlations between the image findings and text description
            2. Notes any consistencies or inconsistencies between the two
            3. Provides refined diagnostic possibilities based on the combined data
            4. Suggests appropriate next steps for clinical management
            
            Present your analysis in clearly separated sections.
            """
            
            # Prepare API request
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            # Call the Gemini API
            response = requests.post(
                f"{self.gemini_api_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                combined_analysis = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Create the final result
                return {
                    "image_analysis": image_analysis,
                    "text_analysis": {
                        "extracted_terms": extracted_terms,
                        "description": text_description
                    },
                    "combined_analysis": combined_analysis,
                    "disclaimer": "This combined analysis is provided for informational purposes only and should not replace professional medical diagnosis."
                }
            else:
                logger.error(f"API error in combined analysis: {response.status_code} - {response.text}")
                return {
                    "image_analysis": image_analysis,
                    "text_analysis": {
                        "extracted_terms": extracted_terms,
                        "description": text_description
                    },
                    "combined_analysis": "Failed to generate combined analysis.",
                    "disclaimer": "This analysis is provided for informational purposes only and should not replace professional medical diagnosis."
                }
                
        except Exception as e:
            logger.error(f"Error in combined analysis: {str(e)}")
            return {
                "image_analysis": image_analysis,
                "text_analysis": {
                    "extracted_terms": extracted_terms,
                    "description": text_description
                },
                "combined_analysis": f"Failed to generate combined analysis: {str(e)}",
                "disclaimer": "This analysis is provided for informational purposes only and should not replace professional medical diagnosis."
            }

# Initialize the model
medbert = MedBERT()


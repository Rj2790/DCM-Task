import requests
import json
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
CORS(app)

# Read API key from api.txt
try:
    with open('api.txt', 'r') as f:
        OPENROUTER_API_KEY = f.read().strip()
except FileNotFoundError:
    print("api.txt file not found!")
    print("Please create api.txt file with your OpenRouter API key")
    sys.exit(1)

GOOGLE_SHEETS_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxuaLIKsCjjKZulByjiAlW7eeEdgUU0a96KByA205BGsHAs9p6ed4cbDq0g0kiC7ts9YA/exec"

# If no API key is set, show usage and exit
if OPENROUTER_API_KEY == "YOUR_API_KEY_HERE" or not OPENROUTER_API_KEY:
    print("❌ Please update api.txt with your OpenRouter API key")
    print("Edit api.txt and replace YOUR_API_KEY_HERE with your actual key")
    print("\nGet a free key at: https://openrouter.ai/")
    sys.exit(1)

# Initialize sentence transformer model
print("Loading sentence transformer model...")
try:
    model = SentenceTransformer('BAAI/bge-small-en')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load precomputed FAQ embeddings
print("Loading precomputed FAQ embeddings...")
try:
    with open("faq_embeddings.json", "r") as f:
        FAQS = json.load(f)
    print(f"Loaded {len(FAQS)} FAQ embeddings!")
except Exception as e:
    print(f"Error loading FAQ embeddings: {e}")
    FAQS = []

def get_embedding(text):
    """Get embedding using sentence-transformers"""
    if model is None:
        return None
    try:
        embedding = model.encode([text])[0]
        return embedding.tolist()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    try:
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0

def get_llm_response(user_query, faq_answer):
    """Use OpenRouter LLM to make the answer more natural"""
    try:
        prompt = f"""You are a professional customer service representative for DCM Moguls, a digital marketing company.

User asked: "{user_query}"

Here is the relevant information: {faq_answer}

Please provide a concise, professional, and polite response that directly answers the user's question. Keep it brief (2-3 sentences maximum) and maintain a professional yet friendly tone. Be straightforward but courteous.

IMPORTANT: Do NOT include any preamble, explanation, or phrases like "Here's a possible response." Do NOT wrap your response in quotes. Only reply with the message you would send to the user.
"""
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://huggingface.co",
                "X-Title": "DCM Moguls Chatbot",
            },
            data=json.dumps({
                "model": "meta-llama/llama-3.2-1b-instruct:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3
            }),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result['choices'][0]['message']['content'].strip()
            
            # Remove quotes from beginning and end if present
            if llm_response.startswith('"') and llm_response.endswith('"'):
                llm_response = llm_response[1:-1]
            
            return llm_response
        else:
            print(f"LLM API error: {response.status_code} - {response.text}")
            return faq_answer
            
    except Exception as e:
        print(f"LLM error: {e}")
        return faq_answer

def send_to_google_sheets(user_data):
    """Send user data to Google Sheets webhook"""
    payload = {
        "name": user_data.get("fullName", ""),
        "email": user_data.get("email", ""),
        "phone": user_data.get("phone", ""),
        "professionType": user_data.get("businessType", "")
    }
    
    print(f"=== GOOGLE SHEETS DEBUG ===")
    print(f"Payload: {payload}")
    print(f"Webhook URL: {GOOGLE_SHEETS_WEBHOOK_URL}")
    
    try:
        response = requests.post(
            GOOGLE_SHEETS_WEBHOOK_URL, 
            json=payload, 
            timeout=15,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        # Google Apps Script returns JSON with status field
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('status') == 'success':
                    print("Data sent to Google Sheets successfully")
                    return True
                else:
                    print(f"Google Sheets returned error: {result}")
                    return False
            except json.JSONDecodeError:
                # If response is not JSON, check if it contains success
                if 'success' in response.text.lower():
                    print("Data sent to Google Sheets successfully")
                    return True
                else:
                    print(f"Unexpected response format: {response.text}")
                    return False
        else:
            print(f"Failed to send to Google Sheets: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error sending to Google Sheets: {e}")
        return False

@app.route("/")
def home():
    return send_from_directory('.', 'web.html')

@app.route("/faq", methods=["POST"])
def faq():
    try:
        user_query = request.json["query"]
        print("Received query:", user_query)
        
        if not FAQS:
            return jsonify({"answer": "FAQ system is not available at the moment."}), 500
        
        # Get embedding for user query
        user_embedding = get_embedding(user_query)
        if user_embedding is None:
            return jsonify({"answer": "Sorry, there was an error processing your question."}), 500

        # Compute similarity with each FAQ
        similarities = []
        for faq in FAQS:
            if faq["embedding"] is not None:
                similarity = cosine_similarity(user_embedding, faq["embedding"])
                similarities.append(similarity)
            else:
                similarities.append(0)
        
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        
        print(f"Best match: {FAQS[best_idx]['question']} (similarity: {best_score:.3f})")

        if best_score > 0.8:  
            faq_answer = FAQS[best_idx]["answer"]
            
            # Use LLM to make the answer more natural
            print("Using LLM to generate natural response...")
            natural_answer = get_llm_response(user_query, faq_answer)
            print(f"LLM response: {natural_answer}")
            
            answer = natural_answer
        else:
            print(f"Similarity score {best_score:.3f} below threshold 0.8 - returning sorry message")
            answer = "I'm sorry, I couldn't find an answer to your question. Do you have any other queries about our company and the services we provide?"

        return jsonify({"answer": answer})
    except Exception as e:
        print(f"FAQ error: {e}")
        return jsonify({"answer": "Sorry, there was an error processing your request."}), 500

@app.route("/save-user", methods=["POST"])
def save_user():
    try:
        user_data = request.json
        print(f"=== SAVE USER DEBUG ===")
        print(f"Received user data: {user_data}")
        print(f"Google Sheets URL configured: {bool(GOOGLE_SHEETS_WEBHOOK_URL)}")
        print(f"Google Sheets URL: {GOOGLE_SHEETS_WEBHOOK_URL}")
        
        success = send_to_google_sheets(user_data)
        
        if success:
            print("User data saved successfully")
            return jsonify({"status": "success"})
        else:
            print("Failed to save user data")
            return jsonify({"status": "error", "message": "Failed to send to Google Sheets"}), 500
    except Exception as e:
        print(f"Save user error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("=== FLASK APP STARTUP ===")
    print(f"OpenRouter API Key configured: {'Yes' if OPENROUTER_API_KEY else 'No'}")
    print(f"Google Sheets URL configured: {'Yes' if GOOGLE_SHEETS_WEBHOOK_URL else 'No'}")
    print(f"Google Sheets URL: {GOOGLE_SHEETS_WEBHOOK_URL}")
    print("=== STARTING SERVER ===")
    app.run(host="0.0.0.0", port=7860, debug=True)

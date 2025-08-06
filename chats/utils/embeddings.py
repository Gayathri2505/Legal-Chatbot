import os
import requests
from dotenv import load_dotenv
from pathlib import Path
# from sentence_transformers import SentenceTransformer

# # Define the path to the locally stored model
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# #BASE_DIR = Path(__file__).resolve().parent.parent
# MODEL_PATH = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")

# # Load the model once when Django starts (from local path)
# model = SentenceTransformer(MODEL_PATH)

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

API_URL = (
    "https://router.huggingface.co/"
    "hf-inference/models/sentence-transformers/"
    "all-MiniLM-L6-v2/pipeline/feature-extraction"
)
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


# def generate_embedding(text: str) -> list:
#     return model.encode(text).tolist()

def generate_embedding(text: str) -> list:
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error {response.status_code}: {response.text}")
    
    data = response.json()
    
    # The API returns [embedding_vector] if single text input
    # Ensure flat list, same as model.encode().tolist()
    if isinstance(data, list) and isinstance(data[0], list):
        return data[0]
    return data
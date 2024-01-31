import os

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Language : only french ("fr") and english ("en") available
LANGUAGE_CHOICE = "fr"
LANGUAGE_DICT = {"fr" : "french", "en" : "english"}

# Upload directory
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# Allowed extensions for the upload
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Interface mode : local by default
BOOL_API = True

# LLM load and location
BOOL_LLM = True
LLM_FOLDER = "local_llm/mistralai/Mistral-7B-v0.1"
EMBEDDINGS_FOLDER = "local_llm/sentence-transformers/all-mpnet-base-v2"

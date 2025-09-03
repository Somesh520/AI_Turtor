import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------------------------------
# --- Core Model & LLM Configuration ---
# --------------------------------------------------------------------------
# This is the main model used for reasoning, routing, and translation.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

# A lower temperature makes the model more predictable and less creative.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# This is the model used to create numerical representations (embeddings) of text for the RAG system.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

# Google API Key - Set this as an environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------------------------------------------------------------
# --- Server Configuration ---
# --------------------------------------------------------------------------
# Backend server configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Frontend server configuration
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# --------------------------------------------------------------------------
# --- Development Configuration ---
# --------------------------------------------------------------------------
# Debug mode
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Verbose logging
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

# --------------------------------------------------------------------------
# --- RAG (Retrieval-Augmented Generation) Configuration ---
# --------------------------------------------------------------------------
# This defines the data sources for our different RAG-based tools.
# Each tool has its own folder of .txt files and its own vector database.

# For the tool that teaches the structured course curriculum.
CURRICULUM_DATA_PATH = os.getenv("CURRICULUM_DATA_PATH", "./data/curriculum")
CURRICULUM_DB_PATH = os.getenv("CURRICULUM_DB_PATH", "./chroma_db_curriculum")

# For the tool that answers general questions about grammar and vocabulary.
GRAMMAR_DATA_PATH = os.getenv("GRAMMAR_DATA_PATH", "./data/grammar_vocab")
GRAMMAR_DB_PATH = os.getenv("GRAMMAR_DB_PATH", "./chroma_db_grammar")

# --------------------------------------------------------------------------
# --- Retriever Search Configuration ---
# --------------------------------------------------------------------------
# This configures how the RAG system searches for relevant documents.
# "mmr" (Maximal Marginal Relevance) is used to get diverse and relevant results.
RETRIEVER_SEARCH_TYPE = "mmr"

# 'k' is the final number of documents to return.
# 'fetch_k' is the number of documents to initially fetch before re-ranking for diversity.
RETRIEVER_SEARCH_KWARGS = {'k': 4, 'fetch_k': 20}

# --------------------------------------------------------------------------
# --- Controlled Lesson Flow Configuration ---
# --------------------------------------------------------------------------
# This feature ensures the bot only loads one lesson at a time for the Curriculum Tutor.
# We will use this filename format to identify and load lessons.
# Example: "lesson_1.txt", "lesson_2.txt", etc.
LESSON_FILENAME_PREFIX = "lesson_"
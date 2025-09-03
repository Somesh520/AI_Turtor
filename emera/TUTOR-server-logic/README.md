
# Sanskrit Tutor Bot - Google Gemini API

A Sanskrit tutoring application powered by Google's Gemini 1.5 Flash model, featuring interactive lessons, grammar assistance, and translation capabilities.

## Features

- **Interactive Lessons**: Structured curriculum with progressive lesson unlocking
- **Grammar & Vocabulary Expert**: RAG-powered assistance for Sanskrit grammar questions
- **Translation Tool**: Sanskrit to English and English to Sanskrit translation
- **Conversational AI**: General conversation and guidance
- **Streaming Responses**: Real-time chat experience
- **Adaptive Learning**: AI that learns from user interactions and adapts teaching style

## Quick Start

### 1. Verify Your Setup
```bash
python check_setup.py
```

This script will check your Python version, dependencies, environment configuration, and data files.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Google Gemini API Key
```bash
python setup_api_key.py
```

This interactive script will:
- Guide you through getting an API key from Google AI Studio
- Create a proper `.env` file with all necessary configuration
- Validate your API key format

### 4. Start the Application

**Terminal 1 - Backend Server:**
```bash
python main.py
```

**Terminal 2 - Frontend Interface:**
```bash
streamlit run app.py
```

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Get API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click 'Create API Key'
4. Copy the generated key

### 2. Create .env File
Create a file named `.env` in the project root:
```env
GOOGLE_API_KEY=your-actual-api-key-here
LLM_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.1
EMBEDDING_MODEL=models/embedding-001
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
STREAMLIT_PORT=8501
```

### 3. Set Environment Variables (Alternative)
**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY='your-api-key-here'
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export GOOGLE_API_KEY='your-api-key-here'
```

## Troubleshooting

### Common Issues

1. **"GOOGLE_API_KEY environment variable is not set"**
   - Run `python setup_api_key.py` to set up your API key
   - Or manually create a `.env` file with your API key

2. **"Could not reach the backend server"**
   - Ensure the backend is running: `python main.py`
   - Check that port 8000 is not blocked by firewall

3. **"Error during startup"**
   - Verify your API key is correct
   - Check your internet connection
   - Ensure you have access to Google AI Studio

4. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Setup Verification
Run the verification script to diagnose issues:
```bash
python check_setup.py
```

## API Documentation

### Base URL
`http://127.0.0.1:8000`

### Endpoints

#### `/chat/stream`

The primary endpoint for all user interactions. Handles routing to the correct tool and returns a streaming response.

- **Method:** `POST`
- **Description:** Sends a user query and conversation context to the agent, which decides which tool to use and streams back the response.

- **Request Body (JSON):**
```json
{
  "query": "What is Sandhi?",
  "previous_query": null,
  "previous_response": null,
  "lesson_to_teach": null
}
```

- **Parameters:**
  - `query` (string, required): The current question from the user
  - `previous_query` (string or null): The user's immediately preceding question
  - `previous_response` (string or null): The agent's immediately preceding response
  - `lesson_to_teach` (integer or null): Direct lesson number to teach (bypasses routing)

- **Success Response:**
  - **Status Code:** `200 OK`
  - **Content-Type:** `text/event-stream`
  - **Body:** Streaming response of text chunks

- **Error Response:**
  - **Status Code:** `500 Internal Server Error`
  - **Body (JSON):**
```json
{
  "detail": "A descriptive error message."
}
```

## Configuration

Key configuration options in `config.py`:

- `LLM_MODEL`: Gemini model name (default: "gemini-1.5-flash")
- `LLM_TEMPERATURE`: Model creativity (default: 0.1)
- `EMBEDDING_MODEL`: Embedding model for RAG (default: "models/embedding-001")
- `SERVER_PORT`: Backend server port (default: 8000)
- `STREAMLIT_PORT`: Frontend port (default: 8501)

## Data Structure

The application expects the following directory structure:

```
data/
├── curriculum/
│   ├── lesson_1.txt
│   ├── lesson_2.txt
│   └── ...
└── grammar_vocab/
    ├── sanskrit_basics.txt
    └── ...
```

## Example Usage

1. **Start a lesson**: Use the sidebar to select and start a lesson
2. **Ask grammar questions**: "What is Sandhi?" or "Explain verb conjugation"
3. **Request translations**: "Translate 'hello' to Sanskrit" or "What does 'namaste' mean?"
4. **General conversation**: Ask about Sanskrit culture, history, or learning tips

## Development

### Project Structure
- `main.py` - FastAPI backend server
- `app.py` - Streamlit frontend interface
- `agent_logic.py` - AI agent and RAG system
- `prompts.py` - AI prompt templates
- `config.py` - Configuration settings
- `setup_api_key.py` - Interactive API key setup
- `check_setup.py` - Setup verification script

### Adding New Features
1. **New Tools**: Add to `agent_logic.py` and update routing logic
2. **New Prompts**: Add to `prompts.py` and reference in agent logic
3. **New Data**: Add files to appropriate data directories and update RAG retrievers

## Support

If you encounter issues:
1. Run `python check_setup.py` to diagnose problems
2. Check the console output for error messages
3. Verify your API key and internet connection
4. Ensure all dependencies are installed correctly

---
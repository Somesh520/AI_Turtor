import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, List
from dotenv import load_dotenv

# --- 1. FastAPI setup & Environment Variables ---
load_dotenv()
app = FastAPI(
    title="AI Tutor API",
    description="API for the multi-language AI Tutor application.",
    version="1.0.0"
)

# --- 2. Agent initialization ---
agent_chains = {}

@app.on_event("startup")
async def startup_event():
    """Initializes the AI agent when the server starts."""
    try:
        print("=" * 50)
        print("🚀 STARTING AGENT INITIALIZATION")
        print("=" * 50)
        
        # Step 1: Check API Key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not found in .env file")
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        print(f"✅ Step 1: API Key found: {api_key[:10]}...")
        
        # Step 2: Test basic imports
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ Step 2: ChatGoogleGenerativeAI import successful")
        
        # Step 3: Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",  # <<< FIX: Changed from "gemini-1.5-flash" to the more stable "gemini-pro"
            temperature=0.7,
            google_api_key=api_key
        )
        print("✅ Step 3: LLM initialized successfully")
        
        # Step 4-6: Import other necessary modules
        from config import GOOGLE_API_KEY as config_key
        print("✅ Step 4: Config import successful")
        from prompts import CURRICULUM_TUTOR_PROMPT
        print("✅ Step 5: Prompts import successful")
        from agent_logic import create_tutor_agent
        print("✅ Step 6: Agent logic import successful")
        
        # Step 7: Create agent
        print("🔧 Step 7: Creating tutor agent...")
        agent_result = create_tutor_agent(llm, grammar_retriever=None)
        print(f"✅ Step 7: Agent creation returned: {type(agent_result)}")
        
        # Step 8: Update agent chains
        agent_chains.update(agent_result)
        print(f"✅ Step 8: Agent chains updated: {list(agent_chains.keys())}")
        
        # Step 9: Final verification
        if "agent" in agent_chains and "curriculum" in agent_chains:
            print("🎉 SUCCESS: Agent ready and all chains loaded!")
        else:
            print(f"⚠️ WARNING: Some chains missing. Available: {list(agent_chains.keys())}")
            
        print("=" * 50)

    except Exception as e:
        print("=" * 50)
        print(f"❌ FATAL: Agent initialization failed: {e}")
        import traceback
        print(f"❌ FULL ERROR: {traceback.format_exc()}")
        print("=" * 50)
        agent_chains.clear()

# --- 3. Pydantic models ---

class ChatRequest(BaseModel):
    query: str
    language: Optional[str] = None
    previous_query: Optional[str] = None
    previous_response: Optional[str] = None
    lesson_to_teach: Optional[int] = None

class Lesson(BaseModel):
    number: int
    title: str

class LessonsResponse(BaseModel):
    lessons: List[Lesson]

# --- 4. API Endpoints ---

@app.get("/")
async def root():
    return {"message": "🚀 AI Tutor API is running!", "version": "1.0.0"}

CURRICULUM_PATH = "curriculum"

@app.get("/lessons", response_model=LessonsResponse)
async def get_lessons(language: str):
    """Fetches the list of available lessons for a given language."""
    if not language:
        raise HTTPException(status_code=400, detail="Language query parameter is required.")
    language_path = os.path.join(CURRICULUM_PATH, language.lower())
    print(f"🔍 Looking for lessons in: {language_path}")
    if not os.path.isdir(language_path):
        return {"lessons": []}
    try:
        files = sorted([f for f in os.listdir(language_path) if f.startswith("lesson_") and f.endswith(".txt")],
                       key=lambda x: int(x.split("_")[1].split(".")[0]))
        lessons = []
        for f in files:
            lesson_num = int(f.split("_")[1].split(".")[0])
            with open(os.path.join(language_path, f), "r", encoding="utf-8") as file:
                title = file.readline().strip() or f"Lesson {lesson_num}"
            lessons.append({"number": lesson_num, "title": title})
        print(f"✅ Returning {len(lessons)} lessons for {language}")
        return {"lessons": lessons}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read lesson files: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handles chat requests by invoking the appropriate agent chain."""
    print("=" * 30, f"\n🔥 CHAT REQUEST | Lesson: {request.lesson_to_teach} | Lang: {request.language}", "\n" + "=" * 30)
    
    if not agent_chains:
        raise HTTPException(status_code=503, detail="Agent is not available. Please check server logs.")

    chain_to_run = None
    agent_input = {}

    if request.lesson_to_teach is not None:
        print(f"📚 Teaching lesson {request.lesson_to_teach}")
        chain_to_run = agent_chains.get("curriculum")
        if not request.language:
            raise HTTPException(status_code=400, detail="Language is required for lessons.")
        file_path = os.path.join(CURRICULUM_PATH, request.language.lower(), f"lesson_{request.lesson_to_teach}.txt")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Lesson file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            lesson_content = f.read()
        agent_input = {"context": lesson_content, "language": request.language}
    else:
        print("💬 General chat request")
        chain_to_run = agent_chains.get("agent")
        agent_input = {
            "current_question": request.query,
            "previous_query": request.previous_query or "",
            "previous_response": request.previous_response or "",
            "language": request.language or "English"
        }

    if not chain_to_run:
        raise HTTPException(status_code=503, detail="Appropriate agent chain is not available.")
        
    try:
        print(f"🤖 Invoking agent with keys: {list(agent_input.keys())}")
        response: Any = await chain_to_run.ainvoke(agent_input)
        output = response.get("output", str(response)) if isinstance(response, dict) else str(response)
        print(f"✅ Agent invocation successful. Output length: {len(output)}")
        return {"response": output}
    except Exception as e:
        print(f"❌ Error during agent invocation: {e}")
        import traceback
        print(f"❌ Full error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing your request: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server and agent status."""
    return {
        "status": "healthy",
        "agent_status": "ready" if agent_chains else "not_ready",
        "available_chains": list(agent_chains.keys()),
        "google_api_key_exists": bool(os.getenv("GOOGLE_API_KEY"))
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
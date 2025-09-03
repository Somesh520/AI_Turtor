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
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("✅ Step 2: ChatGoogleGenerativeAI import successful")
        except Exception as e:
            print(f"❌ Step 2: Import error: {e}")
            raise e
        
        # Step 3: Initialize LLM
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
                google_api_key=api_key
            )
            print("✅ Step 3: LLM initialized successfully")
        except Exception as e:
            print(f"❌ Step 3: LLM initialization error: {e}")
            raise e
        
        # Step 4: Test config import
        try:
            from config import GOOGLE_API_KEY as config_key
            print("✅ Step 4: Config import successful")
        except Exception as e:
            print(f"❌ Step 4: Config import error: {e}")
            # Continue without config for now
        
        # Step 5: Test prompts import
        try:
            from prompts import CURRICULUM_TUTOR_PROMPT
            print("✅ Step 5: Prompts import successful")
        except Exception as e:
            print(f"❌ Step 5: Prompts import error: {e}")
            raise e
        
        # Step 6: Test agent_logic import
        try:
            from agent_logic import create_tutor_agent
            print("✅ Step 6: Agent logic import successful")
        except Exception as e:
            print(f"❌ Step 6: Agent logic import error: {e}")
            raise e
        
        # Step 7: Create agent
        try:
            print("🔧 Step 7: Creating tutor agent...")
            agent_result = create_tutor_agent(llm, grammar_retriever=None)
            print(f"✅ Step 7: Agent creation returned: {type(agent_result)}")
            print(f"✅ Step 7: Agent keys: {list(agent_result.keys()) if agent_result else 'Empty'}")
        except Exception as e:
            print(f"❌ Step 7: Agent creation error: {e}")
            import traceback
            print(f"❌ Step 7: Full traceback: {traceback.format_exc()}")
            raise e
        
        # Step 8: Update agent chains
        try:
            agent_chains.update(agent_result)
            print(f"✅ Step 8: Agent chains updated: {list(agent_chains.keys())}")
        except Exception as e:
            print(f"❌ Step 8: Agent chains update error: {e}")
            raise e
        
        # Step 9: Final verification
        if "agent" in agent_chains and "curriculum" in agent_chains:
            print("🎉 SUCCESS: Agent ready and all chains loaded!")
        else:
            print(f"⚠️ WARNING: Some chains missing. Available: {list(agent_chains.keys())}")
            
        print("=" * 50)

    except Exception as e:
        print("=" * 50)
        print(f"❌ FATAL: Agent initialization failed at: {e}")
        import traceback
        print(f"❌ FULL ERROR: {traceback.format_exc()}")
        print("=" * 50)
        # We keep agent_chains empty so the app knows the agent is not available.
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

# --- 4. Lessons endpoint ---
CURRICULUM_PATH = "curriculum"

@app.get("/lessons", response_model=LessonsResponse)
async def get_lessons(language: str):
    """Fetches the list of available lessons for a given language."""
    if not language:
        raise HTTPException(status_code=400, detail="Language query parameter is required.")

    language_path = os.path.join(CURRICULUM_PATH, language.lower())
    print(f"🔍 Looking for lessons in: {language_path}")
    
    if not os.path.isdir(language_path):
        print(f"⚠️ Directory not found: {language_path}")
        return {"lessons": []}
    
    try:
        files = [f for f in os.listdir(language_path) if f.startswith("lesson_") and f.endswith(".txt")]
        files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        print(f"✅ Found {len(files)} lesson files: {files}")
        
        lessons = []
        for f in files:
            lesson_num = int(f.split("_")[1].split(".")[0])
            file_path = os.path.join(language_path, f)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    title = file.readline().strip() or f"Lesson {lesson_num}"
            except Exception as e:
                print(f"⚠️ Error reading {f}: {e}")
                title = f"Lesson {lesson_num}"
            lessons.append({"number": lesson_num, "title": title})
            
        print(f"✅ Returning {len(lessons)} lessons")
        return {"lessons": lessons}
    except Exception as e:
        print(f"❌ Error reading lesson files for {language}: {e}")
        raise HTTPException(status_code=500, detail="Could not read lesson files from the server.")

# --- 5. Chat endpoint ---
@app.post("/chat")
async def chat(request: ChatRequest):
    """Handles both curriculum-based and general chat requests."""
    print("=" * 30)
    print(f"🔥 CHAT REQUEST RECEIVED")
    print(f"🔥 lesson_to_teach: {request.lesson_to_teach}")
    print(f"🔥 language: {request.language}")
    print(f"🔥 Available agent chains: {list(agent_chains.keys())}")
    print(f"🔥 Agent chains empty?: {len(agent_chains) == 0}")
    print("=" * 30)
    
    if not agent_chains:
        print("❌ NO AGENT CHAINS AVAILABLE!")
        print("❌ This means agent initialization failed during startup")
        print("❌ Check server startup logs above")
        raise HTTPException(status_code=503, detail="Agent is not available or failed to initialize. Please check server logs and restart server.")

    if request.lesson_to_teach is not None:
        print(f"📚 Teaching lesson {request.lesson_to_teach}")
        chain_to_run = agent_chains.get("curriculum")
        
        if not chain_to_run:
            print("❌ Curriculum chain not found!")
            print(f"❌ Available chains: {list(agent_chains.keys())}")
            raise HTTPException(status_code=503, detail="Curriculum chain is not available.")
        
        if not request.language:
            raise HTTPException(status_code=400, detail="Language is required when teaching a lesson.")
            
        language_folder = request.language.lower()
        lesson_number = request.lesson_to_teach
        file_path = os.path.join(CURRICULUM_PATH, language_folder, f"lesson_{lesson_number}.txt")
        print(f"📖 Looking for lesson file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ Lesson file not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_number} for {request.language} not found.")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lesson_content = f.read()
            print(f"✅ Lesson content loaded: {len(lesson_content)} characters")
        except Exception as e:
            print(f"❌ Error reading lesson file: {e}")
            raise HTTPException(status_code=500, detail=f"Error reading lesson file: {e}")

        agent_input = {
            "context": lesson_content,
            "language": request.language
        }
    else:
        print("💬 General chat request")
        chain_to_run = agent_chains.get("agent")
        
        if not chain_to_run:
            print("❌ General agent chain not found!")
            print(f"❌ Available chains: {list(agent_chains.keys())}")
            raise HTTPException(status_code=503, detail="General agent chain is not available.")
            
        agent_input = {
            "current_question": request.query,
            "previous_query": request.previous_query,
            "previous_response": request.previous_response,
            "language": request.language
        }

    try:
        print(f"🤖 Invoking agent with input keys: {list(agent_input.keys())}")
        response: Any = await chain_to_run.ainvoke(agent_input)
        print(f"✅ Agent response received: {type(response)}")
        
        output = response.get("output", str(response)) if isinstance(response, dict) else str(response)
        print(f"✅ Final output length: {len(output)} characters")
        
        return {"response": output}
    except Exception as e:
        print(f"❌ Error during agent invocation: {e}")
        import traceback
        print(f"❌ Full error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing your request: {e}")

# --- 6. Health check endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify server status."""
    agent_status = "ready" if agent_chains else "not_ready"
    available_chains = list(agent_chains.keys()) if agent_chains else []
    
    return {
        "status": "healthy",
        "agent_status": agent_status,
        "available_chains": available_chains,
        "curriculum_path_exists": os.path.exists(CURRICULUM_PATH),
        "google_api_key_exists": bool(os.getenv("GOOGLE_API_KEY"))
    }

# --- 7. Test endpoint ---
@app.get("/test")
async def test_imports():
    """Test endpoint to check all imports work."""
    results = {}
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        results["langchain_google_genai"] = "✅ SUCCESS"
    except Exception as e:
        results["langchain_google_genai"] = f"❌ ERROR: {e}"
    
    try:
        from config import GOOGLE_API_KEY
        results["config"] = "✅ SUCCESS"
    except Exception as e:
        results["config"] = f"❌ ERROR: {e}"
        
    try:
        from prompts import CURRICULUM_TUTOR_PROMPT
        results["prompts"] = "✅ SUCCESS"
    except Exception as e:
        results["prompts"] = f"❌ ERROR: {e}"
        
    try:
        from agent_logic import create_tutor_agent
        results["agent_logic"] = "✅ SUCCESS"
    except Exception as e:
        results["agent_logic"] = f"❌ ERROR: {e}"
    
    return results

# Add this at the bottom of main.py (replace existing if __name__ section):
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT env var
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
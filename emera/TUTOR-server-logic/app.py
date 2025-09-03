import streamlit as st
import requests
import os
from typing import Optional

# --- CONFIGURATION ---
API_BASE_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT_URL = f"{API_BASE_URL}/chat/stream"
CURRICULUM_DATA_PATH = "./data/curriculum"

# --- HELPER FUNCTIONS ---
def get_available_lessons():
    """Scans the curriculum directory to find available lesson files."""
    if not os.path.exists(CURRICULUM_DATA_PATH):
        return []
    
    lesson_files = [f for f in os.listdir(CURRICULUM_DATA_PATH) if f.startswith("lesson_") and f.endswith(".txt")]
    lesson_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return lesson_files

# --- REFACTORED API CALL LOGIC ---
def handle_chat_submission(query: str, display_query: str = None, lesson_number: Optional[int] = None):
    """
    A single function to handle submitting a query to the backend and streaming the response.
    Can handle both general queries and direct lesson requests.
    """
    ui_message = display_query if display_query is not None else query
    
    st.session_state.messages.append({"role": "user", "content": ui_message})
    with st.chat_message("user"):
        st.markdown(ui_message)

    # --- FIX: Force lesson_number to int ---
    lesson_number_int = int(lesson_number) if lesson_number is not None else None
    
    request_payload = {
        "query": query,
        "previous_query": st.session_state.history.get("previous_query"),
        "previous_response": st.session_state.history.get("previous_response"),
        "lesson_to_teach": lesson_number_int
    }

    with st.chat_message("assistant", avatar="🤖"):
        full_response = ""
        try:
            response = requests.post(CHAT_ENDPOINT_URL, json=request_payload, stream=True, timeout=120)
            response.raise_for_status()
            
            # FIX: Properly handle streaming response
            message_placeholder = st.empty()
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        except requests.exceptions.ConnectionError:
            error_message = f"❌ **Connection Error**: Could not reach the backend server at `{API_BASE_URL}`. Please ensure the backend server is running by executing `python main.py` in a separate terminal."
            st.error(error_message)
            full_response = error_message
        except requests.exceptions.Timeout:
            error_message = "⏰ **Timeout Error**: The request took too long to complete. Please try again."
            st.error(error_message)
            full_response = error_message
        except requests.exceptions.RequestException as e:
            error_message = f"🚨 **Request Error**: {str(e)}"
            st.error(error_message)
            full_response = error_message
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.history = {"previous_query": query, "previous_response": full_response}
    
    if lesson_number_int is not None and lesson_number_int == st.session_state.unlocked_lesson:
        st.session_state.unlocked_lesson += 1

# --- UI SETUP ---
st.set_page_config(page_title="Sanskrit Tutor Bot", page_icon="🧘", layout="wide")
st.title("Sanskrit Tutor Bot 🧘")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = {"previous_query": None, "previous_response": None}
if "unlocked_lesson" not in st.session_state:
    st.session_state.unlocked_lesson = 1
if "lesson_to_start" not in st.session_state:
    st.session_state.lesson_to_start = None

# --- SIDEBAR LOGIC ---
with st.sidebar:
    st.header("Course Index")
    available_lessons = get_available_lessons()
    
    unlocked_options = []
    for lesson_file in available_lessons:
        lesson_num = int(lesson_file.split('_')[1].split('.')[0])
        if lesson_num <= st.session_state.unlocked_lesson:
            unlocked_options.append(lesson_file)

    if not unlocked_options:
        st.warning("No lesson files found, or files are not named 'lesson_X.txt'.")
    else:
        selected_lesson_file = st.selectbox(
            "Choose a lesson:",
            options=unlocked_options,
            format_func=lambda x: f"Lesson {x.split('_')[1].split('.')[0]}"
        )
        
        if st.button("Start Selected Lesson", use_container_width=True):
            st.session_state.lesson_to_start = int(selected_lesson_file.split('_')[1].split('.')[0])
            st.rerun()

# --- MAIN PAGE LOGIC ---
if st.session_state.lesson_to_start is not None:
    lesson_num = st.session_state.lesson_to_start
    st.session_state.lesson_to_start = None 
    
    handle_chat_submission(
        query=f"Direct instruction to teach lesson {lesson_num}",
        display_query=f"*(Starting Lesson {lesson_num})*",
        lesson_number=lesson_num
    )
    st.rerun()

# Display prior messages
st.header("Chat & Practice")
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "user"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Get new user input for general chat
if query := st.chat_input("Ask a grammar question or give me something to translate..."):
    handle_chat_submission(query)
    st.rerun()

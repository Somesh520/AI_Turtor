import os
from typing import Dict

# LangChain Imports
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# THIS LINE FIXES THE ERROR 👇
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, Runnable
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local Imports
from config import *
from prompts import *


def create_rag_retriever(name: str, data_path: str, db_path: str):
    """A generic factory to create a RAG retriever for a specific tool."""
    try:
        if os.path.exists(db_path):
            print(f"✅ Loading existing KB for '{name}'.")
            vectorstore = Chroma(persist_directory=db_path, embedding_function=GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
            ))
            return vectorstore.as_retriever(search_type=RETRIEVER_SEARCH_TYPE, search_kwargs=RETRIEVER_SEARCH_KWARGS)

        print(f"🛠️ Creating new KB for '{name}'.")
        if not os.path.exists(data_path) or not os.listdir(data_path):
            print(f"⚠️ Warning: Data directory for '{name}' is empty ('{data_path}'). Tool disabled.")
            return None
        
        documents = [Document(page_content=open(os.path.join(data_path, f), 'r', encoding='utf-8').read()) 
                     for f in os.listdir(data_path) if f.endswith(".txt")]
        if not documents:
            print(f"⚠️ Warning: No .txt files found for '{name}'. Tool disabled.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=db_path)
        print(f"✅ '{name}' KB is ready.")
        return vectorstore.as_retriever(search_type=RETRIEVER_SEARCH_TYPE, search_kwargs=RETRIEVER_SEARCH_KWARGS)
    
    except Exception as e:
        print(f"❌ Error creating RAG retriever for '{name}': {e}. Tool disabled.")
        return None

# --- AGENT ASSEMBLY ---
def create_tutor_agent(llm: ChatGoogleGenerativeAI, grammar_retriever=None) -> Dict[str, Runnable]:
    """Assembles the complete agent with routing and returns a dictionary of chains."""
    try:
        print("🔧 Creating tutor agent...")
        
        # --- 1. Define Tool Chains ---
        
        # Grammar chain with RAG or fallback
        if grammar_retriever:
            print("✅ Creating grammar chain with RAG retriever")
            grammar_prompt = ChatPromptTemplate.from_template(GRAMMAR_VOCAB_PROMPT)
            grammar_chain = (
                {
                    "context": lambda x: grammar_retriever.invoke(x["current_question"]),
                    "language": lambda x: x["language"],
                    "current_question": lambda x: x["current_question"],
                    "previous_query": lambda x: x["previous_query"],
                    "previous_response": lambda x: x["previous_response"],
                }
                | grammar_prompt | llm | StrOutputParser()
            )
        else:
            print("⚠️ Grammar retriever not available, creating fallback.")
            grammar_chain = (
                ChatPromptTemplate.from_template(
                    "You are a {language} grammar expert. Answer this: {current_question}"
                ) | llm | StrOutputParser()
            )

        print("✅ Creating translator chain")
        translator_chain = (
            ChatPromptTemplate.from_template(TRANSLATOR_PROMPT) | llm | StrOutputParser()
        )
        
        print("✅ Creating conversational chain")
        conversational_chain = (
            ChatPromptTemplate.from_template(CONVERSATIONAL_PROMPT) | llm | StrOutputParser()
        )
        
        print("✅ Creating router chain")
        # Router chain that decides which tool to use
        router_chain = (
            PromptTemplate.from_template(ROUTER_PROMPT) | llm | StrOutputParser()
        )

        # --- 2. Main Agent Branch (for routing general chat) ---
        print("✅ Creating main agent branch")
        main_agent_chain = RunnableBranch(
            (lambda x: "translator" in x.get("tool_choice", "").lower(), translator_chain),
            (lambda x: "grammar" in x.get("tool_choice", "").lower(), grammar_chain),
            conversational_chain  # Default branch
        )

        # The full agent chain that first routes, then executes the chosen tool.
        print("✅ Creating full agent chain")
        full_agent_chain = (
            RunnablePassthrough.assign(tool_choice=router_chain)
            | main_agent_chain
        )

        # --- 3. Curriculum Chain (for teaching specific lessons) ---
        print("✅ Creating curriculum chain")
        curriculum_prompt = ChatPromptTemplate.from_template(CURRICULUM_TUTOR_PROMPT)
        curriculum_chain = curriculum_prompt | llm | StrOutputParser()

        result = {
            "agent": full_agent_chain,
            "curriculum": curriculum_chain
        }
        
        print(f"✅ Agent assembled successfully. Created chains: {list(result.keys())}")
        return result
    
    except Exception as e:
        # This will catch any errors during the agent creation process
        print(f"❌ FATAL: Error creating tutor agent: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return {}
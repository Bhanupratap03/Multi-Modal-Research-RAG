import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 🔑 SETUP
# ==========================================
# PASTE YOUR KEY HERE (Same one you used for image captioning)
GOOGLE_API_KEY = "INSERT_API_KEY"

# Define Paths
DB_PATH = "chroma_db_data"

# ==========================================
# 1. WAKE UP THE BRAIN (Load the DB)
# ==========================================
print("🧠 Loading the Knowledge Base...")

# MUST use the exact same embedding model used to build the DB
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to the existing database
db = Chroma(
    persist_directory = DB_PATH,
    embedding_function = embedding_function
)

# ==========================================
# 2. WAKE UP THE WRITER (Gemini)
# ==========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key = GOOGLE_API_KEY,
    temperature=0.3
)

# ==========================================
# 3. THE CHAT LOOP
# ==========================================
print("\n✅ RAG System Ready! (Type 'quit' to exit)\n")

while True:
    query_text = input("❓ Ask a question about your PDF: ")
    if query_text.lower() == "quit":
        break

    # --- A. SEARCH (Retrieval) ---
    # Search for the 5 most relevant chunks (Text OR Image Captions)
    results = db.similarity_search_with_score(query_text, k=5)

    # Combine the retrieved chunks into one big context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    
    # Debug: Show sources (Optional - helps you trust the system)
    print("\n🔍 Found these relevant sources:")
    for doc, score in results:
        source = doc.metadata.get("source", "Unknown")
        print(f"   - [{source}] (Confidence: {score:.4f})")

    # --- B. ANSWER (Generation) ---
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert researcher assistant. Answer the question based ONLY on the following context:

    {context}
    
    Question: {question}
    """)

    prompt = prompt_template.format(context=context_text, question=query_text)
    
    print("\n🤖 Generating answer...")
    response = llm.invoke(prompt)
    
    print(f"\nAnswer:\n{response.content}\n")
    print("-" * 50)

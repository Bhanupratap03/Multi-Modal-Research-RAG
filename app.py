import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multi-Modal Research RAG", layout="wide")
st.title("🔍 Research Asset Search Engine")
st.markdown("Find diagrams, charts, and logos using **Semantic Intelligence**.")

# --- LOAD THE BRAIN ---
@st.cache_resource  # This saves time by loading the model only once
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db

vector_store = load_vector_db()

# --- SEARCH INTERFACE ---
query = st.text_input("What are you looking for?", placeholder="e.g. A chart showing performance growth")

if query:
    # Perform Search
    results = vector_store.similarity_search(query, k=1)
    
    if results:
        res = results[0]
        st.success(f"Top Match Found!")
        
        # Create two columns: Image on the left, Info on the right
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Get path from metadata
            filename = res.metadata.get("id")
            img_path = os.path.join("extracted_content", filename)
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, caption=res.metadata.get("id"), width="stretch")
            else:
                st.error(f"Image file not found at: {img_path}")
                
        with col2:
            st.subheader("AI Summary")
            st.info(res.page_content)
            st.json(res.metadata) # Good for debugging during your presentation
    else:
        st.warning("No relevant assets found. Try a different query.")
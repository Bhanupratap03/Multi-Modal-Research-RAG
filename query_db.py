from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the exact same embedding model you used to build the DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to your existing database folder
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. Define your search question
query = "A logo featuring a marine predator or sea creature"

# 4. Perform the Semantic Search (k=1 means return the top 1 best match)
results = vector_store.similarity_search(query, k=1)

# 5. Display the winning result!
if results:
    print("🎯 Match Found!")
    print(f"Summary: {results[0].page_content}")
    print(f"File Path: {results[0].metadata['id']}")
else:
    print("No matches found.")
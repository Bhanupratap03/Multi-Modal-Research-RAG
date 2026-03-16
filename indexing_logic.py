from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpointEmbeddings
import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

with open('image_summaries.json', 'r', encoding='utf-8') as i:
    data=json.load(i)
        
load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

doc=[]

for filename, info in data.items():
    document = Document(page_content=info["summary"], metadata={"id":filename})
    doc.append(document)
    
        
vector_store = Chroma.from_documents(
    documents = doc,
    embedding = embeddings,
    persist_directory = "./chroma_db"
) 






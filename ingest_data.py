import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf

# ==========================================
# 🔑 SETUP
# ==========================================
# PASTE YOUR KEY HERE
HUGGINGFACEHUB_ACCESS_TOKEN = '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' 

# Define our paths
PDF_FILE = "DeepSeek-2025.pdf"
CAPTIONS_FOLDER = "extracted_content"
DB_PATH = "chroma_db_data"  # This is where the "Brain" will be saved locally

# ==========================================
# 1. SETUP THE EMBEDDING MODEL
# ==========================================
print("🧠 Initializing the Embedding Model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# 2. LOAD THE TEXT (From PDF)
# ==========================================
print(f"📄 Extracting text from {PDF_FILE}... (This uses the Heavy Duty loader)")

# Force Tesseract Path (Just in case)
tesseract_path = r"C:\Program Files\Tesseract-OCR"
if tesseract_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + tesseract_path

# Run the partition
elements = partition_pdf(
    filename=PDF_FILE,
    strategy="hi_res",       # Essential for preserving table structure
    infer_table_structure=True
)

# Convert raw elements into LangChain Documents
text_docs = []
for el in elements:
    if el.category in ["Table", "CompositeElement"]:
        # For tables, we want the HTML or text representation
        text_docs.append(Document(page_content=el.text, metadata={"source": "Table", "page": el.metadata.page_number}))
    elif el.category in ["Title", "NarrativeText", "ListItem"]:
        text_docs.append(Document(page_content=el.text, metadata={"source": "Text", "page": el.metadata.page_number}))

print(f"   ✅ Extracted {len(text_docs)} text chunks from the PDF.")

# ==========================================
# 3. LOAD THE IMAGES (From Captions)
# ==========================================
print(f"🖼️  Loading image captions from {CAPTIONS_FOLDER}...")

image_docs = []
if os.path.exists(CAPTIONS_FOLDER):
    for filename in os.listdir(CAPTIONS_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(CAPTIONS_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                caption = f.read()
                # Create a Document for the image
                image_docs.append(Document(
                    page_content=caption, 
                    metadata={"source": "Image", "filename": filename}
                ))

print(f"    ✅ Loaded {len(image_docs)} image descriptions.")

# ==========================================
# 4. BUILD THE VECTOR STORE
# ==========================================
all_docs = text_docs + image_docs

print(f"🚀 Creating Vector Database with {len(all_docs)} total items...")

# Create (or overwrite) the database
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)  # Clean start

vector_store = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print(f"\n🎉 SUCCESS! Database saved to '{DB_PATH}'.")
print("   Your RAG system is now ready to answer questions!")
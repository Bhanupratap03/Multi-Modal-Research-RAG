import os
import sys

# ==========================================
# 🔧 THE FIX: Force Tesseract into the PATH
# ==========================================
# This ensures Python can always find the OCR engine
tesseract_path = r"C:\Program Files\Tesseract-OCR"
if tesseract_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + tesseract_path

# Now we can safely import the heavy libraries
from unstructured.partition.pdf import partition_pdf

# ==========================================
# 📂 SETUP: Define Input and Output
# ==========================================
pdf_file = "DeepSeek-2025.pdf"        # Your PDF file name
output_folder = "extracted_content"   # Where we will save the images

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print(f"🚀 Starting Heavy Duty Extraction on '{pdf_file}'...")
print(f"📂 Images and Tables will be saved to: {os.path.abspath(output_folder)}")

try:
    # ==========================================
    # 🧠 THE HEAVY LIFTING
    # ==========================================
    elements = partition_pdf(
        filename=pdf_file,
        strategy="hi_res",                      # Use Tesseract/Poppler (The "Smart" mode)    
        # Image Extraction Settings
        extract_images_in_pdf=True,             # "Yes, rip the images out"
        extract_image_block_types=["Image", "Table"],  # "Get me pictures AND tables"
        extract_image_block_output_dir=output_folder,  # "Put them in this folder"
        
        # Table Settings
        infer_table_structure=True              # "Understand the rows and columns"
    )

    # ==========================================
    # 📊 REPORT CARD
    # ==========================================
    # Count what we found
    tables = [el for el in elements if el.category == "Table"]
    images = [el for el in elements if el.category == "Image"]
    
    print(f"\n✅ DONE! Extraction Complete.")
    print(f"   - Found {len(tables)} Tables")
    print(f"   - Found {len(images)} Images")
    print(f"   - Check the '{output_folder}' folder to see the actual image files!")

except FileNotFoundError:
    print(f"\n❌ Error: Could not find '{pdf_file}'. Is it in the folder?")
except Exception as e:
    print(f"\n❌ Error: {e}")
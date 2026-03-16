import os
import google.generativeai as genai
from PIL import Image

# ==========================================
# 🔑 SETUP
# ==========================================
# PASTE YOUR KEY HERE (Keep this file private!)
API_KEY = "AIzaSyD8SmBucptleHOPt8HnWSvi6lL-HOrUoFM"

genai.configure(api_key=API_KEY)

# We use 1.5 Flash because it is fast and cheap/free for this
model = genai.GenerativeModel('gemini-2.5-flash')

# Folders
input_folder = "extracted_content"

# ==========================================
# 👁️ THE VISION LOOP
# ==========================================
print(f"🚀 Starting Image Captioning in '{input_folder}'...\n")

# Get all images
files = os.listdir(input_folder)
images = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]

if not images:
    print("❌ No images found! Did you run the extraction script?")
    exit()

for img_file in images:
    img_path = os.path.join(input_folder, img_file)
    print(f"   Processing {img_file}...", end=" ", flush=True)
    
    try:
        # 1. Open the image
        img = Image.open(img_path)
        
        # 2. Ask Gemini to describe it
        # We give it a specific prompt for RAG
        prompt = """
        Analyze this image in detail. 
        If it is a chart or graph, describe the x-axis, y-axis, and the key trends or data points.
        If it is a table, summarize the rows and columns.
        If it is a diagram, explain the components and their relationships.
        Provide a concise but comprehensive summary suitable for retrieval.
        """
        
        response = model.generate_content([prompt, img])
        description = response.text
        
        # 3. Save the text
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(input_folder, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(description)
            
        print("✅ Done!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n🎉 All images processed! Check the folder for .txt files.")
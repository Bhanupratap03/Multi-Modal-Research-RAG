import os 
from PIL import Image
from dotenv import load_dotenv
from google import genai
import time
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
IMAGE_DIR = "extracted_content"

results = {}
for images in os.listdir(IMAGE_DIR):
    if images.lower().endswith((".jpg" , ".png")):
        full_path = os.path.join(IMAGE_DIR,images)
        img = Image.open(full_path)
        instruction = "Summarize this chart?"
        response = client.models.generate_content(model="gemini-2.5-flash",contents=[img,instruction])
        print(response)  
        results[images] = {"summary":response.text, "path":full_path}    
    time.sleep(20)
with open("image_summaries.json", "w") as f:
    json.dump(results, f, indent=4)     
 #adding .lower() so file like .JPG don't get missed out
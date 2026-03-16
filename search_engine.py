import json

with open("image_summaries.json", "r")as f:
    data = json.load(f)
    
query="orca"

for filename, value in data.items():
        if query.lower() in value["summary"].lower():
            print(filename)
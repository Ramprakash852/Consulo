# emotion_server.py
from fastapi import FastAPI, Request
import uvicorn
import json
import os
import time

app = FastAPI()

@app.post("/emotion")
async def receive_emotion(request: Request):
    data = await request.json()
    emotion = data.get("emotion", "neutral")
    print(emotion)
    
    # Write emotion to a file that Streamlit can read
    with open("emotion_data.json", "w") as f:
        json.dump({"emotion": emotion, "timestamp": time.time()}, f)
    
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8502)
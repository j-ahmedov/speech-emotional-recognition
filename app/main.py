from fastapi import FastAPI, UploadFile, File
from app.inference import predict_emotion
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


app = FastAPI()

# Mount the static folder to serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Serve index.html at the root
@app.get("/")
async def root():
    index_path = Path("app/static/index.html")
    return FileResponse(index_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    audio_data = await file.read()
    emotion = predict_emotion(audio_data)
    return {"emotion": emotion}

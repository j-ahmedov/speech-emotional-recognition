from fastapi import FastAPI, UploadFile, File
from app.inference import predict_emotion

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    audio_data = await file.read()
    emotion = predict_emotion(audio_data)
    return {"emotion": emotion}

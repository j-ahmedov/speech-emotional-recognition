from fastapi import FastAPI, UploadFile, File, HTTPException
from app.inference import predict_emotion
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import subprocess



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    index_path = Path("app/static/index.html")
    return FileResponse(index_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file bytes (likely webm)
        input_bytes = await file.read()

        # Convert webm to wav bytes with ffmpeg
        process = subprocess.run(
            [
                'ffmpeg',
                '-i', 'pipe:0',      # input from stdin
                '-f', 'wav',         # output format wav
                'pipe:1'             # output to stdout
            ],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        wav_bytes = process.stdout

        # Now use your existing function that expects wav bytes
        emotion = predict_emotion(wav_bytes)

        return {"emotion": emotion}

    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode())
        raise HTTPException(status_code=500, detail="Audio conversion failed")
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))
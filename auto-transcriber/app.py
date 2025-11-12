from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from transcribe_pipeline import process_file, UPLOAD_DIR

app = FastAPI(
    title="Local Transcriber/Translator",
    version="0.1.0",
    description="Upload .mp3/.mp4, get transcript + captions, all local.",
)

# Optional: This allows for cross-origin if you later add a web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    whisper_model: str = Form("small"),
    whisper_lang: str | None = Form(None),
    translate: bool = Form(False),
    marian_model: str | None = Form(None),
):
    """
    Upload an audio/video file and run the local pipeline.
    """
    try:
        # 1) Save uploaded file into UPLOAD_DIR
        upload_path = UPLOAD_DIR / file.filename
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2) Run pipeline
        paths = process_file(
            upload_path,
            whisper_model=whisper_model,
            whisper_lang=whisper_lang or None,
            translate=translate,
            marian_model=marian_model,
        )

        return paths

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

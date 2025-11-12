from pathlib import Path
import whisper

AUDIO_PATH = Path("data/uploads/harvard.wav")  # change name if your file is different
OUTPUT_PATH = Path("data/outputs/harvard_simple.txt")

def main():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio not found at {AUDIO_PATH}")

    print("Loading Whisper model 'small'...")
    model = whisper.load_model("small")

    print(f"Transcribing {AUDIO_PATH} ...")
    result = model.transcribe(str(AUDIO_PATH))

    text = result["text"].strip()
    print("\n=== TRANSCRIPT (first 300 chars) ===")
    print(text[:300])
    print("\n=== END PREVIEW ===")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"\nSaved full transcript to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

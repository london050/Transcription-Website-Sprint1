"""
check_dependencies.py
---------------------
Quick script to verify that all required libraries and system dependencies
for the Auto Transcriber project are correctly installed.

Run:
    python check_dependencies.py
"""

import importlib
import subprocess
import sys
import shutil

# === Libraries to check ===
REQUIRED_LIBRARIES = [
    "torch",
    "transformers",
    "whisper",
    "fastapi",
    "uvicorn",
    "python_multipart",
]

def check_library(lib_name):
    try:
        importlib.import_module(lib_name)
        print(f"✅ {lib_name} imported successfully")
        return True
    except ImportError:
        print(f"❌ {lib_name} not found. Try: pip install {lib_name}")
        return False

def check_ffmpeg():
    print("\nChecking FFmpeg installation...")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ FFmpeg found at: {ffmpeg_path}")
                print(result.stdout.splitlines()[0])
                return True
        except Exception as e:
            print(f"⚠️  Error running FFmpeg: {e}")
    print("❌ FFmpeg not found. Install it before running the app.")
    print("   macOS: brew install ffmpeg")
    print("   Ubuntu: sudo apt install ffmpeg")
    print("   Windows: choco install ffmpeg or add to PATH")
    return False

def check_torch_backend():
    print("\nChecking PyTorch backend...")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else (
                 "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        print(f"✅ Torch working. Device available: {device}")
        return True
    except Exception as e:
        print(f"❌ Torch failed: {e}")
        return False

def check_whisper_model_load():
    print("\nTesting Whisper model load (small)...")
    try:
        import whisper
        model = whisper.load_model("small")
        print("✅ Whisper model loaded successfully")
        del model
        return True
    except Exception as e:
        print(f"⚠️  Whisper model failed to load: {e}")
        print("   Check internet connection or Whisper install.")
        return False

def main():
    print("=== Auto Transcriber Environment Check ===\n")

    ok_libs = all(check_library(lib) for lib in REQUIRED_LIBRARIES)
    ok_ffmpeg = check_ffmpeg()
    ok_torch = check_torch_backend()
    ok_whisper = check_whisper_model_load()

    print("\n=== Summary ===")
    if all([ok_libs, ok_ffmpeg, ok_torch, ok_whisper]):
        print("🎉 Environment looks good! You're ready to run the project.")
    else:
        print("⚠️  One or more checks failed. Review the messages above and fix issues before running the app.")

if __name__ == "__main__":
    main()

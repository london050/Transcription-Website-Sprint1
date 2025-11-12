import json
import subprocess
import uuid
from pathlib import Path
from datetime import timedelta
from typing import Optional, List, Dict

import whisper
import torch
from transformers import MarianMTModel, MarianTokenizer

# ---------- Paths ----------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- FFmpeg: convert to 16kHz mono WAV ----------

def run_ffmpeg_to_wav(input_path: Path) -> Path:
    """
    Convert input audio/video to 16kHz mono WAV using ffmpeg.
    """
    output_name = f"{input_path.stem}_{uuid.uuid4().hex[:8]}.wav"
    output_path = OUTPUT_DIR / output_name

    cmd = [
        "ffmpeg",
        "-y",                # overwrite existing
        "-i", str(input_path),
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        str(output_path),
    ]

    subprocess.run(cmd, check=True)
    return output_path


# ---------- Helpers: timestamp formatting ----------

def seconds_to_srt_ts(t: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    td = timedelta(seconds=float(t))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def seconds_to_vtt_ts(t: float) -> str:
    """Convert seconds to VTT timestamp: HH:MM:SS.mmm"""
    td = timedelta(seconds=float(t))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


# ---------- Whisper: local transcription ----------

def run_whisper(
    wav_path: Path,
    model_name: str = "small",
    language: Optional[str] = None,
) -> (List[Dict,], str):
    """
    Run Whisper locally and return (segments, full_text).
    Each segment: {id, start, end, text}.
    """
    model = whisper.load_model(model_name)  # uses GPU if available
    result = model.transcribe(str(wav_path), language=language)

    segments = []
    for seg in result["segments"]:
        segments.append(
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
        )

    full_text = result["text"].strip()
    return segments, full_text


# ---------- MarianMT: local translation ----------

class Translator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate_texts(self, texts: List[str], max_length: int = 512) -> List[str]:
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            generated = self.model.generate(**batch, max_length=max_length)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


def translate_segments(
    segments: List[Dict],
    model_name: str,
) -> (List[Dict], str):
    """
    Translate segment texts and return (translated_segments, full_translated_text).
    """
    translator = Translator(model_name)
    texts = [s["text"] for s in segments]
    translations = translator.translate_texts(texts)

    translated_segments = []
    for seg, tr in zip(segments, translations):
        new_seg = dict(seg)
        new_seg["translated_text"] = tr.strip()
        translated_segments.append(new_seg)

    full_translated = "\n".join(s["translated_text"] for s in translated_segments)
    return translated_segments, full_translated


# ---------- Output formatters ----------

def write_transcript_txt(full_text: str, base_name: str) -> Path:
    path = OUTPUT_DIR / f"{base_name}.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(full_text)
    return path


def write_srt(
    segments: List[Dict],
    base_name: str,
    use_translated: bool = False,
) -> Path:
    path = OUTPUT_DIR / f"{base_name}.srt"
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start_ts = seconds_to_srt_ts(seg["start"])
            end_ts = seconds_to_srt_ts(seg["end"])
            text_field = "translated_text" if use_translated and "translated_text" in seg else "text"
            text = seg[text_field].strip()

            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n\n")
    return path


def write_vtt(
    segments: List[Dict],
    base_name: str,
    use_translated: bool = False,
) -> Path:
    path = OUTPUT_DIR / f"{base_name}.vtt"
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start_ts = seconds_to_vtt_ts(seg["start"])
            end_ts = seconds_to_vtt_ts(seg["end"])
            text_field = "translated_text" if use_translated and "translated_text" in seg else "text"
            text = seg[text_field].strip()

            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n\n")
    return path


def write_segments_json(
    segments: List[Dict],
    base_name: str,
) -> Path:
    path = OUTPUT_DIR / f"{base_name}_segments.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    return path


# ---------- Main pipeline ----------

def process_file(
    input_path: Path,
    whisper_model: str = "small",
    whisper_lang: Optional[str] = None,
    translate: bool = False,
    marian_model: Optional[str] = None,
) -> Dict[str, str]:
    """
    Full pipeline:
    input -> ffmpeg (wav) -> Whisper -> optional Marian -> txt, srt, vtt, segments.json.
    Returns dict of file paths as strings.
    """
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 1) ffmpeg -> wav
    wav_path = run_ffmpeg_to_wav(input_path)

    # 2) whisper
    segments, full_text = run_whisper(
        wav_path,
        model_name=whisper_model,
        language=whisper_lang,
    )

    # 3) optional translate
    translated_segments = None
    full_translated = None
    if translate:
        if not marian_model:
            raise ValueError("marian_model must be provided if translate=True.")
        translated_segments, full_translated = translate_segments(segments, marian_model)

    caption_segments = translated_segments if translated_segments is not None else segments
    base_name = input_path.stem

    # 4) write outputs
    main_text = full_translated if full_translated is not None else full_text
    txt_path = write_transcript_txt(main_text, base_name)
    srt_path = write_srt(caption_segments, base_name, use_translated=translate)
    vtt_path = write_vtt(caption_segments, base_name, use_translated=translate)
    segjson_path = write_segments_json(
        translated_segments if translated_segments is not None else segments,
        base_name,
    )

    return {
        "transcript_txt": str(txt_path),
        "captions_srt": str(srt_path),
        "captions_vtt": str(vtt_path),
        "segments_json": str(segjson_path),
        "wav_intermediate": str(wav_path),
    }

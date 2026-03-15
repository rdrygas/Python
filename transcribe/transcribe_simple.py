# Prosty przykład transkrypcji z użyciem faster-whisper

from faster_whisper import WhisperModel
from pathlib import Path

AUDIO = Path("/home/robert/nagranie.mp3")  # ścieżka do pliku audio (mp3, wav, flac, m4a, itp.)
MODEL_NAME = "turbo"   # "turbo" albo "large-v3"
COMPUTE_TYPE = "float16"   # przy braku VRAM: "int8_float16"

# Funkcja do formatowania timestampów w formacie SRT
def srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    if secs == 60:
        minutes += 1
        secs = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# Inicjalizacja modelu
model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE)

# Transkrypcja
segments, info = model.transcribe(
    str(AUDIO),
    language="pl",
    beam_size=5,
    vad_filter=True,
    word_timestamps=False,
)

# Zapisywanie transkrypcji
segments = list(segments)

# Zapisywanie do pliku TXT i SRT
txt_path = AUDIO.with_suffix(".txt")
srt_path = AUDIO.with_suffix(".srt")

# Zapisywanie do pliku TXT
with txt_path.open("w", encoding="utf-8") as f:
    for seg in segments:
        f.write(seg.text.strip() + "\n")

# Zapisywanie do pliku SRT
with srt_path.open("w", encoding="utf-8") as f:
    for i, seg in enumerate(segments, start=1):
        f.write(f"{i}\n")
        f.write(f"{srt_timestamp(seg.start)} --> {srt_timestamp(seg.end)}\n")
        f.write(seg.text.strip() + "\n\n")

# Informacje o transkrypcji
print(f"Język: {info.language} (p={info.language_probability:.3f})")
print(f"TXT: {txt_path}")
print(f"SRT: {srt_path}")
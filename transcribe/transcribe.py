#        NAME: Transkrypcja pliku audio lub wideo
# DESCRIPTION: Skrypt używa modelu faster-whisper do transkrypcji pliku audio lub wideo, a następnie zapisuje wyniki w formacie TXT i SRT. 
#              Skrypt jest zoptymalizowany pod kątem czytelności napisów, dzieląc tekst na bloki z uwzględnieniem długości, czasu trwania i interpunkcji. 
#              Pasek postępu tqdm pokazuje postęp transkrypcji w czasie rzeczywistym.
#      AUTHOR: Robert Drygas / ChatGPT
#     VERSION: 1.6.0
#     CREATED: 2026-03-14
#    MODIFIED: 2026-03-21
#
# DEPENDENCIES:
#
# TESTED ON:
#    - OS Windows 11 + WSL2 (Ubuntu 24.04, Python 3.14) + GPU NVIDIA GeForce RTX 3060 + CPU Intel Core i7 11700
#
# USAGE:
#    $ python3 transcribe.py <filename> [--style <style>] [--language <lang_code>] [--mux-mkv-srt]
#
# ARGUMENTS:
#    <filename> - ścieżka do pliku audio lub wideo (obowiązkowe)
#    --style    - styl napisów: 'reading' daje dłuższe, wygodniejsze bloki, a 'film' tworzy krótsze, bardziej klasyczne SRT (opcjonalne, domyślnie 'reading')
#    --language - kod języka (ISO 639-1), np. 'pl' dla polskiego, 'en' dla angielskiego lub None dla automatycznej detekcji (opcjonalne, domyślnie 'pl')
#    --mux-mkv-srt - dołącza wygenerowany plik SRT do nowego pliku MKV po wcześniejszej weryfikacji kontenera (opcjonalne)
#
# EXAMPLES:
#    $ python3 transcribe.py nagranie.mp3
#    $ python3 transcribe.py nagranie.mkv
#    $ python3 transcribe.py nagranie.mp4 --style film
#    $ python3 transcribe.py nagranie.mp4 --language en
#    $ python3 transcribe.py nagranie.mp4 --style film --language en
#    $ python3 transcribe.py nagranie.mkv --mux-mkv-srt
#
# CHANGELOG:
#    - 1.0.0 (2026-03-14) Pierwsza wersja
#    - 1.1.0 (2026-03-17) Dodano style napisów
#    - 1.1.1 (2026-03-18) Dodano komentarze i poprawki w dokumentacji
#    - 1.2.0 (2026-03-18) Dodano obsługę argumentu języka transkrypcji
#    - 1.2.1 (2026-03-18) Dodano obsługę przerwania klawiaturą (Ctrl+C) i poprawki w komunikatach o błędach
#    - 1.3.0 (2026-03-19) Dodano komunikaty etapów wykonywania skryptu (funkcja stage)
#    - 1.4.0 (2026-03-19) Dodano wykrywanie cache modelu i komunikaty o jego statusie
#    - 1.4.1 (2026-03-19) Dodano sprawdzanie, czy wybrany model jest wspierany
#    - 1.5.0 (2026-03-19) Dodano weryfikację MKV i opcję dołączania napisów SRT do nowego pliku MKV
#    - 1.6.0 (2026-03-21) Dodano weryfikację CUDA i informację o dostępności GPU/CPU dla modelu
#
# ROADMAP:
#    - [*] Style napisów
#    - [*] Komunikaty etapów wykonywania skryptu
#    - [*] Wykrywanie cache modelu
#    - [*] Połączenie napisów z nagraniem wideo MKV
#
# KNOWN ISSUES:
#
# LICENSE: GPL-3.0


from __future__ import annotations

import argparse
import sys
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from faster_whisper import WhisperModel
from tqdm import tqdm


# KONFIGURACJA MODELU
MODEL_NAME = "large-v3"          # Wybór modelu: "turbo" (szybki), "large-v3" (dokładniejszy)
COMPUTE_TYPE = "float16"      # Typ obliczeń. Dla starszych kart lub CPU zmień na "int8_float16" lub "int8"
LANGUAGE = "pl"               # Kod języka (ISO 639-1). None = automatyczna detekcja
BEAM_SIZE = 5                 # Szerokość poszukiwania (wyższa = lepsza dokładność, ale wolniej)
VAD_FILTER = True             # Voice Activity Detection - pomija ciszę, zmniejsza halucynacje modelu
VAD_PARAMETERS = {"min_silence_duration_ms": 3000}  # Parametry wykrywania ciszy (w tym przypadku: minimum 3 sekundy ciszy, aby rozdzielić napisy)
CONDITION_ON_PREVIOUS_TEXT = False  # Czy model ma brać pod uwagę poprzednie zdania (False zapobiega pętlom/powtórzeniom)

# PARAMETRY FORMATOWANIA NAPISÓW - będą nadpisywane przez preset stylu
MAX_LINE_LENGTH = 42            # Maksymalna liczba znaków w jednej linii napisu
MAX_LINES = 2                   # Maksymalna liczba linii wyświetlanych naraz (zazwyczaj 1-2)
MAX_BLOCK_CHARS = MAX_LINE_LENGTH * MAX_LINES
MAX_BLOCK_DURATION = 6.0        # Maksymalny czas trwania jednego napisu (s)
MIN_BLOCK_CHARS = 18            # Minimalna długość, poniżej której skrypt próbuje łączyć napisy
MIN_BLOCK_DURATION = 1.2        # Minimalny czas trwania napisu na ekranie (s)
MAX_JOIN_GAP = 1.0              # Maksymalna przerwa (s) między wypowiedziami, pozwalająca na ich scalenie

STRONG_PUNCT = ".?!"            # Znaki kończące myśl (wymuszają podział bloku)
SOFT_PUNCT = ",;:)]}"           # Znaki sugerujące naturalną pauzę (dobre miejsce na podział)


@dataclass(frozen=True, slots=True)
class StylePreset:
    """Reprezentuje zestaw parametrów formatowania napisów dla różnych stylów"""
    max_line_length: int
    max_lines: int
    max_block_duration: float
    min_block_chars: int
    min_block_duration: float
    max_join_gap: float


# Predefiniowane style napisów
STYLE_PRESETS = {
    "reading": StylePreset(
        max_line_length=42,
        max_lines=2,
        max_block_duration=6.0,
        min_block_chars=18,
        min_block_duration=1.2,
        max_join_gap=1.0,
    ),
    "film": StylePreset(
        max_line_length=36,
        max_lines=2,
        max_block_duration=4.5,
        min_block_chars=10,
        min_block_duration=0.9,
        max_join_gap=0.55,
    ),
}

# Wspierane modele faster-whisper
VALID_MODEL_NAMES = {
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-medium.en",
    "distil-small.en",
    "distil-large-v3",
    "distil-large-v3.5",
    "large-v3-turbo",
    "turbo",
}

REQUIRED_VIDEO_TOOLS = ("ffprobe", "ffmpeg", "mkvmerge", "mkvinfo", "mediainfo")

@dataclass(slots=True)
class WordToken:
    """Reprezentuje pojedyncze słowo wraz z jego czasem rozpoczęcia i zakończenia."""
    start: float
    end: float
    text: str


@dataclass(slots=True)
class SubtitleBlock:
    """Reprezentuje pełny blok napisu składający się z jednego lub wielu słów."""
    words: list[WordToken]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def plain_text(self) -> str:
        """Zwraca tekst bloku w jednej linii (do pliku TXT)."""
        return render_words(self.words)

    @property
    def text_for_srt(self) -> str:
        """Zwraca tekst sformatowany pod SRT (z zawijaniem linii)."""
        return wrap_block_text(self.words)


def apply_style_preset(style_name: str) -> StylePreset:
    """Ustawia globalne parametry formatowania napisów na podstawie wybranego stylu."""
    global MAX_LINE_LENGTH, MAX_LINES, MAX_BLOCK_CHARS
    global MAX_BLOCK_DURATION, MIN_BLOCK_CHARS, MIN_BLOCK_DURATION, MAX_JOIN_GAP

    preset = STYLE_PRESETS[style_name]
    MAX_LINE_LENGTH = preset.max_line_length
    MAX_LINES = preset.max_lines
    MAX_BLOCK_CHARS = MAX_LINE_LENGTH * MAX_LINES
    MAX_BLOCK_DURATION = preset.max_block_duration
    MIN_BLOCK_CHARS = preset.min_block_chars
    MIN_BLOCK_DURATION = preset.min_block_duration
    MAX_JOIN_GAP = preset.max_join_gap
    return preset


def parse_args() -> argparse.Namespace:
    """Obsługa argumentów wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Transcription of audio/video files to TXT and SRT formats, featuring a progress bar, word timestamps and intelligent subtitle formatting.",
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio or video file, e.g., recording.mp3 or movie.mp4.",
    )
    parser.add_argument(
        "-s", "--style",
        choices=sorted(STYLE_PRESETS.keys()),
        default="reading",
        help="Subtitle style: 'reading' gives longer, more comfortable blocks, while 'film' creates shorter, more classic SRT (default: 'reading').",
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="pl",
        help="Language of the audio for transcription (ISO 639-1 code) or None for automatic detection (default: 'pl')."
    )
    parser.add_argument(
        "--mux-mkv-srt",
        action="store_true",
        help="If the input file is MKV, verify it with ffprobe/mkvinfo/mediainfo and attach the generated SRT subtitle track to a new MKV file.",
    )

    return parser.parse_args()


def stage(message: str) -> None:
    print(f"[STAGE] {message}", flush=True)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Uruchamia polecenie systemowe i zwraca wynik wraz ze stdout/stderr jako tekst."""
    return subprocess.run(command, capture_output=True, text=True, check=False)


def get_available_media_tools() -> dict[str, str]:
    """Zwraca mapę dostępnych narzędzi multimedialnych i ich ścieżek w systemie."""
    available: dict[str, str] = {}
    for tool_name in REQUIRED_VIDEO_TOOLS:
        tool_path = shutil.which(tool_name)
        if tool_path:
            available[tool_name] = tool_path
    return available


def verify_required_media_tools() -> dict[str, str]:
    """Sprawdza dostępność wymaganych narzędzi do weryfikacji i muxowania MKV."""
    available = get_available_media_tools()
    missing = [tool_name for tool_name in REQUIRED_VIDEO_TOOLS if tool_name not in available]
    if missing:
        raise SystemExit(
            "Error: missing required media tools for MKV verification/muxing: "
            + ", ".join(missing)
        )
    return available


def is_mkv_extension(path: Path) -> bool:
    """Sprawdza rozszerzenie pliku wejściowego."""
    return path.suffix.lower() == ".mkv"


def verify_mkv_with_ffprobe(video_path: Path, tools: dict[str, str]) -> None:
    """Potwierdza kontener Matroska przy użyciu ffprobe."""
    result = run_command(
        [
            tools["ffprobe"],
            "-v",
            "error",
            "-show_entries",
            "format=format_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    if result.returncode != 0:
        raise SystemExit(f"Error: ffprobe failed to inspect '{video_path}': {result.stderr.strip()}")

    format_name = result.stdout.strip().lower()
    if "matroska" not in format_name:
        raise SystemExit(
            f"Error: ffprobe reports that '{video_path}' is not an MKV/Matroska container: {format_name or 'unknown'}"
        )


def verify_mkv_with_mkvinfo(video_path: Path, tools: dict[str, str]) -> None:
    """Potwierdza, że mkvinfo potrafi odczytać strukturę kontenera Matroska."""
    result = run_command([tools["mkvinfo"], str(video_path)])
    if result.returncode != 0:
        raise SystemExit(f"Error: mkvinfo failed to inspect '{video_path}': {result.stderr.strip()}")


def verify_mkv_with_mediainfo(video_path: Path, tools: dict[str, str]) -> None:
    """Potwierdza format kontenera przy użyciu mediainfo."""
    result = run_command([tools["mediainfo"], "--Inform=General;%Format%", str(video_path)])
    if result.returncode != 0:
        raise SystemExit(f"Error: mediainfo failed to inspect '{video_path}': {result.stderr.strip()}")

    reported_format = result.stdout.strip().lower()
    if "matroska" not in reported_format:
        raise SystemExit(
            f"Error: mediainfo reports that '{video_path}' is not Matroska: {reported_format or 'unknown'}"
        )


def verify_mkv_video(video_path: Path, tools: dict[str, str]) -> None:
    """Weryfikuje, czy plik jest rzeczywiście kontenerem MKV, używając kilku narzędzi."""
    if not is_mkv_extension(video_path):
        raise SystemExit(
            f"Error: subtitle muxing is only supported for MKV input files. Got '{video_path.name}'."
        )

    verify_mkv_with_ffprobe(video_path, tools)
    verify_mkv_with_mkvinfo(video_path, tools)
    verify_mkv_with_mediainfo(video_path, tools)


def build_muxed_mkv_path(video_path: Path) -> Path:
    """Zwraca ścieżkę pliku wynikowego MKV z dołączonymi napisami."""
    return video_path.with_name(f"{video_path.stem}.subtitled.mkv")


def mux_subtitles_with_mkvmerge(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    language: str | None,
    tools: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Dołącza plik SRT do MKV za pomocą mkvmerge bez rekompresji strumieni."""
    subtitle_language = language or "und"
    return run_command(
        [
            tools["mkvmerge"],
            "-o",
            str(output_path),
            str(video_path),
            "--language",
            f"0:{subtitle_language}",
            "--track-name",
            "0:Transcription",
            str(srt_path),
        ]
    )


def mux_subtitles_with_ffmpeg(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    language: str | None,
    tools: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Fallback: dołącza plik SRT do MKV przy użyciu ffmpeg."""
    subtitle_language = language or "und"
    return run_command(
        [
            tools["ffmpeg"],
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(srt_path),
            "-map",
            "0",
            "-map",
            "1:0",
            "-c",
            "copy",
            "-c:s",
            "srt",
            "-metadata:s:s:0",
            f"language={subtitle_language}",
            "-metadata:s:s:0",
            "title=Transcription",
            str(output_path),
        ]
    )


def mux_srt_with_mkv(video_path: Path, srt_path: Path, language: str | None) -> Path:
    """Weryfikuje MKV i dołącza do niego wygenerowane napisy SRT."""
    tools = verify_required_media_tools()
    verify_mkv_video(video_path, tools)

    output_path = build_muxed_mkv_path(video_path)

    stage("Attaching SRT subtitles to MKV with mkvmerge")
    mkvmerge_result = mux_subtitles_with_mkvmerge(video_path, srt_path, output_path, language, tools)
    if mkvmerge_result.returncode == 0:
        return output_path

    if output_path.exists():
        output_path.unlink()

    stage("mkvmerge failed, retrying subtitle attachment with ffmpeg")
    ffmpeg_result = mux_subtitles_with_ffmpeg(video_path, srt_path, output_path, language, tools)
    if ffmpeg_result.returncode == 0:
        return output_path

    if output_path.exists():
        output_path.unlink()

    raise SystemExit(
        "Error: failed to attach SRT subtitles to MKV. "
        f"mkvmerge: {mkvmerge_result.stderr.strip() or mkvmerge_result.stdout.strip()} | "
        f"ffmpeg: {ffmpeg_result.stderr.strip() or ffmpeg_result.stdout.strip()}"
    )


def get_hf_cache_root() -> Path:
    """Zwraca ścieżkę do katalogu cache HuggingFace, uwzględniając różne możliwe zmienne środowiskowe i standardowe lokalizacje."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def detect_model_cache(model_name: str) -> tuple[bool, Path | None]:
    """Sprawdza, czy model jest już pobrany w lokalnym cache HuggingFace. Zwraca krotkę (znaleziono, ścieżka)."""
    if Path(model_name).exists():
        return True, Path(model_name)

    cache_root = get_hf_cache_root()
    if not cache_root.exists():
        return False, cache_root

    normalized = model_name.lower().replace("/", "--")
    needles = {model_name.lower(), normalized}

    try:
        for p in cache_root.rglob("*"):
            name = p.name.lower()
            if any(n in name for n in needles):
                return True, cache_root
    except Exception:
        return False, cache_root

    return False, cache_root


def srt_timestamp(seconds: float) -> str:
    """Konwertuje sekundy na format czasu SRT: HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    # Korekta zaokrągleń (np. 999.5 ms -> 1000 ms -> +1 sekunda)
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


def get_media_duration_seconds(path: Path) -> float | None:
    """Odczytuje całkowity czas trwania pliku przy użyciu biblioteki PyAV."""
    with av.open(str(path)) as container:
        if container.duration is not None:
            return float(container.duration / av.time_base)
        
        # Próba odczytu ze strumieni, jeśli kontener nie posiada metadanych czasu
        for stream in container.streams:
            if stream.type in ("audio", "video") and stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)

    return None


def create_progress_bar(total_duration: float | None) -> tqdm:
    """Tworzy pasek postępu tqdm dostosowany do czasu trwania pliku."""
    common_kwargs = {
        "desc": "Transcribing",
        "dynamic_ncols": True,
        "leave": True,
    }

    if total_duration and total_duration > 0:
        return tqdm(
            total=total_duration,
            unit="s",
            # Dostosowany format paska, pokazujący czas w sekundach i szacowany czas pozostały
            bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]",
            **common_kwargs,
        )

    # Jeśli nie znamy czasu, liczymy po prostu przetworzone segmenty
    return tqdm(unit="seg", **common_kwargs)


def render_words(words: list[WordToken]) -> str:
    """Łączy listę słów w jeden ciąg tekstowy, czyszcząc zbędne spacje."""
    text = "".join(word.text for word in words).strip()
    return " ".join(text.split())


def ends_with_punctuation(text: str, punctuation: str) -> bool:
    """Sprawdza, czy tekst kończy się wybranym znakiem interpunkcyjnym (ignoruje cudzysłowy)."""
    stripped = text.rstrip('"”’» ')
    return bool(stripped) and stripped[-1] in punctuation


def should_break_after(current_words: list[WordToken], next_word: WordToken | None) -> bool:
    """
    Logika decydująca o tym, czy zakończyć obecny napis i zacząć nowy.
    Bierze pod uwagę: długość tekstu, czas trwania, interpunkcję i przerwy w mowie.
    """
    if not current_words:
        return False

    current_text = render_words(current_words)
    current_duration = current_words[-1].end - current_words[0].start

    # 1. Zbyt długi tekst lub zbyt długi czas wyświetlania
    if len(current_text) >= MAX_BLOCK_CHARS:
        return True
    if current_duration >= MAX_BLOCK_DURATION:
        return True

    # 2. Ostatnie słowo w nagraniu
    if next_word is None:
        return True

    # 3. Wykryta długa pauza w mowie (cisza)
    gap_after = max(0.0, next_word.start - current_words[-1].end)
    if gap_after >= MAX_JOIN_GAP:
        return True

    # 4. Dodanie następnego słowa przekroczyłoby limit znaków
    if len(render_words(current_words + [next_word])) > MAX_BLOCK_CHARS:
        return True

    # 5. Silna interpunkcja (kropka, wykrzyknik) - naturalne zakończenie bloku
    if ends_with_punctuation(current_text, STRONG_PUNCT):
        return True

    # 6. Słaba interpunkcja (przecinek) - może być dobrym miejscem na podział, ale tylko jeśli blok jest już dość długi
    punct_threshold = 0.45 if MAX_LINE_LENGTH <= 36 else 0.55
    if ends_with_punctuation(current_text, SOFT_PUNCT) and len(current_text) >= int(MAX_BLOCK_CHARS * punct_threshold):
        return True

    return False


def words_to_initial_blocks(words: list[WordToken]) -> list[SubtitleBlock]:
    """Wstępny podział listy słów na bloki napisów na podstawie logiki logicznej."""
    if not words:
        return []

    blocks: list[SubtitleBlock] = []
    current: list[WordToken] = []

    for index, word in enumerate(words):
        # Sprawdzanie przerw przed dodaniem słowa
        if current:
            gap_before = max(0.0, word.start - current[-1].end)
            if gap_before >= MAX_JOIN_GAP:
                blocks.append(SubtitleBlock(words=current))
                current = []

        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None

        # Decyzja o zamknięciu bloku
        if should_break_after(current, next_word):
            blocks.append(SubtitleBlock(words=current))
            current = []

    if current:
        blocks.append(SubtitleBlock(words=current))

    return blocks


def is_short_block(block: SubtitleBlock) -> bool:
    """Sprawdza, czy blok jest 'zbyt krótki' (wymaga ewentualnego scalenia)."""
    return (
        len(block.plain_text) < MIN_BLOCK_CHARS
        or block.duration < MIN_BLOCK_DURATION
        or len(block.words) <= 2
    )


def can_merge_blocks(left: SubtitleBlock, right: SubtitleBlock) -> bool:
    """Sprawdza, czy dwa sąsiednie bloki można bezpiecznie połączyć w jeden."""
    gap = max(0.0, right.start - left.end)
    if gap > MAX_JOIN_GAP:
        return False

    merged_words = left.words + right.words
    merged_text = render_words(merged_words)
    merged_duration = right.end - left.start

    # Nie łączymy, jeśli przekroczymy limity długości lub czasu trwania bloku
    if len(merged_text) > MAX_BLOCK_CHARS:
        return False
    if merged_duration > MAX_BLOCK_DURATION:
        return False

    return True


def merge_short_blocks(blocks: list[SubtitleBlock]) -> list[SubtitleBlock]:
    """
    Próbuje scalić krótkie bloki z ich sąsiadami, aby napisy nie 'migały' na ekranie.
    Wykonuje do 3 przebiegów, aby zoptymalizować strukturę.
    """
    if not blocks:
        return []

    merged = blocks[:]

    for _ in range(3):
        result: list[SubtitleBlock] = []
        i = 0
        changed = False

        while i < len(merged):
            current = merged[i]

            if is_short_block(current):
                # Próbuj połączyć z poprzednim
                if result and can_merge_blocks(result[-1], current):
                    previous = result.pop()
                    result.append(SubtitleBlock(words=previous.words + current.words))
                    changed = True
                    i += 1
                    continue

                # Próbuj połączyć z następnym
                if i + 1 < len(merged) and can_merge_blocks(current, merged[i + 1]):
                    next_block = merged[i + 1]
                    result.append(SubtitleBlock(words=current.words + next_block.words))
                    changed = True
                    i += 2
                    continue

            result.append(current)
            i += 1

        merged = result
        if not changed:
            break

    return merged


def choose_line_split_index(words: list[WordToken]) -> int | None:
    """
    Znajduje optymalne miejsce do przełamania tekstu na dwie linie (Enter).
    Dąży do równej długości linii i unikania łamania w środku fraz.
    """
    if len(words) < 2:
        return None

    best_index: int | None = None
    best_score: tuple[float, float, float] | None = None

    for idx in range(1, len(words)):
        left = render_words(words[:idx])
        right = render_words(words[idx:])

        left_len = len(left)
        right_len = len(right)
        # Koszt: kara za przekroczenie limitu znaków w linii
        overflow = max(0, left_len - MAX_LINE_LENGTH) + max(0, right_len - MAX_LINE_LENGTH)
        # Koszt: kara za nierówne linie (estetyka)
        imbalance = abs(left_len - right_len)
        max_len = max(left_len, right_len)

        # Bonus: premiujemy łamanie na znakach interpunkcyjnych, zwłaszcza silnych (kropka, wykrzyknik)
        punctuation_bonus = 0.0
        if ends_with_punctuation(left, STRONG_PUNCT):
            punctuation_bonus = -8.0
        elif ends_with_punctuation(left, SOFT_PUNCT):
            punctuation_bonus = -4.0

        # Wynik (im niższy, tym lepiej)
        score = (overflow, max_len + punctuation_bonus, imbalance)

        if best_score is None or score < best_score:
            best_score = score
            best_index = idx

    if best_index is None:
        return None

    left = render_words(words[:best_index])
    right = render_words(words[best_index:])
    if len(left) <= MAX_LINE_LENGTH and len(right) <= MAX_LINE_LENGTH:
        return best_index

    # Jeśli nie dało się zmieścić obu linii w limicie, zwracamy najlepsze wymuszone cięcie.
    return best_index


def wrap_block_text(words: list[WordToken]) -> str:
    """Zawija tekst bloku w dwie linie, jeśli przekracza limit MAX_LINE_LENGTH."""
    text = render_words(words)
    if len(text) <= MAX_LINE_LENGTH:
        return text

    split_index = choose_line_split_index(words)
    if split_index is None:
        return text

    line1 = render_words(words[:split_index])
    line2 = render_words(words[split_index:])
    return f"{line1}\n{line2}"


def build_subtitle_blocks(words: list[WordToken]) -> list[SubtitleBlock]:
    """Główna funkcja orkiestrująca budowę napisów z surowych słów."""
    initial_blocks = words_to_initial_blocks(words)
    return merge_short_blocks(initial_blocks)


def collect_words_and_metadata(
    segments,
    total_duration: float | None,
    pbar: tqdm,
) -> tuple[list[WordToken], object]:
    """
    Iteruje po segmentach zwracanych przez WhisperModel, aktualizuje pasek postępu
    i zbiera wszystkie słowa wraz z ich metadanymi.
    """
    all_words: list[WordToken] = []
    last_progress_seconds = 0.0
    info = None
    segment_count = 0

    for segment_count, seg in enumerate(segments, start=1):
        # Aktualizacja paska postępu na podstawie czasu trwania segmentu
        if total_duration and total_duration > 0:
            current_progress_seconds = min(float(seg.end), total_duration)
            delta = max(0.0, current_progress_seconds - last_progress_seconds)
            if delta:
                pbar.update(delta)
            last_progress_seconds = current_progress_seconds
            pbar.set_postfix(segment=segment_count, refresh=False)
        else:
            pbar.update(1)
            pbar.set_postfix(segment=segment_count, refresh=False)

        # Pobranie słów z segmentu (wymaga word_timestamps=True w modelu)
        words = getattr(seg, "words", None) or []
        for word in words:
            if word.start is None or word.end is None:
                continue
            token_text = word.word or ""
            if not token_text.strip():
                continue
            all_words.append(WordToken(start=float(word.start), end=float(word.end), text=token_text))

        # Fallback: jeśli z jakiegoś powodu brak słów, bierzemy cały segment jako jeden 'token'
        if not words and seg.text.strip():
            all_words.append(
                WordToken(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=f" {seg.text.strip()}",
                )
            )

        info = getattr(seg, "_info", info)

    # Dopełnienie paska na koniec
    if total_duration and total_duration > 0 and last_progress_seconds < total_duration:
        pbar.update(total_duration - last_progress_seconds)

    pbar.set_postfix(segment=segment_count, words=len(all_words), refresh=False)
    return all_words, info


def set_language(lang_code: str | None) -> str | None:
    """Ustawia globalną zmienną LANGUAGE na podstawie argumentu wiersza poleceń."""
    global LANGUAGE

    normalized = lang_code.strip() if lang_code is not None else None

    # Akceptujemy: None (autodetekcja) albo dwuznakowy kod ISO 639-1.
    if normalized is None or normalized.lower() == "none":
        LANGUAGE = None
    elif len(normalized) == 2 and normalized.isalpha():
        LANGUAGE = normalized.lower()
    else:
        raise SystemExit(
            "Error: language code should be None or a 2-letter ISO 639-1 code, "
            f"got '{lang_code}'"
        )

    if LANGUAGE is None:
        print("Language set to automatic detection (None). The model will try to detect the language of the audio.")
    else:
        print(f"Language set to: {LANGUAGE}")

    return LANGUAGE


def validate_model_name(model_name: str) -> str:
    """Waliduje nazwę modelu faster-whisper względem listy wspieranych wartości."""
    if model_name not in VALID_MODEL_NAMES:
        allowed = ", ".join(sorted(VALID_MODEL_NAMES))
        raise SystemExit(
            "Error: invalid MODEL_NAME. "
            f"Got '{model_name}'. Allowed values: {allowed}"
        )
    return model_name


def get_device() -> str:
    """Determines the best device to use: 'cuda' if available, otherwise 'cpu'."""
    if not TORCH_AVAILABLE:
        print("[WARNING] PyTorch not found. Using CPU for transcription (this will be slow).")
        return "cpu"
    
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Using CPU for transcription (this will be slow).")
        return "cpu"
    
    try:
        # Try to allocate a small tensor on CUDA to verify it actually works
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        print(f"[INFO] CUDA is available and working. Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    except Exception as e:
        print(f"[WARNING] CUDA is available but failed to initialize: {e}. Falling back to CPU.")
        return "cpu"


def main() -> None:
    """Główny punkt wejścia skryptu."""
    validate_model_name(MODEL_NAME)

    args = parse_args()
    audio = args.audio_file.expanduser().resolve()
    preset = apply_style_preset(args.style)

    if not audio.exists():
        raise SystemExit(f"Error: file does not exist: {audio}")
    
    print(f"File: {audio}")
    print(f"Subtitle style: {args.style}")
    print(f"Mux SRT into MKV: {'yes' if args.mux_mkv_srt else 'no'}")
    print(
        "Style parameters: "
        f"max_line_length={preset.max_line_length}, "
        f"max_block_duration={preset.max_block_duration:.1f}s, "
        f"max_join_gap={preset.max_join_gap:.2f}s"
    )
    
    total_duration = get_media_duration_seconds(audio)
    if total_duration:
        print(f"File duration: {total_duration:.1f} s")
    else:
        print("Failed to read file duration. Progress bar will be based on segments.")
    
    set_language(args.language)

    cached, cache_root = detect_model_cache(MODEL_NAME)
    if cached:
        stage(f"Model '{MODEL_NAME}' found in cache at: {cache_root}")
    else:
        stage(f"Model '{MODEL_NAME}' not found in cache. It will be downloaded to: {cache_root}")

    # Inicjalizacja modelu AI
    # Uwaga: Za pierwszym razem pobierze model z HuggingFace (ok. 1.5GB - 3GB)
    stage("Loading Whisper model")
    device = get_device()
    model = WhisperModel(MODEL_NAME, device=device, compute_type=COMPUTE_TYPE)
    stage("Whisper model loaded")

    # Uruchomienie transkrypcji (generator)
    stage("Starting transcription")
    segments, info = model.transcribe(
        str(audio),
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,
        vad_parameters=VAD_PARAMETERS,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        word_timestamps=True,
    )

    # Przetworzenie wyników i zbieranie słów
    with create_progress_bar(total_duration) as pbar:
        words, _ = collect_words_and_metadata(segments, total_duration, pbar)

    if not words:
        raise SystemExit("Error: model did not return any words to save.")

    # Przetworzenie słów na inteligentne bloki napisów
    subtitle_blocks = build_subtitle_blocks(words)

    # Zapis do plików
    stage("Saving subtitles to TXT and SRT")
    txt_path = audio.with_suffix(".txt")
    srt_path = audio.with_suffix(".srt")

    with txt_path.open("w", encoding="utf-8") as txt_f, srt_path.open("w", encoding="utf-8") as srt_f:
        for i, block in enumerate(subtitle_blocks, start=1):
            # TXT: prosta lista zdań/bloków
            txt_f.write(block.plain_text + "\n")
            # SRT: format czasowy
            srt_f.write(f"{i}\n")
            srt_f.write(f"{srt_timestamp(block.start)} --> {srt_timestamp(block.end)}\n")
            srt_f.write(block.text_for_srt + "\n\n")

    # Podsumowanie
    stage("Transcription completed")
    print(f"Language: {info.language} (probability={info.language_probability:.3f})")
    print(f"Number of words: {len(words)}")
    print(f"Number of merged subtitles: {len(subtitle_blocks)}")
    print(f"TXT: {txt_path}")
    print(f"SRT: {srt_path}")

    if args.mux_mkv_srt:
        muxed_path = mux_srt_with_mkv(audio, srt_path, LANGUAGE)
        print(f"MKV with subtitles: {muxed_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Transcription stopped.", file=sys.stderr)
        raise SystemExit(130)

#        NAME: Transkrypcja pliku audio lub wideo
# DESCRIPTION: Skrypt używa modelu faster-whisper do transkrypcji pliku audio lub wideo, a następnie zapisuje wyniki w formacie TXT i SRT. 
#              Skrypt jest zoptymalizowany pod kątem czytelności napisów, dzieląc tekst na bloki z uwzględnieniem długości, czasu trwania i interpunkcji. 
#              Pasek postępu tqdm pokazuje postęp transkrypcji w czasie rzeczywistym.
#      AUTHOR: Robert Drygas / ChatGPT
#     VERSION: 1.1.0
#     CREATED: 2026-03-14
#    MODIFIED: 2026-03-17
#
# DEPENDENCIES:
#
# TESTED ON:
#    - OS Windows 11 + WSL2 (Ubuntu 24.04, Python 3.14) + GPU NVIDIA GeForce RTX 3060 + CPU Intel Core i7 11700
#
# USAGE:
#    $ python3 transcribe.py <filename> [--style <style>]
#
# ARGUMENTS:
#    <filename> - ścieżka do pliku audio lub wideo (obowiązkowe)
#    --style    - styl napisów: 'reading' daje dłuższe, wygodniejsze bloki, a 'film' tworzy krótsze, bardziej klasyczne SRT (opcjonalne, domyślnie 'reading')
#
# EXAMPLES:
#    $ python3 transcribe.py nagranie.mp3
#    $ python3 transcribe.py nagranie.mkv
#    $ python3 transcribe.py nagranie.mp4 --style film
#
# CHANGELOG:
#    - 1.0.0 (2026-03-14) Pierwsza wersja
#    - 1.1.0 (2026-03-17) Dodano style napisów
#
# ROADMAP:
#    - [*] Style napisów
#    - [ ] Komunikaty etapów wykonywania skryptu
#    - [ ] Wykrywanie cache modelu
#    - [ ] Połączenie napisów z nagraniem wideo MKV
#
# LICENSE: GPL-3.0


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import av
from faster_whisper import WhisperModel
from tqdm import tqdm

MODEL_NAME = "turbo"          # "turbo" albo "large-v3"
COMPUTE_TYPE = "float16"      # przy braku VRAM: "int8_float16"
LANGUAGE = "pl"               # kod języka, np. "pl" dla polskiego, "en" dla angielskiego, itp. Można też ustawić na None, by model wykrył język automatycznie.
BEAM_SIZE = 5
VAD_FILTER = True
VAD_PARAMETERS = {"min_silence_duration_ms": 3000}
CONDITION_ON_PREVIOUS_TEXT = False

# Parametry formatowania napisów - będą nadpisywane przez preset stylu
MAX_LINE_LENGTH = 42            # maks. liczba znaków w jednej linii
MAX_LINES = 2                   # maks. liczba linii w jednym napisie
MAX_BLOCK_CHARS = MAX_LINE_LENGTH * MAX_LINES
MAX_BLOCK_DURATION = 6.0        # maks. czas jednego napisu w sekundach
MIN_BLOCK_CHARS = 18            # krótsze napisy będą scalane, jeśli to możliwe
MIN_BLOCK_DURATION = 1.2
MAX_JOIN_GAP = 1.0              # maks. przerwa między blokami, by je scalić

STRONG_PUNCT = ".?!"
SOFT_PUNCT = ",;:)]}"


@dataclass(frozen=True, slots=True)
class StylePreset:
    max_line_length: int
    max_lines: int
    max_block_duration: float
    min_block_chars: int
    min_block_duration: float
    max_join_gap: float


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

@dataclass(slots=True)
class WordToken:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class SubtitleBlock:
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
        return render_words(self.words)

    @property
    def text_for_srt(self) -> str:
        return wrap_block_text(self.words)


def apply_style_preset(style_name: str) -> StylePreset:
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
    parser = argparse.ArgumentParser(
        description="Transcription of audio/video files to TXT and SRT formats, featuring a progress bar, word timestamps and intelligent subtitle formatting.",
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio or video file, e.g., recording.mp3 or movie.mp4",
    )
    parser.add_argument(
        "-s", "--style",
        choices=sorted(STYLE_PRESETS.keys()),
        default="reading",
        help="Subtitle style: 'reading' gives longer, more comfortable blocks, while 'film' creates shorter, more classic SRT. Default is 'reading'.",
    )
    return parser.parse_args()


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


def get_media_duration_seconds(path: Path) -> float | None:
    with av.open(str(path)) as container:
        if container.duration is not None:
            return float(container.duration / av.time_base)

        for stream in container.streams:
            if stream.type in ("audio", "video") and stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)

    return None


def create_progress_bar(total_duration: float | None) -> tqdm:
    common_kwargs = {
        "desc": "Transcribing",
        "dynamic_ncols": True,
        "leave": True,
    }

    if total_duration and total_duration > 0:
        return tqdm(
            total=total_duration,
            unit="s",
            bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]",
            **common_kwargs,
        )

    return tqdm(unit="seg", **common_kwargs)


def render_words(words: list[WordToken]) -> str:
    text = "".join(word.text for word in words).strip()
    return " ".join(text.split())


def ends_with_punctuation(text: str, punctuation: str) -> bool:
    stripped = text.rstrip('"”’» ')
    return bool(stripped) and stripped[-1] in punctuation


def should_break_after(current_words: list[WordToken], next_word: WordToken | None) -> bool:
    if not current_words:
        return False

    current_text = render_words(current_words)
    current_duration = current_words[-1].end - current_words[0].start

    if len(current_text) >= MAX_BLOCK_CHARS:
        return True

    if current_duration >= MAX_BLOCK_DURATION:
        return True

    if next_word is None:
        return True

    gap_after = max(0.0, next_word.start - current_words[-1].end)
    if gap_after >= MAX_JOIN_GAP:
        return True

    if len(render_words(current_words + [next_word])) > MAX_BLOCK_CHARS:
        return True

    if ends_with_punctuation(current_text, STRONG_PUNCT):
        return True

    punct_threshold = 0.45 if MAX_LINE_LENGTH <= 36 else 0.55
    if ends_with_punctuation(current_text, SOFT_PUNCT) and len(current_text) >= int(MAX_BLOCK_CHARS * punct_threshold):
        return True

    return False


def words_to_initial_blocks(words: list[WordToken]) -> list[SubtitleBlock]:
    if not words:
        return []

    blocks: list[SubtitleBlock] = []
    current: list[WordToken] = []

    for index, word in enumerate(words):
        if current:
            gap_before = max(0.0, word.start - current[-1].end)
            if gap_before >= MAX_JOIN_GAP:
                blocks.append(SubtitleBlock(words=current))
                current = []

        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None

        if should_break_after(current, next_word):
            blocks.append(SubtitleBlock(words=current))
            current = []

    if current:
        blocks.append(SubtitleBlock(words=current))

    return blocks


def is_short_block(block: SubtitleBlock) -> bool:
    return (
        len(block.plain_text) < MIN_BLOCK_CHARS
        or block.duration < MIN_BLOCK_DURATION
        or len(block.words) <= 2
    )


def can_merge_blocks(left: SubtitleBlock, right: SubtitleBlock) -> bool:
    gap = max(0.0, right.start - left.end)
    if gap > MAX_JOIN_GAP:
        return False

    merged_words = left.words + right.words
    merged_text = render_words(merged_words)
    merged_duration = right.end - left.start

    if len(merged_text) > MAX_BLOCK_CHARS:
        return False

    if merged_duration > MAX_BLOCK_DURATION:
        return False

    return True


def merge_short_blocks(blocks: list[SubtitleBlock]) -> list[SubtitleBlock]:
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
                if result and can_merge_blocks(result[-1], current):
                    previous = result.pop()
                    result.append(SubtitleBlock(words=previous.words + current.words))
                    changed = True
                    i += 1
                    continue

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
    if len(words) < 2:
        return None

    best_index: int | None = None
    best_score: tuple[float, float, float] | None = None

    for idx in range(1, len(words)):
        left = render_words(words[:idx])
        right = render_words(words[idx:])

        left_len = len(left)
        right_len = len(right)
        overflow = max(0, left_len - MAX_LINE_LENGTH) + max(0, right_len - MAX_LINE_LENGTH)
        imbalance = abs(left_len - right_len)
        max_len = max(left_len, right_len)

        punctuation_bonus = 0.0
        if ends_with_punctuation(left, STRONG_PUNCT):
            punctuation_bonus = -8.0
        elif ends_with_punctuation(left, SOFT_PUNCT):
            punctuation_bonus = -4.0

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
    initial_blocks = words_to_initial_blocks(words)
    return merge_short_blocks(initial_blocks)


def collect_words_and_metadata(
    segments,
    total_duration: float | None,
    pbar: tqdm,
) -> tuple[list[WordToken], object]:
    all_words: list[WordToken] = []
    last_progress_seconds = 0.0
    info = None
    segment_count = 0

    for segment_count, seg in enumerate(segments, start=1):
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

        words = getattr(seg, "words", None) or []
        for word in words:
            if word.start is None or word.end is None:
                continue
            token_text = word.word or ""
            if not token_text.strip():
                continue
            all_words.append(WordToken(start=float(word.start), end=float(word.end), text=token_text))

        # Fallback, jeśli model nie zwrócił timestampów słów dla danego segmentu.
        if not words and seg.text.strip():
            all_words.append(
                WordToken(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=f" {seg.text.strip()}",
                )
            )

        info = getattr(seg, "_info", info)

    if total_duration and total_duration > 0 and last_progress_seconds < total_duration:
        pbar.update(total_duration - last_progress_seconds)

    pbar.set_postfix(segment=segment_count, words=len(all_words), refresh=False)
    return all_words, info


def main() -> None:
    args = parse_args()
    audio = args.audio_file.expanduser().resolve()
    preset = apply_style_preset(args.style)

    if not audio.exists():
        raise SystemExit(f"Error: file does not exist: {audio}")

    total_duration = get_media_duration_seconds(audio)
    print(f"File: {audio}")
    print(f"Subtitle style: {args.style}")
    print(
        "Style parameters: "
        f"max_line_length={preset.max_line_length}, "
        f"max_block_duration={preset.max_block_duration:.1f}s, "
        f"max_join_gap={preset.max_join_gap:.2f}s"
    )
    if total_duration:
        print(f"File duration: {total_duration:.1f} s")
    else:
        print("Failed to read file duration. Progress bar will be based on segments.")

    model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE)

    segments, info = model.transcribe(
        str(audio),
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,
        vad_parameters=VAD_PARAMETERS,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        word_timestamps=True,
    )

    with create_progress_bar(total_duration) as pbar:
        words, _ = collect_words_and_metadata(segments, total_duration, pbar)

    if not words:
        raise SystemExit("Error: model did not return any words to save.")

    subtitle_blocks = build_subtitle_blocks(words)

    txt_path = audio.with_suffix(".txt")
    srt_path = audio.with_suffix(".srt")

    with txt_path.open("w", encoding="utf-8") as txt_f, srt_path.open("w", encoding="utf-8") as srt_f:
        for i, block in enumerate(subtitle_blocks, start=1):
            txt_f.write(block.plain_text + "\n")

            srt_f.write(f"{i}\n")
            srt_f.write(f"{srt_timestamp(block.start)} --> {srt_timestamp(block.end)}\n")
            srt_f.write(block.text_for_srt + "\n\n")

    print(f"Language: {info.language} (p={info.language_probability:.3f})")
    print(f"Number of words: {len(words)}")
    print(f"Number of merged subtitles: {len(subtitle_blocks)}")
    print(f"TXT: {txt_path}")
    print(f"SRT: {srt_path}")


if __name__ == "__main__":
    main()

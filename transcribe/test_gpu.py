import torch
from faster_whisper import WhisperModel

print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    model = WhisperModel("turbo", device="cuda", compute_type="float16")
    print("OK: model załadowany na GPU")
else:
    print("BŁĄD: CUDA niedostępna, model nie został załadowany")
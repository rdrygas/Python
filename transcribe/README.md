# Transkrypcja plików audio/wideo w WSL2 (Ubuntu) + GPU NVIDIA

## Cel

Celem jest uruchomienie lokalnej transkrypcji plików audio lub wideo z wykorzystaniem karty NVIDIA pod **Windows 11 + WSL2 + Ubuntu**.

## Wymagania

- Windows 11 25H2 + WSL2 + Ubuntu 24.04
- Windows 11: aktualne sterowniki GeForce
- Ubuntu 24.04: Python 3.11+, PyTorch, faster-whisper

## Krok po kroku

### 1. Otwórz PowerShell i zweryfikuj, czy działa sterownik NVIDIA

```PowerShell
nvidia-smi
```

Jeśli dostajesz tabelkę z nazwą GPU i wersją sterownika, to sterownik po stronie hosta działa.  
Jeśli komenda nie działa albo pokazuje błąd, najpierw zainstaluj lub zaktualizuj sterownik NVIDIA z oficjalnej strony producenta. 
NVIDIA w dokumentacji CUDA i cuDNN podaje sterownik jako element wymaganej konfiguracji dla Windows.
Microsoft wymaga sterownika NVIDIA dla WSL, a NVIDIA wskazuje, że to właśnie sterownik hosta daje dostęp do CUDA w WSL.

[CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
[NVIDIA cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/index.html)
[Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
[CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### 2. Uruchom Ubuntu i sprawdź, czy WSL widzi GPU

```Bash
/usr/lib/wsl/lib/nvidia-smi
```

### 3. Nie instaluj w Ubuntu pakietów typu `nvidia-driver-*`

Tego kroku **nie rób**:

```Bash
# NIE rób tego w WSL:
# sudo apt install nvidia-driver-XXX
```

NVIDIA ostrzega, że w WSL nie wolno instalować linuksowego sterownika, a przy instalacji toolkita trzeba uważać, żeby nie nadpisać sterownika mapowanego z Windows. Pod WSL nie wybiera się meta-pakietów `cuda`, `cuda-12-x` ani `cuda-drivers`, bo próbują instalować linuksowy sterownik.

### 4. Zainstaluj podstawy w Ubuntu

```Bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg
python3 --version
```

PyTorch na Linuksie instaluje się przez `pip`, a instrukcja zaznacza, że na Linuksie `pip` nie zawsze jest domyślnie obecny. 
`ffmpeg` nie jest wymagany przez `faster-whisper` tak jak przez `openai-whisper`, ale warto go mieć pod ręką do konwersji plików. 
PyTorch oficjalnie prowadzi instalację przez `pip` dla Linuksa.

[PyTorch: Get Started](https://pytorch.org/get-started/locally/)
[SYSTRAN/faster-whisper: Faster Whisper transcription with CTranslate2](https://github.com/SYSTRAN/faster-whisper)

### 5. Załóż czyste środowisko Python

```Bash
mkdir -p ~/ai/whisper-pl
cd ~/ai/whisper-pl
python3 -m venv .venv
source .venv/bin/activate
```

I dalej w środowisku `venv`:

```Bash
python -m pip install --upgrade pip
which python
python --version
```

To izoluje paczki od reszty systemu. Microsoft w swoim przewodniku dla ML w WSL też rekomenduje osobne środowisko Python.

[Get started with GPU acceleration for Machine Learning in WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)

### 6. Zainstaluj PyTorch z CUDA

PyTorch zaleca dla Linuksa wybrać
- OS: Linux, 
- Package: Pip
- odpowiednią wersję CUDA; zwykle najnowsza jest dobrym wyborem (aktualnie praktycznym wyborem jest build cu126 albo nowszy).

W środowisku `venv`:

```Bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

lub:

```Bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Weryfikacja:

```Bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'BRAK')"
```

Przykładowy wynik działania:

```Text
Torch: 2.10.0+cu130
CUDA build: 13.0
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

Jeśli dostaniesz `CUDA available: True`, to tor **WSL → Python → PyTorch → GPU** działa. 
PyTorch oficjalnie używa właśnie takiego sprawdzenia przez `torch.cuda.is_available()`.

[PyTorch: Get Started](https://pytorch.org/get-started/locally/)

### 7. Zainstaluj `faster-whisper`

Repo projektu podaje, że najnowsze wersje `ctranslate2` wspierają **CUDA 12 i cuDNN 9**. `faster-whisper` pokazuje też bezpośrednio uruchamianie na GPU przez `device="cuda", compute_type="float16"`

W środowisku `.venv`:

```Bash
pip install faster-whisper
```

[SYSTRAN/faster-whisper: Faster Whisper transcription with CTranslate2](https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file)

### 8. Dodaj wymagane biblioteki NVIDIA po stronie Linuxa

W środowisku `.venv`:

```Bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

I ustaw `LD_LIBRARY_PATH` przed uruchomieniem Pythona. Żeby nie robić tego ręcznie za każdym razem, dopisz do aktywacji środowiska:

```Bash
cat >> .venv/bin/activate <<'EOF'
export LD_LIBRARY_PATH=$(python3 -c 'import importlib.util; cublas_spec = importlib.util.find_spec("nvidia.cublas.lib"); cudnn_spec = importlib.util.find_spec("nvidia.cudnn.lib"); cublas_dir = list(cublas_spec.submodule_search_locations)[0]; cudnn_dir = list(cudnn_spec.submodule_search_locations)[0]; print(f"{cublas_dir}:{cudnn_dir}")'):$LD_LIBRARY_PATH
EOF
```

Po tym wyłącz i włącz środowisko:

```Bash
deactivate
source .venv/bin/activate
```

### 9. Minimalny test, czy `faster-whisper` wstaje na CUDA

Utwórz plik `test_gpu.py`:

```Python
import torch
from faster_whisper import WhisperModel

print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    model = WhisperModel("turbo", device="cuda", compute_type="float16")
    print("OK: model załadowany na GPU")
else:
    print("BŁĄD: CUDA niedostępna, model nie został załadowany")
```

Uruchom:

```Bash
python test_gpu.py
```


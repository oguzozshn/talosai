"""Report whether this machine can run Talos, and where it will fall short.

Standard library only, on purpose: this script has to be useful *before*
anything is installed, so it can tell you what is missing. It never imports a
third-party package at module level.

Run it before and after `pip install -r requirements.txt`.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"

# Speech recognition and the language model share the GPU. Below this there is
# not enough room for both at once and one of them has to run on CPU.
MIN_VRAM_MB = 6000

# Blackwell and newer report compute capability 12.0. Wheels built against
# anything older than CUDA 12.8 do not contain code for it and will silently
# fall back to CPU rather than erroring.
BLACKWELL_MAJOR = 12


@dataclass
class Result:
    status: str
    label: str
    detail: str = ""
    hint: str = ""


@dataclass
class Section:
    title: str
    results: list[Result] = field(default_factory=list)

    def add(self, status: str, label: str, detail: str = "", hint: str = "") -> None:
        self.results.append(Result(status, label, detail, hint))


def run(*cmd: str, timeout: int = 20) -> tuple[int, str]:
    """Run a command, returning (exit code, combined output). Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    except OSError as exc:
        return 1, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def total_ram_gb() -> float | None:
    """Physical RAM in GB, or None if this is not Windows."""
    if platform.system() != "Windows":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullTotalPhys / (1024**3)


def check_system() -> Section:
    section = Section("System")

    major, minor = sys.version_info[:2]
    version = platform.python_version()
    if (major, minor) == (3, 12):
        section.add(OK, "Python", version)
    elif (major, minor) in {(3, 10), (3, 11)}:
        section.add(
            WARN,
            "Python",
            f"{version} — workable, but 3.12 is what this is developed against",
        )
    elif (major, minor) >= (3, 13):
        section.add(
            FAIL,
            "Python",
            f"{version} — too new",
            "faster-whisper and several audio packages have no wheels for 3.13+. "
            'Create the venv with "py -3.12 -m venv ..." and run this again.',
        )
    else:
        section.add(FAIL, "Python", f"{version} — too old, need 3.12")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        section.add(OK, "Virtual environment", sys.prefix)
    else:
        section.add(
            WARN,
            "Virtual environment",
            "not active — installing into the system Python",
            "Activate the venv first, or packages land somewhere you did not intend.",
        )

    section.add(INFO, "Platform", f"{platform.system()} {platform.release()}")

    cores = os.cpu_count() or 0
    # Whisper on CPU is the fallback path if CUDA does not work out, and it
    # needs real parallelism to stay near real time.
    section.add(OK if cores >= 8 else WARN, "CPU threads", str(cores))

    ram = total_ram_gb()
    if ram is None:
        section.add(INFO, "RAM", "unknown")
    elif ram >= 16:
        section.add(OK, "RAM", f"{ram:.1f} GB")
    else:
        section.add(
            WARN,
            "RAM",
            f"{ram:.1f} GB",
            "Tight. Expect to run a smaller language model.",
        )

    return section


def check_gpu() -> Section:
    section = Section("GPU")

    if not shutil.which("nvidia-smi"):
        section.add(
            WARN,
            "NVIDIA GPU",
            "nvidia-smi not found",
            "Everything still runs on CPU, just slower. If you do have an "
            "NVIDIA card, the driver is not installed properly.",
        )
        return section

    code, out = run(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    )
    if code != 0 or not out.strip():
        section.add(WARN, "NVIDIA GPU", "nvidia-smi failed to report")
        return section

    fields = [f.strip() for f in out.strip().splitlines()[0].split(",")]
    if len(fields) < 4:
        section.add(WARN, "NVIDIA GPU", out.strip())
        return section

    name, vram_raw, driver, compute = fields[:4]
    section.add(OK, "GPU", name)
    section.add(INFO, "Driver", driver)

    try:
        vram = int(float(vram_raw))
    except ValueError:
        vram = 0
    if vram >= MIN_VRAM_MB:
        section.add(OK, "VRAM", f"{vram} MB")
    else:
        section.add(
            WARN,
            "VRAM",
            f"{vram} MB",
            "Not enough for speech recognition and the language model at the "
            "same time. Run Whisper on CPU and give the GPU to the model.",
        )

    try:
        compute_major = int(float(compute))
    except ValueError:
        compute_major = 0
    if compute_major >= BLACKWELL_MAJOR:
        section.add(
            WARN,
            "Compute capability",
            f"{compute} (Blackwell or newer)",
            "Packages built before CUDA 12.8 do not support this architecture "
            "and will fall back to CPU without saying so. Check the CUDA line "
            "below to see whether that has happened.",
        )
    else:
        section.add(OK, "Compute capability", compute)

    return section


def check_packages() -> Section:
    section = Section("Python packages")

    # (import name, why Talos needs it)
    required = [
        ("numpy", "audio buffers"),
        ("sounddevice", "microphone and speaker I/O"),
        ("soundfile", "reading and writing wav"),
        ("faster_whisper", "speech-to-text"),
        ("ctranslate2", "the engine underneath faster-whisper"),
        ("ollama", "talking to the local language model"),
    ]
    # Not needed until later phases, but cheap to report on now.
    optional = [
        ("onnxruntime", "wake word and voice activity detection (phase 2)"),
        ("piper", "text-to-speech, if installed as a package rather than a binary"),
    ]

    missing = []
    for module, purpose in required:
        if importlib.util.find_spec(module) is not None:
            section.add(OK, module, purpose)
        else:
            section.add(FAIL, module, f"missing — needed for {purpose}")
            missing.append(module)

    for module, purpose in optional:
        status = OK if importlib.util.find_spec(module) is not None else INFO
        detail = purpose if status == OK else f"not installed — {purpose}"
        section.add(status, module, detail)

    if missing:
        section.add(
            INFO,
            "Next step",
            "pip install -r requirements.txt",
        )

    return section


def check_cuda_runtime() -> Section:
    """Ask CTranslate2 directly whether it can see the GPU.

    This is the check that matters. nvidia-smi reporting a healthy card says
    nothing about whether the installed wheels contain code for it.
    """
    section = Section("CUDA runtime")

    if importlib.util.find_spec("ctranslate2") is None:
        section.add(INFO, "CTranslate2", "not installed yet — install and re-run")
        return section

    try:
        import ctranslate2  # noqa: PLC0415  (deliberately deferred)

        devices = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001 — any failure here means "no CUDA"
        section.add(
            WARN,
            "CTranslate2 CUDA",
            f"unavailable ({type(exc).__name__}: {exc})",
            "Speech recognition will run on CPU.",
        )
        return section

    if devices > 0:
        section.add(OK, "CTranslate2 CUDA", f"{devices} device(s) visible")
    else:
        section.add(
            WARN,
            "CTranslate2 CUDA",
            "no devices visible",
            "Usually one of: cuDNN 9 for CUDA 12 is missing from PATH, or the "
            "installed build predates support for this GPU. Whisper falls back "
            "to CPU, which on a many-core machine is slow but usable.",
        )
    return section


def find_ollama() -> str | None:
    """Locate the Ollama binary.

    PATH alone is not enough: the installer adds it, but only shells started
    afterwards inherit the change, so a freshly installed Ollama looks missing
    until the user opens a new terminal. Fall back to the default install
    location before concluding it is absent.
    """
    found = shutil.which("ollama")
    if found:
        return found

    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = os.path.join(local, "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def check_ollama() -> Section:
    section = Section("Ollama")

    binary = find_ollama()
    if not binary:
        section.add(
            FAIL,
            "Ollama",
            "not installed",
            "Install from https://ollama.com/download, then pull a model.",
        )
        return section

    code, out = run(binary, "--version")
    section.add(OK, "Ollama", out.strip().splitlines()[0] if out.strip() else "installed")
    if not shutil.which("ollama"):
        section.add(
            WARN,
            "PATH",
            "installed but not on PATH in this shell",
            "Scripts that shell out to the CLI will not find it until you open "
            "a new terminal. Nothing in Talos depends on that — it talks to the "
            "service over HTTP — but it is worth knowing.",
        )

    code, out = run(binary, "list", timeout=30)
    if code != 0:
        section.add(
            WARN,
            "Service",
            "not responding",
            "The background service is not running. Start it with 'ollama serve', "
            "or launch the Ollama app.",
        )
        return section

    lines = [ln for ln in out.strip().splitlines()[1:] if ln.strip()]
    if not lines:
        section.add(
            WARN,
            "Models",
            "none pulled",
            "Try: ollama pull qwen3:8b   (and gemma3:4b to compare Turkish)",
        )
    else:
        names = [ln.split()[0] for ln in lines]
        section.add(OK, "Models", ", ".join(names))

    return section


def check_audio() -> Section:
    section = Section("Audio devices")

    if importlib.util.find_spec("sounddevice") is None:
        section.add(INFO, "Devices", "sounddevice not installed yet")
        return section

    try:
        import sounddevice as sd  # noqa: PLC0415  (deliberately deferred)

        devices = sd.query_devices()
        default_in, default_out = sd.default.device
    except Exception as exc:  # noqa: BLE001
        section.add(WARN, "Devices", f"could not enumerate ({exc})")
        return section

    inputs = [d for d in devices if d["max_input_channels"] > 0]
    outputs = [d for d in devices if d["max_output_channels"] > 0]

    if inputs:
        try:
            name = devices[default_in]["name"]
        except (IndexError, TypeError):
            name = inputs[0]["name"]
        section.add(OK, "Microphone", f"{name}  ({len(inputs)} available)")
    else:
        section.add(FAIL, "Microphone", "no input device found")

    if outputs:
        try:
            name = devices[default_out]["name"]
        except (IndexError, TypeError):
            name = outputs[0]["name"]
        section.add(OK, "Speaker", f"{name}  ({len(outputs)} available)")
    else:
        section.add(FAIL, "Speaker", "no output device found")

    return section


def render(sections: list[Section]) -> tuple[int, int]:
    marks = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL ", INFO: "  --  "}
    failures = warnings = 0
    hints: list[tuple[str, str]] = []

    for section in sections:
        print(f"\n{section.title}")
        print("-" * len(section.title))
        for r in section.results:
            detail = f"  {r.detail}" if r.detail else ""
            print(f"[{marks[r.status]}] {r.label:<22}{detail}")
            if r.status == FAIL:
                failures += 1
            elif r.status == WARN:
                warnings += 1
            if r.hint:
                hints.append((r.label, r.hint))

    if hints:
        print("\n\nNotes")
        print("-----")
        for label, hint in hints:
            print(f"\n{label}:")
            for line in _wrap(hint, 74):
                print(f"  {line}")

    return failures, warnings


def _wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main() -> int:
    print("Talos environment check")
    print("=" * 23)

    sections = [
        check_system(),
        check_gpu(),
        check_packages(),
        check_cuda_runtime(),
        check_ollama(),
        check_audio(),
    ]

    failures, warnings = render(sections)

    print("\n")
    if failures:
        print(f"{failures} blocking problem(s), {warnings} warning(s).")
        print("Fix the failures above, then run this again.")
        return 1

    if warnings:
        print(f"No blocking problems. {warnings} warning(s) — read the notes.")
        print("Talos will run, possibly with some parts on CPU.")
        return 0

    print("Everything checks out. Move on to the individual test scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

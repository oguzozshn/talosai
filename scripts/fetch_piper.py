"""Download the Piper binary and a Turkish voice into the right directories.

Piper has no usable Windows wheel, so it is a binary plus two voice files that
otherwise have to be fetched by hand from two different sites. This does that.

    python scripts/fetch_piper.py [--voice NAME] [--force]

Standard library only, so it runs before `pip install` if you want it to.
Everything it writes is git-ignored.

The release asset is resolved through the GitHub API rather than hardcoded:
Piper's release tags and asset names have changed shape more than once, and a
pinned URL that 404s two years from now is a worse failure than an API call.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPER_DIR = ROOT / "vendor" / "piper"
VOICE_DIR = ROOT / "models" / "piper"

RELEASES_API = "https://api.github.com/repos/rhasspy/piper/releases"
VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICES_TREE = "https://huggingface.co/api/models/rhasspy/piper-voices/tree/main"

DEFAULT_LANGUAGE = "tr/tr_TR"

# "medium" first: "low" is noticeably robotic, and "high" costs more synthesis
# time than the latency budget has to spare. The rest are fallbacks for
# languages that do not publish a medium.
QUALITY_PREFERENCE = ("medium", "high", "low", "x_low")

USER_AGENT = "talos-setup"


def get(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download(url: str, target: Path) -> bool:
    """Stream a URL to disk with a progress line. Returns success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with partial.open("wb") as handle:
                while chunk := response.read(256 * 1024):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        bar = "#" * (pct // 3) + "." * (33 - pct // 3)
                        print(f"\r  [{bar}] {pct:3d}%  {done / 1e6:6.1f} MB", end="")
                    else:
                        print(f"\r  {done / 1e6:6.1f} MB", end="")
    except urllib.error.HTTPError as exc:
        print(f"\r  HTTP {exc.code} for {url}")
        partial.unlink(missing_ok=True)
        return False
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"\r  network error: {exc}")
        partial.unlink(missing_ok=True)
        return False

    print()
    partial.replace(target)
    return True


def find_windows_asset() -> tuple[str, str] | None:
    """Return (asset name, download URL) for the Windows x64 build."""
    try:
        releases = json.loads(get(RELEASES_API))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"Could not reach the GitHub releases API: {exc}")
        return None

    # Newest first. Not every release ships a Windows asset, so scan until one
    # does rather than assuming the latest is the right one.
    for release in releases:
        for asset in release.get("assets", []):
            name = asset.get("name", "").lower()
            if "windows" in name and ("amd64" in name or "x64" in name) and name.endswith(".zip"):
                return asset["name"], asset["browser_download_url"]
    return None


def install_binary(force: bool) -> bool:
    exe = PIPER_DIR / "piper.exe"
    if exe.exists() and not force:
        print(f"Binary already present: {exe}")
        return True

    print("Resolving the latest Windows release...")
    found = find_windows_asset()
    if not found:
        print(
            "No Windows asset found in any recent release.\n"
            "Grab it manually from https://github.com/rhasspy/piper/releases\n"
            f"and unzip so that {exe} exists."
        )
        return False

    name, url = found
    print(f"Downloading {name}")
    archive = PIPER_DIR / name
    if not download(url, archive):
        return False

    print("  extracting...")
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(PIPER_DIR)
    archive.unlink(missing_ok=True)

    # Archives sometimes contain a top-level piper/ directory, sometimes not.
    if not exe.exists():
        nested = next(PIPER_DIR.rglob("piper.exe"), None)
        if nested is None:
            print(f"  extracted, but no piper.exe anywhere under {PIPER_DIR}")
            return False
        for item in nested.parent.iterdir():
            shutil.move(str(item), str(PIPER_DIR / item.name))

    print(f"  installed: {exe}")
    return True


def resolve_voice(language: str, wanted: str | None) -> str | None:
    """Find a voice model path under a language directory.

    Asks the Hugging Face tree API rather than assuming a path. The voice
    families are not stable enough to hardcode — tr_TR is published under
    `dfki`, and guessing a name here produced two 404s and no useful error.

    Returns the repo-relative path of the .onnx, or None.
    """
    try:
        entries = json.loads(get(f"{VOICES_TREE}/{language}?recursive=true"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"Could not list voices for {language}: {exc}")
        return None

    models = [e["path"] for e in entries if e.get("path", "").endswith(".onnx")]
    if not models:
        print(f"No voices published under {language}.")
        return None

    if wanted:
        matches = [p for p in models if Path(p).stem == wanted]
        if not matches:
            print(f"No voice named '{wanted}'. Available:")
            for path in models:
                print(f"  {Path(path).stem}")
            return None
        return matches[0]

    for quality in QUALITY_PREFERENCE:
        for path in models:
            if f"/{quality}/" in path:
                return path
    return models[0]


def install_voice(language: str, wanted: str | None, force: bool) -> bool:
    model_path = resolve_voice(language, wanted)
    if model_path is None:
        return False

    name = Path(model_path).stem
    ok = True
    for suffix in (".onnx", ".onnx.json"):
        target = VOICE_DIR / f"{name}{suffix}"
        if target.exists() and not force:
            print(f"Voice file already present: {target.name}")
            continue
        print(f"Downloading {name}{suffix}")
        if not download(f"{VOICES_BASE}/{model_path[:-5]}{suffix}", target):
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="repo language directory, e.g. tr/tr_TR or en/en_US",
    )
    parser.add_argument(
        "--voice",
        help="specific voice name; by default the best available quality is picked",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    print("Fetching Piper\n")
    binary_ok = install_binary(args.force)
    print()
    voice_ok = install_voice(args.language, args.voice, args.force)

    print()
    if binary_ok and voice_ok:
        print("Done. Verify with:  python scripts/test_tts.py")
        return 0

    print(
        "Incomplete. test_tts.py still works using the voice built into\n"
        "Windows, so this is not blocking — see scripts/README.md for the\n"
        "manual steps."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

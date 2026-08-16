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

# tr_TR has exactly one voice family on piper-voices at the time of writing.
# "medium" is the right default: "low" is noticeably robotic and "high" costs
# more synthesis time than the latency budget has to spare.
DEFAULT_VOICE = "tr_TR-fettah-medium"
VOICE_PATHS = {"tr_TR-fettah-medium": "tr/tr_TR/fettah/medium"}

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


def install_voice(voice: str, force: bool) -> bool:
    subpath = VOICE_PATHS.get(voice)
    if subpath is None:
        print(f"Unknown voice '{voice}'. Known: {', '.join(VOICE_PATHS)}")
        print("Browse others at https://huggingface.co/rhasspy/piper-voices")
        return False

    ok = True
    for suffix in (".onnx", ".onnx.json"):
        target = VOICE_DIR / f"{voice}{suffix}"
        if target.exists() and not force:
            print(f"Voice file already present: {target.name}")
            continue
        print(f"Downloading {voice}{suffix}")
        if not download(f"{VOICES_BASE}/{subpath}/{voice}{suffix}", target):
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    print("Fetching Piper\n")
    binary_ok = install_binary(args.force)
    print()
    voice_ok = install_voice(args.voice, args.force)

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

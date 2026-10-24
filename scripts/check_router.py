"""Measure how often the router offers the skill a request needs.

    run scripts\\check_router.py
    run scripts\\check_router.py --top 4

The router's one failure is quiet: a skill that is not offered cannot be
called, and the model answers without it as if nothing were missing. So this
checks every case for whether the skill it needs made the cut, without asking
the model anything — only the embedding model has to be running, and the
whole run takes seconds.

The requests are deliberately not the examples written in the skills. Testing
the router on the sentences it was built from would only measure whether it
can find an exact copy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from talos.router import ALWAYS, Router  # noqa: E402
from talos.skills import load as load_skills  # noqa: E402

CASES = [
    ("Anything due for me today?", "my_day"),
    ("What am I doing on Thursday?", "my_day"),
    ("Do I have homework due in Todoist?", "todoist_search"),
    ("Put 'renew passport' on my Todoist list.", "todoist_add"),
    ("Tick off the groceries task.", "todoist_complete"),
    ("How many hours of studying are left for the final?", "study_progress"),
    ("Remind me what my big goals are.", "goals"),
    ("Where did I put the Kurose networking book?", "find_file"),
    ("Is there a folder called Yüksek Lisans?", "find_file"),
    ("Show me what's in the notes.md of Talos.", "read_file"),
    ("Pop open the CV folder for me.", "show_in_explorer"),
    ("Text me the networking PDF.", "send_file"),
    ("Can you mail the homework file to my phone?", "send_file"),
    ("Which repos have I committed to lately?", "recent_work"),
    ("Is BACnet clean or does it have changes?", "project_status"),
    ("How many projects are there?", "list_projects"),
    ("Jot down that the meeting moved to Friday.", "take_note"),
    ("What's in my notes about Argos?", "read_notes"),
    ("Didn't we talk about the parser bug last week?", "recall"),
    ("Remind me what we just discussed.", "last_conversations"),
    ("Read me the error on the screen.", "read_screen"),
    ("Is it cold outside?", "weather"),
    ("What time is it in New York right now?", "time_in"),
    ("Is it already evening in Tokyo?", "time_in"),
    ("How do I get you to send me a file?", "about_me"),
    ("Who won the last World Cup?", "web_search"),
    ("Quit listening, I'm in a meeting.", "listening"),
    ("Correct my English while we talk.", "english_practice"),
    ("How much did I use Visual Studio today?", "app_usage"),
    ("I need the GPU back for a game.", "release_gpu"),
    ("Which of your features should I know about?", "about_me"),
    ("Bugün neler var?", "my_day"),
    ("Dersler klasörü nerede?", "find_file"),
    ("Hava nasıl olacak?", "weather"),
    # Turkish. nomic-embed-text is mostly English, and these are where the
    # router first fell short — see MEASUREMENTS.md.
    ("Almanya'da saat kaç?", "time_in"),
    ("Fransa'da şimdi saat kaç?", "time_in"),
    ("Yarın ne var?", "my_day"),
    ("Dersler klasörünü aç", "show_in_explorer"),
    ("CV'mi telefonuma gönder", "send_file"),
    ("Todoist'e süt almayı ekle", "todoist_add"),
    ("Vizeye kaç saat çalışmam kaldı?", "study_progress"),
    ("Final sınavına ne kadar çalışmam lazım?", "study_progress"),
    ("Bunu not al: toplantı cuma", "take_note"),
    ("Ekranda ne yazıyor?", "read_screen"),
    ("Geçen hafta BACnet hakkında ne konuştuk?", "recall"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    registry = load_skills()
    router = Router(registry, top=args.top)
    if not router.prepare():
        print("Could not embed the skills. Is Ollama running, with nomic-embed-text pulled?")
        return 1

    hits = 0
    sizes = []
    for request, needed in CASES:
        offered = router.choose(request) or []
        sizes.append(len(offered))
        found = needed in offered
        hits += found
        mark = "ok  " if found else "MISS"
        extra = "" if found else f"   offered: {', '.join(n for n in offered if n not in ALWAYS)}"
        print(f"  [{mark}] {request[:50]:<52} needs {needed}{extra}")

    print(
        f"\n{hits}/{len(CASES)} offered the skill they needed, "
        f"out of {len(registry)} skills, offering {min(sizes)}-{max(sizes)} per request."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

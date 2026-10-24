"""Searching the web.

The second thing that leaves the machine, after Todoist. DuckDuckGo's HTML
endpoint, which needs no key and no account — the query goes to their servers
and nothing else does.

Results are trimmed hard. Three results with a sentence each is what a person
can hold from something read aloud; ten with full snippets is a wall of text
nobody follows past the second one.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.error
import urllib.parse
import urllib.request

from talos.skills._net import ipv4_only
from talos.skills.base import Permission, skill

log = logging.getLogger(__name__)

ENDPOINT = "https://html.duckduckgo.com/html/"
TIMEOUT = 15

# The HTML endpoint is meant for browsers, so it is served without JavaScript
# and is stable enough to parse. It still wants to look like a browser asked.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]*>(?P<title>.*?)</a>.*?'
    r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAGS = re.compile(r"<[^>]+>")


def _clean(fragment: str) -> str:
    return " ".join(html.unescape(_TAGS.sub(" ", fragment)).split())


@skill(Permission.OPEN)
def web_search(query: str, limit: int = 3) -> str:
    """Look something up online: current facts, weather, news, prices, documentation.

    Use this for anything you do not know or that changes — today's weather,
    what a library does, what happened recently. Do not offer to search; search.

    Args:
        query: What to look up
        limit: How many results to report, at most

    Examples:
        how many people live in the world
        search the web for the latest Python release
        what is RAG
        internette ara
    """
    query = query.strip()
    if not query:
        return "I need something to search for."

    request = urllib.request.Request(
        ENDPOINT,
        data=urllib.parse.urlencode({"q": query}).encode("utf-8"),
        headers={"User-Agent": USER_AGENT},
        method="POST",
    )

    try:
        with ipv4_only(), urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            page = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return f"The search engine returned an error ({exc.code})."
    except (urllib.error.URLError, TimeoutError):
        return "I can't reach the internet."
    except Exception as exc:  # noqa: BLE001
        log.warning("search failed: %s", exc)
        return "The search didn't work."

    results = []
    for match in _RESULT.finditer(page):
        title = _clean(match.group("title"))
        snippet = _clean(match.group("snippet"))
        if title and snippet:
            # A long snippet read aloud is worse than no snippet: by the end
            # of it the listener has lost the question.
            if len(snippet) > 200:
                snippet = snippet[:197].rsplit(" ", 1)[0] + "..."
            results.append((title, snippet))
        if len(results) >= max(1, min(limit, 5)):
            break

    if not results:
        # Either nothing matched or the page shape changed. Both look the same
        # from here, and saying so is more useful than claiming no results.
        return f"I found nothing usable for '{query}'. The search page may have changed."

    lines = [f"Top {len(results)} result(s) for '{query}':"]
    for title, snippet in results:
        lines.append(f"  {title} — {snippet}")
    lines.append("These are search results, not something I know to be true.")
    return "\n".join(lines)

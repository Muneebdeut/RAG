"""
fetch_dataset.py — Download and clean Wikipedia Simple English articles
                   for use as the AetherRAG knowledge base.

This script fetches articles from the Wikipedia Simple English edition
(https://simple.wikipedia.org) using the public MediaWiki REST API.
No API key is required.  All dependencies are from the Python standard
library except for the optional ``requests`` package (falls back to
``urllib`` automatically).

Algorithm
---------
1. Query the Wikipedia category listing API to get article titles
   from a seed category (default: "Category:Sciences").
2. Fetch the plain-text extract for each article via the Action API
   (``action=query&prop=extracts&explaintext=1``).
3. Apply the same ``clean_text()`` pipeline used by the RAG engine to
   remove residual markup, HTML entities, and boilerplate headings.
4. Write one ``<slug>.txt`` file per article into ``dataset/wikipedia/``.
5. Print a progress summary.

Usage
-----
    python fetch_dataset.py                          # fetch 100 articles
    python fetch_dataset.py --articles 50            # fetch 50 articles
    python fetch_dataset.py --category "Mathematics" # different seed category
    python fetch_dataset.py --out dataset/my_wiki    # custom output directory
    python fetch_dataset.py --delay 0.3              # polite crawl delay (seconds)
    python fetch_dataset.py --help                   # show all options
"""

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
import re
import time
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from urllib.parse import urlencode, quote

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("fetch_dataset")

# ── Constants ─────────────────────────────────────────────────────────────────
SIMPLE_WIKI_API = "https://simple.wikipedia.org/w/api.php"
DEFAULT_CATEGORY = "Science"
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "wikipedia")
DEFAULT_ARTICLES = 100
DEFAULT_DELAY = 0.25   # seconds between API calls — be a polite citizen
MIN_ARTICLE_CHARS = 300  # skip stub articles shorter than this after cleaning

# Sections to strip from extracted text
_BOILERPLATE_HEADINGS = re.compile(
    r"^(References|External links|See also|Further reading|Notes|"
    r"Bibliography|Sources|Footnotes|Citations|Related pages)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# ── HTTP helper ───────────────────────────────────────────────────────────────

def _http_get(url: str, params: dict, timeout: int = 20) -> dict:
    """
    Perform a GET request and parse JSON.  Uses ``requests`` if installed,
    falls back to ``urllib.request``.
    """
    query_string = urlencode(params)
    full_url = f"{url}?{query_string}"

    try:
        import requests
        resp = requests.get(full_url, timeout=timeout,
                            headers={"User-Agent": "AetherRAG-Fetcher/1.0 (educational project)"})
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        pass  # fall through to urllib

    from urllib.request import Request, urlopen
    from urllib.error import URLError

    req = Request(
        full_url,
        headers={"User-Agent": "AetherRAG-Fetcher/1.0 (educational project)"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except URLError as exc:
        raise RuntimeError(f"HTTP request failed: {exc}") from exc


# ── Text cleaning (mirrors rag_engine.clean_text) ─────────────────────────────

_RE_HTML_TAG      = re.compile(r"<[^>]+>")
_RE_WIKI_TEMPLATE = re.compile(r"\{\{[^}]*\}\}")
_RE_WIKI_FILE     = re.compile(r"\[\[(?:File|Image|Media):[^\]]*\]\]", re.IGNORECASE)
_RE_WIKI_LINK     = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_RE_EXT_LINK      = re.compile(r"\[https?://\S+\s*([^\]]*)\]")
_RE_HTML_ENTITY   = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE   = re.compile(r"[ \t]{2,}")


def _clean_article_text(text: str) -> str:
    """
    Strip residual markup, boilerplate headings, and normalise whitespace
    from a Wikipedia plain-text extract.
    """
    if not text:
        return ""

    # Remove boilerplate sections and everything after them
    match = _BOILERPLATE_HEADINGS.search(text)
    if match:
        text = text[:match.start()]

    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_WIKI_TEMPLATE.sub("", text)
    text = _RE_WIKI_FILE.sub("", text)
    text = _RE_WIKI_LINK.sub(r"\1", text)
    text = _RE_EXT_LINK.sub(r"\1", text)
    text = _RE_HTML_ENTITY.sub(" ", text)

    # Unicode normalisation
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFC", text)

    text = _RE_CONTROL_CHARS.sub("", text)
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    text = _RE_MULTI_SPACE.sub(" ", text)

    return text.strip()


# ── Wikipedia API helpers ──────────────────────────────────────────────────────

def _get_category_members(category: str, limit: int = 500) -> list[str]:
    """
    Return up to *limit* article titles from the given Wikipedia Simple English
    category (and sub-categories up to one level deep).
    """
    titles: list[str] = []
    seen: set[str] = set()

    def _fetch_members(cat: str, ns: int) -> None:
        """Fetch members of a single category page (namespace *ns*)."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat}",
            "cmlimit": min(limit, 500),
            "cmnamespace": ns,   # 0 = articles, 14 = sub-categories
            "format": "json",
        }
        try:
            data = _http_get(SIMPLE_WIKI_API, params)
            members = data.get("query", {}).get("categorymembers", [])
            for m in members:
                title = m.get("title", "")
                if title and title not in seen:
                    seen.add(title)
                    if ns == 0:
                        titles.append(title)
        except Exception as exc:
            logger.warning("Could not fetch category '%s' (ns=%d): %s", cat, ns, exc)

    logger.info("Fetching article list from category '%s'...", category)
    _fetch_members(category, ns=0)   # direct articles

    # One level of sub-categories for broader coverage
    if len(titles) < limit:
        sub_params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": 20,
            "cmnamespace": 14,
            "format": "json",
        }
        try:
            sub_data = _http_get(SIMPLE_WIKI_API, sub_params)
            sub_cats = sub_data.get("query", {}).get("categorymembers", [])
            for sc in sub_cats:
                if len(titles) >= limit:
                    break
                sub_title = sc.get("title", "").replace("Category:", "")
                if sub_title:
                    _fetch_members(sub_title, ns=0)
        except Exception as exc:
            logger.warning("Could not expand sub-categories: %s", exc)

    logger.info("Found %d article titles (target: %d).", len(titles), limit)
    return titles[:limit]


def _fetch_article_extract(title: str) -> str | None:
    """
    Fetch the plain-text extract for a single Wikipedia article.

    Returns the raw extract string, or None on error.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,      # plain text, no HTML
        "exsectionformat": "plain",
        "redirects": 1,
        "format": "json",
    }
    try:
        data = _http_get(SIMPLE_WIKI_API, params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                return None
            return page.get("extract", None)
    except Exception as exc:
        logger.warning("Failed to fetch '%s': %s", title, exc)
    return None


# ── File helpers ───────────────────────────────────────────────────────────────

def _title_to_filename(title: str) -> str:
    """Convert an article title to a safe filesystem slug."""
    slug = title.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return f"{slug}.txt"


# ── Main fetch logic ───────────────────────────────────────────────────────────

def fetch_wikipedia_dataset(
    category: str = DEFAULT_CATEGORY,
    out_dir: str = DEFAULT_OUT_DIR,
    num_articles: int = DEFAULT_ARTICLES,
    delay: float = DEFAULT_DELAY,
    skip_existing: bool = True,
) -> dict:
    """
    Download and save Wikipedia Simple English articles to disk.

    Args:
        category:       Seed category name (without 'Category:' prefix).
        out_dir:        Directory to write ``.txt`` article files into.
        num_articles:   Target number of articles to download.
        delay:          Seconds to wait between API calls.
        skip_existing:  If True, skip articles whose output file already exists.

    Returns:
        Summary dict with keys: fetched, skipped, failed, too_short.
    """
    summary = {"fetched": 0, "skipped": 0, "failed": 0, "too_short": 0}

    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # Get article list
    titles = _get_category_members(category, limit=num_articles * 2)  # overfetch for stubs
    if not titles:
        logger.error("No articles found in category '%s'. Check the category name.", category)
        return summary

    fetched = 0
    for title in titles:
        if fetched >= num_articles:
            break

        filename = _title_to_filename(title)
        out_path = os.path.join(out_dir, filename)

        if skip_existing and os.path.exists(out_path):
            logger.info("  [SKIP] '%s' already saved.", filename)
            summary["skipped"] += 1
            continue

        logger.info("  [FETCH] '%s'...", title)
        extract = _fetch_article_extract(title)

        if not extract:
            logger.warning("  [FAIL] Could not retrieve extract for '%s'.", title)
            summary["failed"] += 1
            time.sleep(delay)
            continue

        cleaned = _clean_article_text(extract)

        if len(cleaned) < MIN_ARTICLE_CHARS:
            logger.info(
                "  [STUB] '%s' too short after cleaning (%d chars < %d) — skipping.",
                title, len(cleaned), MIN_ARTICLE_CHARS,
            )
            summary["too_short"] += 1
            time.sleep(delay)
            continue

        # Prepend title as the document header
        full_text = f"{title}\n{'=' * len(title)}\n\n{cleaned}"

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            logger.info(
                "  [OK] Saved '%s' (%d chars).", filename, len(full_text)
            )
            summary["fetched"] += 1
            fetched += 1
        except OSError as exc:
            logger.error("  [ERR] Could not write '%s': %s", out_path, exc)
            summary["failed"] += 1

        time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 45)
    logger.info("  Dataset Fetch Complete")
    logger.info("  Articles saved    : %d", summary["fetched"])
    logger.info("  Already existed   : %d", summary["skipped"])
    logger.info("  Too short / stubs : %d", summary["too_short"])
    logger.info("  Failed            : %d", summary["failed"])
    logger.info("  Output directory  : %s", out_dir)
    logger.info("=" * 45)
    logger.info("")
    if summary["fetched"] > 0:
        logger.info(
            "Next step: python ingest.py --dir \"%s\"", out_dir
        )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Wikipedia Simple English articles into the AetherRAG dataset directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=DEFAULT_ARTICLES,
        metavar="N",
        help="Number of articles to download.",
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        metavar="NAME",
        help="Seed Wikipedia category (without 'Category:' prefix).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help="Output directory for downloaded .txt files.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        metavar="SECS",
        help="Polite delay (seconds) between Wikipedia API calls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download articles even if output files already exist.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = fetch_wikipedia_dataset(
        category=args.category,
        out_dir=args.out,
        num_articles=args.articles,
        delay=args.delay,
        skip_existing=not args.force,
    )
    sys.exit(0 if result["failed"] == 0 else 1)

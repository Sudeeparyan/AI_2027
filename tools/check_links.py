"""Check every URL in curriculum/sources.yaml and every other URL used in the slides, notes, labs and course documents.

YouTube links are verified with the oEmbed endpoint, which returns the real video title and
fails for removed or mistyped videos (the normal watch page returns 200 even when a video is gone).
Writes build/link_check.json and prints failures.

Run: .venv/Scripts/python.exe tools/check_links.py
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import re
import sys
import urllib.parse
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) teaching-pack-link-check/1.0"}


def check(src: dict) -> dict:
    url = src["url"]
    out = {"id": src["id"], "url": url, "ok": False, "status": None, "title": None}
    try:
        if "youtube.com/watch" in url:
            o = requests.get("https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe=""), headers=UA, timeout=20)
            out["status"] = o.status_code
            if o.ok:
                data = o.json()
                out["title"] = data.get("title")
                out["author"] = data.get("author_name")
                out["ok"] = True
            return out
        r = requests.get(url, headers=UA, timeout=25, allow_redirects=True)
        out["status"] = r.status_code
        out["final_url"] = r.url
        # A blocked request is unverified, not evidence of a reachable page.
        out["ok"] = r.status_code < 400
        if r.status_code in (403, 429):
            out["blocked"] = True
            out["note"] = "blocked automated check; open manually"
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:200]
    return out


URL_RE = re.compile(r"https?://[^\s\"'<>)\]|`]+")
SKIP = ("localhost", "127.0.0.1", "generativelanguage.googleapis.com", "<server>", "<resource>", "example.net", "example.com", "northbridge.example")


def source_text(path: Path) -> str:
    """Read authored text after YAML has resolved quoted line continuations."""
    if path.suffix not in (".yaml", ".yml"):
        return path.read_text(encoding="utf-8")

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    return "\n".join(strings(yaml.safe_load(path.read_text(encoding="utf-8"))))


def content_urls(known: set[str]) -> list[dict]:
    """Every URL used in slides, notes, labs and course documents that is not already in the ledger."""
    files = [*(ROOT / "curriculum").glob("weeks/*.yaml"), *(ROOT / "curriculum").glob("notes/*.md"),
             *(ROOT / "curriculum").glob("labs/*.py"), *(ROOT / "curriculum").glob("course/*")]
    found: dict[str, str] = {}
    for f in sorted(files):
        for u in URL_RE.findall(source_text(f)):
            u = u.rstrip(".,;:*")
            if u not in known and not any(s in u for s in SKIP) and not u.endswith("/v1"):
                found.setdefault(u, f.name)
    return [{"id": f"{name}", "url": u} for u, name in sorted(found.items())]


def main() -> int:
    sources = yaml.safe_load((ROOT / "curriculum" / "sources.yaml").read_text(encoding="utf-8"))["sources"]
    sources += content_urls({s["url"] for s in sources})
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(check, sources))
    report = {"checked": dt.date.today().isoformat(), "results": results}
    out = ROOT / "build" / "link_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    bad = [r for r in results if not r["ok"] and not r.get("blocked")]
    blocked = [r for r in results if r.get("blocked")]
    for r in results:
        if r.get("title"):
            print(f"video  {r['id']}: {r['title']} ({r.get('author')})")
        elif r.get("note"):
            print(f"check  {r['id']}: {r['status']} {r['note']}")
    for r in bad:
        print(f"FAIL   {r['id']}: {r.get('status')} {r.get('error', '')} {r['url']}")
    print(f"{sum(r['ok'] for r in results)}/{len(results)} URLs reachable; {len(blocked)} blocked automated checks; {len(bad)} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

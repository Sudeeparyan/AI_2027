"""Change chosen keys of chosen slides in curriculum/weeks/week_XX.yaml and leave every other line as it was.

A full yaml.safe_dump would re-wrap and re-quote the whole file, which makes reviews hard. This helper
re-serialises only the keys you change, then parses the result and checks that nothing else moved.

Use it from a short script:

    import sys; sys.path.insert(0, "tools")
    from yaml_edit import edit_slides
    edit_slides(5, {8: {"notes": "Say: ..."}, 22: {"options": ["a", "b", "c", "d"]}})

Slide numbers count from 1, as in the built deck before the beginner slides are added. A value of None
deletes that key. The function returns the parsed slides after the edit.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _dump_key(key, value, first):
    text = yaml.safe_dump({key: value}, allow_unicode=True, sort_keys=False, width=120)
    lines = ["  " + line if line else line for line in text.rstrip("\n").split("\n")]
    if first:
        lines[0] = "- " + lines[0][2:]
    return [line + "\n" for line in lines]


def _slide_ranges(lines):
    start = next(i for i, line in enumerate(lines) if line.rstrip("\n") == "slides:") + 1
    starts = [i for i in range(start, len(lines)) if lines[i].startswith("- ")]
    end = next((i for i in range(start, len(lines)) if re.match(r"[A-Za-z_]", lines[i])), len(lines))
    return [(a, b) for a, b in zip(starts, starts[1:] + [end])]


def _key_range(lines, a, b, key):
    """Line range [i, j) of one top-level key inside the slide block lines[a:b]."""
    for i in range(a, b):
        if re.match(rf"(?:- |  ){re.escape(key)}:(?: |\n)", lines[i]):
            j = i + 1
            while j < b and (lines[j].startswith("   ") or lines[j].startswith("  - ") or not lines[j].strip()):
                j += 1
            return i, j
    return None


def edit_slides(week, changes, path=None):
    path = Path(path) if path else ROOT / "curriculum" / "weeks" / f"week_{week:02d}.yaml"
    raw = path.read_text(encoding="utf-8")
    before = yaml.safe_load(raw)
    expected = yaml.safe_load(raw)
    lines = raw.splitlines(keepends=True)
    ranges = _slide_ranges(lines)
    assert len(ranges) == len(before["slides"]), "could not split the slides list"
    # Work from the last slide back so earlier line numbers stay valid.
    for number in sorted(changes, reverse=True):
        a, b = ranges[number - 1]
        for key, value in changes[number].items():
            found = _key_range(lines, a, b, key)
            if value is None:
                expected["slides"][number - 1].pop(key, None)
            else:
                expected["slides"][number - 1][key] = value
            if found:
                i, j = found
                new = [] if value is None else _dump_key(key, value, lines[i].startswith("- "))
                if value is None and lines[i].startswith("- "):
                    raise ValueError(f"slide {number}: cannot delete the first key {key!r}")
                lines[i:j] = new
                b += len(new) - (j - i)
            elif value is not None:
                new = _dump_key(key, value, False)
                lines[b:b] = new
                b += len(new)
    out = "".join(lines)
    after = yaml.safe_load(out)
    if after != expected:
        bad = [i + 1 for i, (x, y) in enumerate(zip(after["slides"], expected["slides"])) if x != y]
        raise ValueError(f"edit changed more than requested (slides {bad}); file left untouched")
    path.write_text(out, encoding="utf-8")
    return after["slides"]

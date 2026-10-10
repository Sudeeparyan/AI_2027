"""Find typing slips in reader-facing prose that the other audits do not check.

It reads the teaching notes, the course guides, the slide YAML, the beginner JSON and the Markdown cells
of the labs. Code (fenced blocks, `code` spans, code fields), $...$ maths, URLs and link targets are
skipped. This audit reports:

1. MINUS: a negative number or "-ln" typed with a hyphen ("-0.5") instead of the minus sign (−0.5).
   TIMES: multiplication typed with the letter x ("8 x 4") instead of ×.
   APPROX: "≈80%" without the space that the rest of the pack uses ("≈ 80%").
   HYPHEN: a spaced hyphen inside a sentence ("1 + 1 - 0", "rules - and"); use − for subtraction, – for a dash.
2. RANGE: a number range typed with a hyphen ("5-8") instead of an en dash (5–8).
3. LOST: a "?", U+FFFD or a quote where a symbol was lost in an earlier copy ("x ? y", "decide\"act\"observe").
4. LOWER: a sentence that starts in lower case after a full stop ("... the loss. ways to reduce").
5. DOUBLE: a word typed twice ("the the").
6. REPEAT: the same speaker-note sentence on three or more slides of one week. Each slide's Ask, Answer,
   Analogy and Common mistake should be about that slide.
7. LABEL: "Common mistake: No." and similar, where an answer was labelled as a mistake.
8. TAG: a bare <tag> in Markdown prose (notes, guides, lab Markdown). Word and Jupyter treat it as HTML and
   drop it, so "use tags such as <document>" printed as "use tags such as". Put it in `code`.
   ARROW: an arrow typed as "->" or "=>" in prose instead of →.
9. QUIZ: a quiz whose notes give a different letter from its answer index, or one letter used for more
   than 40% of the course's quiz answers (21 of 24 were B before October 2026) or of the multiple-choice
   practice questions in the notes (33 of 45 were (b)). It also fails when, in more than a third of either
   set, the correct option is over 30% longer than every distractor (17 of 24 and 32 of 45 were).

ALLOW lists the few matches that are correct as written (document numbers, names that start in lower
case). Add to it only after reading the match in context.

This audit must report nothing before a release.

Usage: .venv/Scripts/python.exe tools/audit_prose.py
"""
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "curriculum"
# Fields that hold code, maths source, file names or ids rather than prose.
CODE_KEYS = {"code", "symbol", "eq", "figure", "icon", "url", "image", "id", "layout", "kind", "type", "src"}

MINUS = re.compile(r"(?:(?<=[\s\[(=,:/])|^)-(?=\d|ln\b|log\b|exp\b)")
TIMES = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s?x\s?[\d(]")  # "8 x 4" should be 8 × 4
APPROX = re.compile(r"≈(?=\d)")  # "≈80%" should be "≈ 80%"
HYPHEN = re.compile(r"(?<=[^\s|-]) - (?=\S)")  # not a list bullet or table cell
# A common word capitalised after a semicolon ("GPU; Some models"); proper nouns after a semicolon are fine.
SEMI_CAPITAL = re.compile(r"; (?:Some|The|This|That|These|It|Its|A|An|In|For|Use|They|We|Their|Check|Each|Most)\b")
RANGE = re.compile(r"(?<![\w.\-/−])(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?![\w.\-/%])")
LOST = re.compile(r"\?\s*[=<>×+−]|[=(]\s*\?(?![?)])|[A-Za-z]\?[A-Za-z]|\ufffd|[a-z]\"[a-z]+\"[a-z]")
LOWER = re.compile(r"([A-Za-z0-9)”’\"])\. ([a-z][a-z]+)")
ABBREV = re.compile(r"(?:\be\.g|\bi\.e|et al|\bvs|\betc|\bcf|\bapprox|\bch|\bFig|\bNo|\bp|\bpp|\bed|\bvol)\.$")
DOUBLE = re.compile(r"\b([A-Za-z]{2,})[ \t]+\1\b", re.I)
TAG = re.compile(r"<(/?[A-Za-z][\w-]*)(?:\s[^<>\n]*)?>")
ALLOWED_TAGS = {"br", "sub", "sup", "var"}  # inline markup the build understands
ARROW = re.compile(r"(?<![-<])->(?!>)|(?<![=<>!])=>")

# (check, text that may appear in the match context) pairs that were read and are correct.
ALLOW = [
    ("RANGE", "NIST AI 600-1"),          # a document number, not a range
    ("LOWER", "von Werra"),              # a surname that starts in lower case
    ("LOWER", "pgvector"),               # a product name that starts in lower case
    ("DOUBLE", "that that"),             # grammatical in "shows that that ..."
    ("HYPHEN", "Descriptor - Generative AI"),  # the descriptor's file name
]


def blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))


def clean(text, keep_tags=False):
    """Blank out code, maths, URLs and link targets (and <tags> unless kept), keeping offsets and line numbers."""
    text = re.sub(r"(```|~~~).*?\1", blank, text, flags=re.S)
    tags = "" if keep_tags else r"|<[^>\n]+>"
    return re.sub(r"\$\$.+?\$\$|\$[^$\n]+\$|`[^`\n]*`|https?://\S+|\]\([^)]*\)" + tags, blank, text, flags=re.S)


def walk(node, key=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v, key)
    elif isinstance(node, str) and key not in CODE_KEYS:
        yield key, node


def lab_markdown(path):
    """The Markdown cells of a percent-format lab script, with the leading '# ' removed."""
    out, inside = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            inside = "[markdown]" in line
            out.append("")
        elif inside and (line.startswith("# ") or line == "#"):
            out.append(line[2:])
        else:
            inside = inside and not line.strip()
            out.append("")
    return "\n".join(out)


def sources():
    for p in sorted((ROOT / "notes").glob("week_*.md")) + sorted((ROOT / "course").glob("*.md")):
        yield p.relative_to(ROOT.parent).as_posix(), p.read_text(encoding="utf-8"), True
    for p in sorted((ROOT / "labs").glob("week_*_lab.py")):
        yield p.relative_to(ROOT.parent).as_posix(), lab_markdown(p), True
    for p in sorted((ROOT / "weeks").glob("week_*.yaml")) + sorted((ROOT / "course").glob("*.yaml")):
        for key, value in walk(yaml.safe_load(p.read_text(encoding="utf-8"))):
            yield f"{p.relative_to(ROOT.parent).as_posix()} [{key}]", value, False
    for p in sorted((ROOT / "beginner").glob("week_*.json")):
        for key, value in walk(json.loads(p.read_text(encoding="utf-8"))):
            yield f"{p.relative_to(ROOT.parent).as_posix()} [{key}]", value, False


def problems():
    found = []
    for where, raw, numbered in sources():
        text = clean(raw)
        hits = []
        hits += [("MINUS", m) for m in MINUS.finditer(text)]
        hits += [("TIMES", m) for m in TIMES.finditer(text)]
        hits += [("APPROX", m) for m in APPROX.finditer(text)]
        hits += [("HYPHEN", m) for m in HYPHEN.finditer(text)]
        hits += [("CAPITAL", m) for m in SEMI_CAPITAL.finditer(text)]
        hits += [("RANGE", m) for m in RANGE.finditer(text)
                 if not (len(m.group(1)) == 4 and len(m.group(2)) in (2, 4))]  # years such as 2023-24
        hits += [("LOST", m) for m in LOST.finditer(text)]
        hits += [("LOWER", m) for m in LOWER.finditer(text)
                 if not ABBREV.search(text[max(0, m.start() - 12):m.start() + 2].strip())
                 and not (m.group(1).isdigit() and text[max(0, m.start() - 1):m.start()].strip() == "")]  # "1. build"
        hits += [("DOUBLE", m) for m in DOUBLE.finditer(text)]
        with_tags = clean(raw, keep_tags=True)
        hits += [("ARROW", m) for m in ARROW.finditer(with_tags)]
        if numbered:  # Markdown sources; slide YAML and JSON text is placed literally
            hits += [("TAG", m) for m in TAG.finditer(with_tags) if m.group(1).strip("/").lower() not in ALLOWED_TAGS]
        for check, m in hits:
            context = raw[max(0, m.start() - 40):m.end() + 25].replace("\n", " ")
            if any(check == c and allowed in context for c, allowed in ALLOW):
                continue
            line = f":{raw[:m.start()].count(chr(10)) + 1}" if numbered else ""
            found.append(f"{check:6} {where}{line}: ...{context}...")
    return found


def repeated_notes():
    """Speaker-note sentences pasted onto three or more slides of one week, and answers labelled as mistakes."""
    found = []
    for p in sorted((ROOT / "weeks").glob("week_*.yaml")):
        slides = yaml.safe_load(p.read_text(encoding="utf-8"))["slides"]
        seen = {}
        for i, s in enumerate(slides, 1):
            notes = str(s.get("notes", ""))
            for sentence in {x.strip() for x in re.split(r"(?<=[.?!])\s+", notes) if len(x.split()) >= 7}:
                seen.setdefault(sentence, []).append(i)
            for m in re.finditer(r"Common mistake: (?:Yes|No)\.", notes):
                found.append(f"LABEL  {p.relative_to(ROOT.parent).as_posix()} slide {i}: '{m.group(0)}' is an answer, "
                             "not a mistake")
        for sentence, where in seen.items():
            if len(where) >= 3:
                found.append(f"REPEAT {p.relative_to(ROOT.parent).as_posix()} slides {where}: {sentence[:90]}")
    return found


def quiz_answers():
    """Each quiz's notes must name the letter its answer index marks, and no letter may dominate the course."""
    found, letters, cues = [], [], {"quiz": [], "practice-question": []}
    for p in sorted((ROOT / "weeks").glob("week_*.yaml")):
        for i, s in enumerate(yaml.safe_load(p.read_text(encoding="utf-8"))["slides"], 1):
            if s.get("type") != "quiz":
                continue
            letter = "ABCD"[s["answer"]]
            letters.append(letter)
            cues["quiz"].append(longest_is_answer(s["options"], s["answer"]))
            m = re.search(r"Answer:?\s+([A-D])\b", str(s.get("notes", "")))
            if not m or m.group(1) != letter:
                found.append(f"QUIZ   {p.relative_to(ROOT.parent).as_posix()} slide {i}: answer index is {letter}, "
                             f"notes say {m.group(1) if m else 'nothing'}")
    practice = []  # "1. ... **(a)** ...; **(b)** ... *Answer: (b).*" lines in the notes' practice questions
    for p in sorted((ROOT / "notes").glob("week_*.md")):
        for line in re.findall(r"^\d\. .*\*\*\(d\)\*\* .*\*Answer: \([a-d]\).*$", p.read_text(encoding="utf-8"), re.M):
            answer = re.search(r"\*Answer: \(([a-d])\)", line).group(1)
            practice.append(answer)
            options = re.findall(r"\*\*\([a-d]\)\*\* (.*?)(?=[;.] \*\*\(|\. \*Answer)", line)
            cues["practice-question"].append(len(options) == 4 and longest_is_answer(options, "abcd".index(answer)))
    for name, answers in (("quiz", letters), ("practice-question", [a.upper() for a in practice])):
        for letter in "ABCD":
            if answers.count(letter) > 0.4 * len(answers):
                found.append(f"QUIZ   {answers.count(letter)} of {len(answers)} {name} answers are {letter}; "
                             "students learn the pattern")
        if sum(cues[name]) > len(cues[name]) / 3:
            found.append(f"QUIZ   in {sum(cues[name])} of {len(cues[name])} {name}s the answer is clearly the longest "
                         "option; write the distractors as fully as the answer")
    return found


def longest_is_answer(options, answer):
    """True when the correct option is longer than every other option by more than 30% (a guessing cue)."""
    lengths = [len(re.sub(r"\\[A-Za-z]+|[${}`*_^]", "", str(o))) for o in options]  # length as printed
    others = lengths[:answer] + lengths[answer + 1:]
    return lengths[answer] > 1.3 * max(others)


def main():
    found = problems() + repeated_notes() + quiz_answers()
    for f in found:
        print(f)
    print(f"{len(found)} prose problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())

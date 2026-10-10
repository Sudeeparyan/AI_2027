"""Find maths notation that would reach readers unconverted.

The builder turns inline $...$ maths into Unicode, with Word/PowerPoint sub- and superscripts where Unicode has
no script letter (tools/build_week.py: inline_math, used by prepare_notes for the notes and slide_math for the
projected slide text). This audit reports:

1. $...$ expressions with LaTeX commands the builder cannot convert (they would print as words, e.g. "sqrt2")
   or with ^/_ left over after conversion.
2. Maths-style scripts typed as plain text outside $...$ and code, such as q_φ, x_t, p_data or x^2, in the
   teaching notes, in the projected slide text and in the diagram text. Readers would see a raw _ or ^.
   Snake_case code names (causal_mask) and constants (Z_DIM) are not flagged.
3. A $ that opens maths but never closes, which would print as a dollar sign. Prices ($0.10) are fine.

This audit must report nothing before a release.

Usage: .venv/Scripts/python.exe tools/audit_inline_math.py
"""
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_week as bw  # noqa: E402

# One letter (Latin or Greek) then _ or ^ and a short script: x_t, q_φ, z_ij, p_data, x^2, x_{t-1}, e^2.41.
RAW_SCRIPT = re.compile(r"(?<![A-Za-z0-9_.\\])[A-Za-zΑ-Ωα-ωϕϑ][_^](?:\{[^{}\s]*\}|[0-9.]+|[A-Za-zΑ-Ωα-ωϕϑ0-9]{1,4}\b)")
UNCLOSED = re.compile(r"(?<![\\$])\$(?=[A-Za-z\\])")
# A script after a bracket, such as (a + b)^2 or f(x)_i, would also print a raw ^ or _.
BRACKET_SCRIPT = re.compile(r"[)\]][\^_](?:\{[^{}\s]*\}|[0-9A-Za-z])")
# ASCII stand-ins for maths in prose: pi_theta, abar_t, epsilon_theta, r_phi, sqrt(2).
ASCII_MATH = re.compile(r"\b(?:alpha|abar|beta|gamma|delta|epsilon|theta|lambda|sigma|phi|pi|tau|eta|rho|omega)_\w+"
                        r"|\b[A-Za-z]+_(?:theta|phi)\b|\bsqrt\(")
# Maths symbols spelled as words in prose: "epsilon = -0.5", "abar = 0.64", "x0 is clean data", "temperature tau".
# Code names belong in `code` spans (or code fields); "mu" is left alone because labs name encoder outputs mu, logvar.
# A name in brackets after its symbol, "θ (theta)", is a pronunciation guide; Beta-VAE and the Phi models are names.
# Python counts "²" as a word character, so \b never matches in "sigma²". A name with a superscript is always maths,
# never a pronunciation guide or a code name, so "(mu² + sigma²" is reported even after a bracket.
WORD_MATH = re.compile(r"\b(?:abar|alphabar|eps|epsilon|alpha|beta|sigma|theta|lambda|eta|tau|phi|mu)(?=[²³])"
                       r"|(?<!\()(?:\b(?:abar|alphabar|eps|epsilon|alpha|beta|sigma|theta|lambda|eta|tau|phi|x0|xt)\b"
                       r"|\b(?:Alpha|Beta|Epsilon|Lambda|Mu|Sigma|Tau|Theta)\b(?!-))(?!\))")
CODE_KEYS = {"code", "symbol"}  # beginner JSON fields that hold code names


def unknown_commands(expr: str) -> list[str]:
    cmds = re.findall(r"\\([A-Za-z]+)", expr)
    return sorted({c for c in cmds if c not in bw.SYMBOLS and c not in bw.STRUCTURAL})


def is_constant(token: str) -> bool:
    """Z_DIM-style code constants: an upper-case letter and a script of two or more capitals."""
    return bool(re.fullmatch(r"[A-Z]_[A-Z0-9]{2,}", token))


def check_text(text: str, where) -> int:
    """Report problems in one piece of reader-facing text; `where(offset)` names the location."""
    bad = 0
    for m in bw.INLINE_MATH.finditer(text):
        unk = unknown_commands(m.group(1))
        converted = bw.inline_math(m.group(1))
        raw = re.findall(r"[\^_]", re.sub(r"</?su[bp]>", "", converted))
        if unk or raw:
            bad += 1
            print(f"{where(m.start())}: {m.group(1)!r} -> {converted!r}  unknown: {unk}  raw ^/_: {len(raw)}")
    rest = bw.INLINE_MATH.sub(lambda m: " " * len(m.group(0)), text)
    for m in list(RAW_SCRIPT.finditer(rest)) + list(BRACKET_SCRIPT.finditer(rest)):
        if not is_constant(m.group(0)):
            bad += 1
            print(f"{where(m.start())}: plain-text script {m.group(0)!r}; write it as $...$ maths")
    for m in ASCII_MATH.finditer(rest):
        bad += 1
        print(f"{where(m.start())}: ASCII maths {m.group(0)!r}; write it as $...$ maths")
    for m in WORD_MATH.finditer(rest):
        bad += 1
        print(f"{where(m.start())}: symbol spelled as a word {m.group(0)!r}; write it as $...$ maths or `code`")
    for m in UNCLOSED.finditer(rest):
        bad += 1
        print(f"{where(m.start())}: unclosed $ maths near {rest[m.start():m.start() + 25]!r}")
    return bad


def plain_markdown(text: str) -> str:
    """Notes text without code, display maths and link targets, keeping line positions."""
    blank = lambda m: re.sub(r"[^\n]", " ", m.group(0))  # noqa: E731
    text = re.sub(r"```.*?```", blank, text, flags=re.S)
    text = re.sub(r"\$\$.+?\$\$", blank, text, flags=re.S)  # display maths is rendered as images
    text = re.sub(r"`[^`\n]+`", blank, text)
    text = re.sub(r"\]\([^)\s]+\)", blank, text)
    text = re.sub(r"<!--.*?-->", blank, text, flags=re.S)
    return text


def visible_strings(node, key=""):
    if key in bw.SLIDE_TEXT_SKIP:
        return
    if isinstance(node, str):
        yield key, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from visible_strings(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from visible_strings(v, key)


def main() -> int:
    bad = 0
    for p in sorted((bw.CUR / "notes").glob("week_*.md")):
        text = plain_markdown(p.read_text(encoding="utf-8"))
        bad += check_text(text, lambda i, p=p, text=text: f"{p.name}:{text[:i].count(chr(10)) + 1}")
    for p in sorted((bw.CUR / "weeks").glob("week_*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        for n, slide in enumerate(data.get("slides", []), start=1):
            for key, value in visible_strings(slide):
                value = re.sub(r"`[^`]+`", lambda m: " " * len(m.group(0)), value)
                bad += check_text(value, lambda i, p=p, n=n, s=slide, k=key: f"{p.name} slide {n} ({s.get('type')}) {k}")
    for p in sorted((bw.CUR / "beginner").glob("week_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for key, value in visible_strings(data):
            # Diagram text is drawn as measured plain text, so it cannot use $...$ either. Code-map cells are
            # Markdown, where `code` spans are allowed.
            value = re.sub(r"`[^`]+`", lambda m: " " * len(m.group(0)), value)
            for m in list(RAW_SCRIPT.finditer(value)) + list(BRACKET_SCRIPT.finditer(value)):
                if not is_constant(m.group(0)):
                    bad += 1
                    print(f"{p.name} {key}: plain-text script {m.group(0)!r}; use Unicode")
            if key not in CODE_KEYS:
                for m in WORD_MATH.finditer(value):
                    bad += 1
                    print(f"{p.name} {key}: symbol spelled as a word {m.group(0)!r}; use the Unicode symbol or words")
            if "$" in value and bw.INLINE_MATH.search(value):
                bad += 1
                print(f"{p.name} {key}: $...$ maths is not converted in diagrams; use Unicode")
    print(f"{bad} maths notation problems need attention")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

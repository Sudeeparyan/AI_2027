"""List inline $...$ expressions in the teaching notes whose LaTeX commands the notes builder cannot convert.

The notes builder turns inline math into Unicode (tools/build_week.py: inline_math). Commands it does not know
would be printed as plain words (e.g. "sqrt2"), so this audit must report nothing before a release.

Usage: .venv/Scripts/python.exe tools/audit_inline_math.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_week as bw  # noqa: E402

def unknown_commands(expr: str) -> list[str]:
    cmds = re.findall(r"\\([A-Za-z]+)", expr)
    return sorted({c for c in cmds if c not in bw.SYMBOLS and c not in bw.STRUCTURAL})


def main() -> int:
    bad = 0
    for p in sorted((bw.CUR / "notes").glob("week_*.md")):
        text = re.sub(r"\$\$(.+?)\$\$", "", p.read_text(encoding="utf-8"), flags=re.S)  # display math is rendered as images
        for m in re.finditer(r"(?<![\\$])\$([^$\n]+?)\$", text):
            unk = unknown_commands(m.group(1))
            if unk:
                bad += 1
                line = text[: m.start()].count("\n") + 1
                print(f"{p.name}:{line}: {m.group(1)!r} -> {bw.inline_math(m.group(1))!r}  unknown: {unk}")
    print(f"{bad} inline expressions need attention")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

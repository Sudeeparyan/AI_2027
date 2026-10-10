"""Render generated course documents with the repository's Office pipeline."""
from pathlib import Path
import shutil
import subprocess

from render import contact_sheets

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "build/render/course"
    directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / f"deliverables/00_Course/{name}.docx" for name in
               ("Instructor_Guide", "Course_Glossary", "Section_7.3_Rationale")]
    copies = []
    for source in sources:
        destination = directory / source.name
        shutil.copy2(source, destination)
        copies.append(destination)
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "tools/render_office.ps1"),
                    *map(str, copies)], check=True, timeout=900)
    for copy in copies:
        prefix = copy.stem
        for old in directory.glob(f"{prefix}-*.png"):
            old.unlink()
        subprocess.run(["pdftoppm", "-png", "-r", "100", str(copy.with_suffix(".pdf")), str(directory / prefix)], check=True)
        pages = sorted(directory.glob(f"{prefix}-*.png"))
        contact_sheets(pages, directory, prefix=prefix, per=6, cols=3)
        print(f"{prefix}: {len(pages)} pages")


if __name__ == "__main__":
    main()

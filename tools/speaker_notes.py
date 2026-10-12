"""Shared speaker-note migration checks; this module does not read or write files.

Word limits include the six printed labels, matching ``len(notes.split())`` in
the weekly build reports. Set STRICT_LEGACY_NOTES at the final migration gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

LABELS = ("In short", "Explain", "Example", "Ask", "Answer", "Watch out")
MIN_WORDS, MAX_WORDS, MIN_LEGACY_WORDS = 110, 180, 45
STRICT_LEGACY_NOTES = True
LABEL_RE = re.compile(
    r"(?<!\w)(?:\*\*)?(In short|Explain|Example|Ask|Answer|Watch out)(?:\*\*)?:(?:\*\*)?\s*",
    re.I,
)
LEGACY_RE = re.compile(r"\b(?:Say|Analogy|Common mistake):", re.I)
ALL_LABEL_RE = re.compile(
    r"(?<!\w)(?:\*\*)?(?:In short|Explain|Example|Ask(?: students| class)?|"
    r"(?:Expected )?Answers?|Watch out|Say|Analogy|Common mistake)(?:\*\*)?:(?:\*\*)?\s*",
    re.I,
)
CANONICAL = {label.lower(): label for label in LABELS}


@dataclass
class NoteValidation:
    modern: bool
    words: int
    sections: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def modern_note(text: str) -> bool:
    """Only labels unique to the new template signal migration, not legacy Ask/Answer cues."""
    return any(match.group(1).lower() in {"in short", "explain", "watch out"}
               for match in LABEL_RE.finditer(text))


def note_sections(text: str) -> dict[str, str]:
    """Read plain or bold section labels; retain the first copy of duplicates."""
    matches = list(LABEL_RE.finditer(text))
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        label = CANONICAL[match.group(1).lower()]
        sections.setdefault(label, text[match.end():end].strip())
    return sections


def note_sentences(text: str) -> list[str]:
    """Split teaching prose without treating decimal points or common abbreviations as endings."""
    protected = re.sub(
        r"`[^`]*`|\b(?:e\.g\.|i\.e\.|et al\.|vs\.|etc\.|Fig\.|No\.|Dr\.)",
        lambda match: match.group(0).replace(".", "\u0000"), text,
    )
    return [part.replace("\u0000", ".").strip() for part in
            re.split(r"(?<=[.!?])[\"'”’)]*\s+", protected) if part.strip()]


def quiz_answer_letter(text: str) -> str | None:
    """Keep the existing Answer: X convention, also accepting bold section labels."""
    if modern_note(text):
        answer = note_sections(text).get("Answer", "").lstrip("*")
        match = re.match(r"([A-D])\b", answer)
    else:
        # Legacy quizzes can begin "Say: Answer D" and later answer a different
        # follow-up question. Preserve the original first-letter interpretation.
        match = re.search(r"\bAnswer(?:\*\*)?:?(?:\*\*)?\s+([A-D])\b", text)
    return match.group(1) if match else None


def validate_speaker_notes(text: str, *, strict_legacy: bool | None = None,
                           quiz_answer: int | None = None) -> NoteValidation:
    """Malformed modern notes fail immediately; legacy formatting warns during migration."""
    text = str(text)
    matches = list(LABEL_RE.finditer(text))
    labels = [CANONICAL[match.group(1).lower()] for match in matches]
    legacy = bool(LEGACY_RE.search(text))
    modern = modern_note(text)
    result = NoteValidation(modern, len(text.split()), note_sections(text))
    strict_legacy = STRICT_LEGACY_NOTES if strict_legacy is None else strict_legacy
    if not modern:
        message = "legacy speaker-note format; use In short/Explain/Example/Ask/Answer/Watch out"
        (result.errors if strict_legacy else result.warnings).append(message)
        if result.words < MIN_LEGACY_WORDS:
            result.errors.append(f"speaker notes only {result.words} words (legacy minimum {MIN_LEGACY_WORDS})")
    else:
        if legacy:
            result.errors.append("speaker notes mix modern and legacy section labels")
        for label in LABELS:
            count = labels.count(label)
            if count != 1:
                result.errors.append(f"speaker-note label '{label}:' occurs {count} times; expected exactly once")
        if len(labels) == len(LABELS) and len(set(labels)) == len(LABELS) and tuple(labels) != LABELS:
            result.errors.append("speaker-note labels are out of order; expected " + " -> ".join(LABELS))
        if matches and text[:matches[0].start()].strip():
            result.errors.append("speaker notes have text before the In short label")
        for label, content in result.sections.items():
            if not content:
                result.errors.append(f"speaker-note section '{label}:' is empty")
        if not MIN_WORDS <= result.words <= MAX_WORDS:
            result.errors.append(f"speaker notes have {result.words} words; expected {MIN_WORDS}–{MAX_WORDS} including labels")
        if result.sections.get("In short") and len(note_sentences(result.sections["In short"])) != 1:
            result.errors.append("In short must contain one sentence")
        if result.sections.get("Explain"):
            sentences = note_sentences(result.sections["Explain"])
            if not 3 <= len(sentences) <= 6:
                result.errors.append(f"Explain has {len(sentences)} sentences; expected 3–6")
            for index, sentence in enumerate(sentences, 1):
                words = len(sentence.split())
                if words > 20:
                    result.errors.append(f"Explain sentence {index} has {words} words; maximum is 20")
    if quiz_answer is not None:
        if isinstance(quiz_answer, bool) or not isinstance(quiz_answer, int) or not 0 <= quiz_answer < 4:
            result.errors.append("quiz answer index must be an integer from 0 to 3")
        else:
            expected, actual = "ABCD"[quiz_answer], quiz_answer_letter(text)
            if actual != expected:
                result.errors.append(f"quiz answer index is {expected}; notes say {actual or 'nothing'}")
    return result


def legacy_warning(slide_numbers: list[int]) -> str | None:
    """One migration warning per week rather than one message per slide."""
    if not slide_numbers:
        return None
    return (f"{len(slide_numbers)} slide(s) use legacy speaker-note formatting; "
            "migrate to the six-section, 110–180-word template before the final gate")

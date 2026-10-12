"""Authored lecture flows shared by editable slides and portable diagrams.

Unannotated legacy flows keep their existing layout. Adding arrows or a role
opts into the strict scene schema; partial annotations must not silently fall
back to an unlabelled diagram.
"""
from __future__ import annotations

import re

ROLES = {"data", "model", "loss", "metric", "output", "tool", "input", "human"}


def annotated(slide: dict) -> bool:
    steps = slide.get("steps", [])
    return slide.get("type") == "flow" and (
        "arrows" in slide or (isinstance(steps, list) and any(isinstance(s, dict) and "role" in s for s in steps)))


def errors(slide: dict) -> list[str]:
    if not annotated(slide):
        return []
    result = []
    steps = slide.get("steps")
    if not isinstance(steps, list) or not 2 <= len(steps) <= 7:
        return ["annotated flow needs 2–7 steps"]
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            result.append(f"flow step {index + 1} must be an object")
            continue
        if not isinstance(step.get("label"), str) or not step["label"].strip():
            result.append(f"flow step {index + 1} needs a real operation label")
        if "sub" in step and not isinstance(step["sub"], str):
            result.append(f"flow step {index + 1} sub must be a string")
        if not isinstance(step.get("role"), str) or step["role"] not in ROLES:
            result.append(f"flow step {index + 1} has an unknown or missing role")
    arrows = slide.get("arrows")
    if not isinstance(arrows, list) or len(arrows) != len(steps) - 1:
        result.append("flow needs exactly n−1 arrows, with null for independent inputs")
        arrows = []
    elif any(a is not None and (not isinstance(a, str) or not a.strip()) for a in arrows):
        result.append("flow data arrows need nonempty labels or null")
    touched = {i + offset for i, arrow in enumerate(arrows) if isinstance(arrow, str) and arrow.strip()
               for offset in (0, 1)}
    branches = slide.get("branches", [])
    if not isinstance(branches, list):
        result.append("flow branches must be a list")
        branches = []
    for branch in branches:
        if not isinstance(branch, dict):
            result.append("flow branch must be an object")
            continue
        if any(type(branch.get(k)) is not int or not 0 <= branch[k] < len(steps) for k in ("from", "to")):
            result.append("flow branch endpoint is outside the diagram")
        else:
            touched.update((branch["from"], branch["to"]))
            if branch["to"] <= branch["from"]:
                result.append("flow data branches move forward; use feedback for repetition")
        if not isinstance(branch.get("label"), str) or not branch["label"].strip():
            result.append("flow branch needs a nonempty data label")
    if touched != set(range(len(steps))):
        result.append("flow has a disconnected block; label its data arrow or branch")
    if "feedback" in slide:
        if not isinstance(slide["feedback"], str) or not slide["feedback"].strip():
            result.append("flow feedback needs a nonempty repetition label")
        for key, default in (("feedback_from", len(steps) - 1), ("feedback_to", 0)):
            value = slide.get(key, default)
            if type(value) is not int or not 0 <= value < len(steps):
                result.append(f"flow {key} is outside the diagram")
    return result


def collect(week: dict) -> list[dict]:
    """Select strict flows and give them stable IDs from built slide positions."""
    selected = []
    for number, slide in enumerate(week.get("slides", []), 1):
        if not annotated(slide):
            continue
        problems = errors(slide)
        if problems:
            raise ValueError(f"Slide {number}: " + "; ".join(problems))
        identifier = f"lecture_flow_{number:02d}"
        slide["flow_asset"] = identifier
        selected.append({"id": identifier, "slide": slide})
    return selected


def notes_with_flows(markdown: str, slides: list[dict]) -> str:
    """Place the same image beside its exact companion section, once only."""
    additions = []
    for number, slide in enumerate(slides, 1):
        identifier = slide.get("flow_asset")
        if not identifier or re.search(r"\]\(fig:" + re.escape(identifier) + r"\)", markdown):
            continue
        image = f"![{slide['title']}](fig:{identifier})\n\n"
        heading = re.compile(r"(?m)^## Slide " + str(number) + r": " + re.escape(slide["title"]) + r"\s*$")
        match = heading.search(markdown)
        if match:
            # Keep the exact heading and all its authored content. The image
            # belongs to this section rather than a second slide companion.
            markdown = markdown[:match.end()] + "\n\n" + image + markdown[match.end():].lstrip("\r\n")
        else:
            additions.append(f"## Lecture flow: {slide['title']}\n\n" + image)
    if additions:
        section = "\n# Lecture flow diagrams\n\n" + "\n".join(additions)
        recap = re.search(r"(?m)^# Class recap\s*$", markdown)
        position = recap.start() if recap else len(markdown)
        markdown = markdown[:position].rstrip() + "\n\n" + section + "\n" + markdown[position:]
    return markdown


def asset_ids(spec: dict | None) -> set[str]:
    return {s["flow_asset"] for s in (spec or {}).get("slides", []) if s.get("flow_asset")}

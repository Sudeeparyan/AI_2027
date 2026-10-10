"""Compatibility entry point: fill measured Week 5 values without rewriting sources.

Prefer tools/fill_results.py --weeks 5. Keep placeholders in the authored YAML
and Markdown so every rebuild reads the same archived measurements.
"""
import json

from fill_results import ASSETS, week5


def main() -> int:
    output = ASSETS / "week_05" / "fill.json"
    output.write_text(json.dumps(week5(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"week 5: measured values -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

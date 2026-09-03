#!/usr/bin/env python3
"""Screen a pool of phrasings against a configured model.

    # offline dry run, no model needed
    python examples/01_screen_pool.py --config config.example.yaml --reps 8

    # against a local vLLM / llama.cpp / Ollama server
    python examples/01_screen_pool.py --config my.yaml --reps 40 --out kept.json

Config format: see config.example.yaml. The `backend` block is passed straight to
`load_backend`, so anything that module accepts works here without code changes.

Sizing note. `reps` is the number of times each variant is measured. Fire-rate is the
expensive thing to bound: at reps=8 you can only distinguish "always fires" from "fails
often". Screen wide at reps=8-16 to throw out the obvious failures, then confirm the
survivors at reps>=40 before you ship them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptlab import Segment, Template, ToolCallObjective, load_backend, screen_pool


def load_config(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            raise SystemExit("YAML config needs: pip install pyyaml (or use a .json config)") from None
        return yaml.safe_load(text)
    return json.loads(text)


def build_template(cfg: dict) -> Template:
    parts: list[str | Segment] = []
    for part in cfg["template"]:
        if isinstance(part, str):
            parts.append(part)
            continue
        if part.get("derive_from"):
            src, sep = part["derive_from"], part.get("derive_split", "@")
            parts.append(
                Segment(part["name"], derive=lambda c, s=src, p=sep: c[s].split(p)[0])
            )
        else:
            parts.append(
                Segment(part["name"], part["values"],
                        period=part.get("period"), stride=part.get("stride", 1))
            )
    return Template(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--segment", help="which segment to screen (default: config's screen.segment)")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="keep variants within (1+tol) x the control's token cost")
    ap.add_argument("--allow-partial-fire", action="store_true",
                    help="do not require 100%% fire (you almost never want this)")
    ap.add_argument("--out", type=Path, help="write the kept pool to this JSON file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend = load_backend(cfg["backend"])
    template = build_template(cfg)
    screen_cfg = cfg["screen"]
    segment = args.segment or screen_cfg["segment"]

    obj_cfg = cfg["objective"]
    objective = ToolCallObjective(
        tool_name=obj_cfg["tool"],
        required_args=obj_cfg.get("required_args", []),
        per_call=obj_cfg.get("per_call", 1.0),
        max_calls=obj_cfg.get("max_calls", 1),
        ceiling=obj_cfg.get("ceiling", 1.0),
    )

    banned = [b.lower() for b in cfg.get("forbid_substrings", [])]

    def forbid(msg: str) -> str | None:
        low = msg.lower()
        hit = next((b for b in banned if b in low), None)
        return f"assembled message contains {hit!r}" if hit else None

    print(f"backend  : {backend.name}")
    print(f"segment  : {segment}   variants: {len(screen_cfg['variants'])}   reps: {args.reps}")
    print(f"example  : {template.render(0)}")
    print()

    def progress(done: int, total: int) -> None:
        if done % max(1, total // 20) == 0 or done == total:
            print(f"\r  measuring {done}/{total}", end="", file=sys.stderr, flush=True)

    report = screen_pool(
        backend,
        template,
        segment=segment,
        variants=screen_cfg["variants"],
        objective=objective,
        reps=args.reps,
        control=screen_cfg.get("control"),
        token_tolerance=args.tolerance,
        require_full_fire=not args.allow_partial_fire,
        system=cfg.get("system"),
        tools=cfg.get("tools"),
        forbid=forbid if banned else None,
        progress=progress,
    )
    print("\r" + " " * 40 + "\r", end="", file=sys.stderr, flush=True)
    print(report.summary())
    print()
    print(report.table(limit=None))

    if args.reps < 40 and report.kept:
        print()
        print(f"NOTE: reps={args.reps} bounds fire rate only loosely. Re-confirm these "
              f"{len(report.kept)} survivors at reps>=40 before shipping them.")

    if args.out:
        args.out.write_text(json.dumps({
            "segment": segment,
            "kept": report.kept,
            "rejected": dict(report.rejected),
            "control_tokens": report.control_tokens,
            "reps": args.reps,
            "token_reliability": report.token_reliability,
        }, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

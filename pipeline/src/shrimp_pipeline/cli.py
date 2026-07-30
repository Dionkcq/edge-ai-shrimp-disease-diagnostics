# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shrimp_pipeline.convert import ConversionError, generate_evidence_report, prepare_archive
from shrimp_pipeline.gate import MappingGateError
from shrimp_pipeline.manifest import inventory_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shrimp-pipeline",
        description="Fail-closed shrimp dataset audit and preparation tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="hash and inventory source ZIP archives read-only")
    audit.add_argument("archives", nargs="+", type=Path)
    audit.add_argument("--output", type=Path)

    evidence = sub.add_parser("evidence", help="render supplied boxes for human review")
    evidence.add_argument("archive", type=Path)
    evidence.add_argument("output", type=Path)
    evidence.add_argument("--minimum-overlays", type=int, default=60)
    evidence.add_argument("--seed", type=int, default=20260730)

    prepare = sub.add_parser("prepare", help="convert a verified archive after the human gate")
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument(
        "--acceptance", type=Path, default=Path("datasets/mapping_acceptance.json")
    )
    prepare.add_argument("--seed", type=int, default=20260730)
    sub.add_parser("train", help="report unavailable training; no weights exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        reports = [inventory_archive(path).to_dict() for path in args.archives]
        rendered = json.dumps({"schema_version": "2.0.0", "archives": reports}, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    if args.command == "evidence":
        try:
            report = generate_evidence_report(
                args.archive, args.output, minimum_overlays=args.minimum_overlays, seed=args.seed
            )
        except (ConversionError, FileExistsError, OSError, ValueError) as exc:
            print(f"REFUSED: {exc}")
            return 2
        print(f"Evidence generated without human acceptance: {report}")
        return 0
    if args.command == "prepare":
        try:
            result = prepare_archive(args.archive, args.acceptance, args.output, seed=args.seed)
        except (MappingGateError, ConversionError, FileExistsError, OSError) as exc:
            print(f"REFUSED: {exc}")
            return 2
        print(f"Prepared {result.canonical_images} canonical images: {result.manifest}")
        return 0
    print(
        "UNAVAILABLE: no trained weights or Ultralytics training environment is installed. "
        "See models/MODEL_CARD.md."
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

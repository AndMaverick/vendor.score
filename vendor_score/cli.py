"""Command line: ``python -m vendor_score analyze scenario.json``.

Exit codes: 0 gates pass (or none set), 1 a gate fails, 2 the input is unusable."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import report, scenario
from ._version import __version__
from .values import SpecError


def _load(path: str) -> dict:  # type: ignore[type-arg]
    try:
        with open(path, encoding="utf-8") as fh:
            return scenario.loads(fh.read())
    except OSError as exc:
        raise SpecError(f"cannot read {path}: {exc.strerror}") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vendor-score", description="Stress-test a vendor scorecard before it goes to the board.")
    p.add_argument("--version", action="version", version=f"vendor.score {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="full analysis with gates")
    a.add_argument("scenario")
    a.add_argument("--json", action="store_true", help="print the result as JSON")
    a.add_argument("--trials", type=int, help="override simulation.trials")
    a.add_argument("--seed", type=int, help="override simulation.seed")
    f = sub.add_parser("fingerprint", help="print the SHA-256 fingerprint of a scenario")
    f.add_argument("scenario")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = _load(args.scenario)
        if args.command == "fingerprint":
            scenario.analyze(data)             # refuse to fingerprint a scenario that does not parse
            print(scenario.fingerprint(data))
            return 0
        if args.trials is not None or args.seed is not None:
            sim = dict(data.get("simulation") or {})
            if args.trials is not None:
                sim["trials"] = args.trials
            if args.seed is not None:
                sim["seed"] = args.seed
            data = {**data, "simulation": sim}
        result = scenario.analyze(data)
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report.to_json(result) if args.json else report.to_text(result), end="" if not args.json else "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

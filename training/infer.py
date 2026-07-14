#!/usr/bin/env python3
"""Deprecated legacy inference entrypoint."""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "ERROR: training/infer.py has been retired. "
        "Use full-pipeline/run.py for downstream inference or train.py --evaluate for checkpoint evaluation."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

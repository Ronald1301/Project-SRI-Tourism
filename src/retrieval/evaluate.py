"""Compatibility wrapper for the modular evaluation package.

The implementation lives in ``src.evaluation``. This file keeps old
imports such as ``from src.retrieval.evaluate import evaluate_systems`` working.
"""

from src.evaluation import *  # noqa: F401,F403
from src.evaluation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

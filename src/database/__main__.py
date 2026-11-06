"""Run with `python -m src.database`."""

import asyncio
import sys

from .cli import run_cli


if __name__ == "__main__":
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run_cli())

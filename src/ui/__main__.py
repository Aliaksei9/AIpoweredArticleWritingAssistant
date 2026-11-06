"""Launch the editor with a Windows selector event loop for psycopg."""

import argparse
import asyncio
import errno
import os
from pathlib import Path
import socket
import sys

from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.getenv("UI_PORT", "8000")))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be in [1, 65535]")
    # Report a duplicate launch before importing and initializing the application.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as error:
            if error.errno != errno.EADDRINUSE and getattr(error, "winerror", None) != 10048:
                raise
            print(f"Порт {args.port} уже занят. Если редактор уже запущен, откройте:")
            print(f"http://127.0.0.1:{args.port}/")
            print("Для запуска на другом порту: .\\start-ui.cmd --port 8001")
            return
    import uvicorn
    from src.controller import EditorController, EditorSettings
    from .app import create_app
    settings = EditorSettings(root=root, max_examples=int(os.getenv("UI_MAX_EXAMPLES", "3")),
                              min_score=float(os.getenv("TEXT_CORRECTION_MIN_SCORE", "5")),
                              min_similarity=float(os.getenv("TEXT_CORRECTION_MIN_SIMILARITY", "0.6")))
    app = create_app(EditorController(settings))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info"))
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(server.serve())


if __name__ == "__main__":
    main()

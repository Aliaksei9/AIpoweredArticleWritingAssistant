"""Record a real proofreading request on a local test text, without API secrets."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
from src.text_correction import LanguageCheck, OpenRouterClient, OpenRouterConfig


async def run():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, required=True)
    args = parser.parse_args()
    text = args.text_file.read_text(encoding="utf-8-sig").rstrip("\r\n")
    config = OpenRouterConfig.from_env()
    folder = Path("model_tests/results") / ("language_diagnostic_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True)
    def save(name, value):
        encoded = json.dumps(value, ensure_ascii=False, indent=2)
        if config.api_key:
            encoded = encoded.replace(config.api_key, "[redacted]")
        (folder/name).write_text(encoded, encoding="utf-8")
    class RecordingClient(OpenRouterClient):
        calls = 0
        async def complete(self, messages, schema, *, max_tokens):
            self.calls += 1
            save(f"request_{self.calls}.json", dict(messages=messages, schema=schema))
            result = await super().complete(messages, schema, max_tokens=max_tokens)
            save(f"completion_{self.calls}.json", result)
            return result
    print(f"Diagnostic output: {folder}", flush=True)
    report = dict(model=config.model, characters=len(text))
    started = time.perf_counter()
    try:
        async with RecordingClient(config) as client:
            result = await LanguageCheck(client).check(text)
        save("result.json", result)
        report.update(status="succeeded", issues=len(result["issues"]), context=result["context"])
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    report["elapsed_seconds"] = time.perf_counter()-started
    save("report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report["status"] == "succeeded"


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        success = runner.run(run())
    sys.exit(0 if success else 2)

"""Record prepared prompts and redacted provider responses for one real input."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv

from src.database.connection import connect_database
from src.database.model_embedding import TextEmbedder
from src.database.table_classes import ArticleVectorTable
from src.text_correction import LLManalysis, OpenRouterClient, OpenRouterConfig, SearchSimillar, TextCorrectionModule


async def run():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--topic", default="Аниме")
    args = parser.parse_args()
    config = OpenRouterConfig.from_env()
    folder = Path("model_tests/results") / ("correction_diagnostic_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True)

    def encoded(value):
        content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        if config.api_key:
            content = content.replace(config.api_key, "[redacted]")
        return content

    def save(name, value):
        (folder/name).write_text(encoded(value), encoding="utf-8")

    class RecordingClient(OpenRouterClient):
        responses = 0
        completions = 0
        async def _request(self, method, path, *, payload=None):
            body = await super()._request(method, path, payload=payload)
            if path == "chat/completions":
                self.responses += 1
                save(f"provider_response_{self.responses}.json", body)
            return body

        async def complete(self, messages, schema, *, max_tokens):
            completion = await super().complete(messages, schema, max_tokens=max_tokens)
            self.completions += 1
            save(f"completion_{self.completions}.json", completion)
            return completion

    text = args.text_file.read_text(encoding="utf-8-sig").rstrip("\r\n")
    started = time.perf_counter()
    report = dict(status="running", model=config.model, characters=len(text), topic=args.topic)
    print(f"Diagnostic output: {folder}", flush=True)
    try:
        import torch
        torch.set_num_threads(4)
        async with RecordingClient(config) as client:
            embedder = TextEmbedder()
            async with await connect_database(os.environ["DATABASE_URL"]) as conn:
                module = TextCorrectionModule(SearchSimillar(ArticleVectorTable(conn), embedder),
                    LLManalysis(client, max_examples=int(os.getenv("UI_MAX_EXAMPLES", "3"))))
                found, plan = await module.prepare(text, args.topic)
            save("plan.json", dict(plan.to_dict(), retrieval=found.retrieval_metadata()))
            print(json.dumps(dict(sentences=plan.context["sentence_count"], batches=len(plan.batches),
                                  examples=len(plan.references), input_bound=plan.context["estimated_input_tokens"])), flush=True)
            result = await module.analysis.execute(plan)
            save("result.json", result)
            report.update(status="succeeded", sentences=len(result["sentences"]), context=result["context"])
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    report["elapsed_seconds"] = time.perf_counter()-started
    save("report.json", report)
    print(encoded(report), flush=True)
    return report["status"] == "succeeded"


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        success = runner.run(run())
    sys.exit(0 if success else 2)

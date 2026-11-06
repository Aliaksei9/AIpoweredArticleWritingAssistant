"""Compare live free OpenRouter models on identical retrieval inputs."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.database.connection import connect_database
from src.database.model_embedding import TextEmbedder
from src.database.table_classes import ArticleVectorTable
from src.text_correction import LLManalysis, OpenRouterClient, OpenRouterConfig, SearchSimillar
from src.text_correction.errors import TextCorrectionError
from src.text_correction.sentences import split_sentences

DEFAULT_MODELS = ["nvidia/nemotron-3.5-super-vl:free",
                  "dots-studio/dots-3-note-preview:free", "liquid/lfm-2.5-2.6b:free"]
CASES = [dict(name="technology", topic="Технологии",
    text="Новый ноутбук получил процессор с 8 ядрами и 16 ГБ памяти. Он достаточно быстрый для работы. "
         "Производитель обещает до 10 часов работы без подзарядки."),
    dict(name="games", topic="Игры",
    text="Студия представила обновление игры с 3 новыми картами. Игроки смогут попробовать их бесплатно. "
         "Разработчики также исправили ошибки сетевого режима.")]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


async def run():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = replace(OpenRouterConfig.from_env(), free_only=True, max_output_tokens=8192, retries=0)
    if not config.api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env; no paid model is used")
    output = args.output or root / "model_tests/results" / datetime.now(timezone.utc).strftime("text_correction_llms_%Y%m%d_%H%M%S_UTC")
    if (output / "report.json").exists():
        raise SystemExit("Choose a new output directory to preserve the existing report")
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from threadpoolctl import threadpool_limits
    previous_threads = torch.get_num_threads()
    retrieved = []
    try:
        torch.set_num_threads(4)
        with threadpool_limits(limits=4):
            embedder = TextEmbedder()
            async with await connect_database(os.environ["DATABASE_URL"]) as conn:
                search = SearchSimillar(ArticleVectorTable(conn), embedder)
                for case in CASES:
                    found = await search.search(case["text"], case["topic"])
                    retrieved.append(found)
                    print(f"Retrieved {len(found.articles)} popular examples for {case['name']}", flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    write_json(output / "inputs.json", [dict(case, retrieval=found.to_dict()) for case, found in zip(CASES, retrieved)])
    report = dict(status="running", created_at=datetime.now(timezone.utc).isoformat(),
        free_only=True, cases=len(CASES), results={},
        selection_rule="all cases valid JSON with full sentence coverage and preserved numeric anchors, then lowest median API time",
        limitations=["Two short fixture texts; not an expert assessment of style or popularity uplift",
                     "Free endpoint latency and availability vary", "Numeric checks do not prove full semantic fidelity"])
    for name in args.models:
        results = []
        model_config = replace(config, model=name)
        async with OpenRouterClient(model_config) as client:
            for case, found in zip(CASES, retrieved):
                started = time.perf_counter()
                print(f"Testing {name}: {case['name']}", flush=True)
                try:
                    info = await client.model_info()
                    analysis = LLManalysis(client, max_examples=3)
                    plan = await analysis.prepare(case["text"], found.articles, topic=case["topic"])
                    safe_name = name.replace("/", "__").replace(":", "_")
                    write_json(output / f"{safe_name}_{case['name']}_prompt.json", plan.to_dict())
                    api_started = time.perf_counter()
                    result = await analysis.execute(plan)
                    api_seconds = time.perf_counter()-api_started
                    path = output / f"{name.replace('/', '__')}_{case['name']}.json"
                    # ':' is illegal in Windows file names.
                    path = path.with_name(path.name.replace(":", "_"))
                    write_json(path, result)
                    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", case["text"]))
                    rewritten_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", result["rewritten_text"]))
                    row = dict(case=case["name"], valid=True, api_seconds=api_seconds,
                        total_seconds=time.perf_counter()-started, expected_sentences=len(split_sentences(case["text"])),
                        returned_sentences=len(result["sentences"]), examples_used=len(result["references"]),
                        numeric_anchors_preserved=numbers.issubset(rewritten_numbers),
                        no_new_numbers=rewritten_numbers.issubset(numbers), result_file=path.name,
                        usage=result["context"]["usage"], model_info=asdict(info))
                    print(f"OK: {name} / {case['name']}: {api_seconds:.2f}s", flush=True)
                except (TextCorrectionError, ValueError) as error:
                    row = dict(case=case["name"], valid=False, seconds=time.perf_counter()-started,
                               error=str(error), error_type=type(error).__name__)
                    print(f"FAILED: {name} / {case['name']}: {error}", flush=True)
                results.append(row)
                report["results"][name] = results
                write_json(output / "report.json", report)
    eligible = [name for name, rows in report["results"].items() if len(rows)==len(CASES)
                and all(r["valid"] and r["numeric_anchors_preserved"] and r["no_new_numbers"] for r in rows)]
    report["selected_model"] = min(eligible, key=lambda n: statistics.median(
        r["api_seconds"] for r in report["results"][n])) if eligible else None
    report.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "report.json", report)
    lines = ["# Бесплатные модели корректировки текста", "",
        "Два текста по три предложения; одинаковые примеры поиска для всех моделей. Проверяется полный JSON-контракт.", "",
        "| Модель | Успешно | Числа сохранены без добавлений | Медиана API, с |",
        "| --- | ---: | ---: | ---: |"]
    for name, rows in report["results"].items():
        successful = [r for r in rows if r["valid"]]
        anchors = sum(r["numeric_anchors_preserved"] and r["no_new_numbers"] for r in successful)
        latency = f"{statistics.median(r['api_seconds'] for r in successful):.2f}" if successful else "—"
        lines.append(f"| {name} | {len(successful)}/{len(rows)} | {anchors}/{len(rows)} | {latency} |")
    lines += ["", f"Выбрана по формальным проверкам и скорости: **{report['selected_model']}**.", "",
              "Это не экспертная оценка редакторского качества и не проверка роста популярности. "
              "Все запросы ограничены бесплатными моделями и нулевыми ценами провайдера."]
    (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Selected: {report['selected_model']}; results: {output}", flush=True)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run())

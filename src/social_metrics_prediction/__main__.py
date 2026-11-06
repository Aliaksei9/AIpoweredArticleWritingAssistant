"""Train the selected family or predict counts for a topic and resource."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from dotenv import load_dotenv
from threadpoolctl import threadpool_limits

from src.database.connection import connect_database
from src.database.model_embedding import MODEL_NAME
from src.database.table_classes import ResourceTable
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased, TextTopicEncoder

from .dataset import load_metric_dataset
from .predictor import SocialMetricsPredictor
from .resources import ResourceInfo
from .training import cross_fitted_scores, train_text_head


def progress(message):
    print(message, flush=True)


async def database_resources(dsn):
    async with await connect_database(dsn) as conn:
        rows = await ResourceTable(conn).list_prediction_resources()
    return [ResourceInfo.from_database(r) for r in rows if r["reader_count"] and r["reader_count"] > 0]


async def run():
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--output", type=Path, default=Path("artifacts/social_metrics/model.joblib"))
    train.add_argument("--text-model-file", type=Path, default=Path("artifacts/text_models/neural_network.joblib"))
    train.add_argument("--folds", type=int, default=3)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=4)
    predict = commands.add_parser("predict")
    predict.add_argument("--model-file", type=Path, default=Path("artifacts/social_metrics/model.joblib"))
    predict.add_argument("--text-model-file", type=Path, default=Path("artifacts/text_models/neural_network.joblib"))
    predict.add_argument("--topic", required=True)
    predict.add_argument("--resource", required=True, help="Resource UUID or exact database name")
    predict.add_argument("--source", default="TG")
    predict.add_argument("--offline-resources", action="store_true", help="Use audience snapshot from model artifact")
    score = predict.add_mutually_exclusive_group(required=True)
    score.add_argument("--score", type=float)
    score.add_argument("--text-file", type=Path)
    commands.add_parser("resources")
    args = parser.parse_args()
    dsn = os.getenv("DATABASE_URL")
    if args.command == "resources":
        if not dsn:
            parser.error("Set DATABASE_URL in .env")
        from dataclasses import asdict
        print(json.dumps([asdict(r) for r in await database_resources(dsn)], ensure_ascii=False, indent=2))
        return
    if args.command == "predict":
        model = SocialMetricsPredictor.load(args.model_file)
        if not args.offline_resources:
            if not dsn:
                parser.error("Set DATABASE_URL or use --offline-resources")
            model.set_resources(await database_resources(dsn))
        popularity_score = args.score
        if popularity_score is None:
            text_model = AssesmentConformPPtextBased.load(args.text_model_file)
            expected = model.metadata.get("text_model_sha256")
            if expected and hashlib.sha256(args.text_model_file.read_bytes()).hexdigest() != expected:
                parser.error("Text model differs from training bundle; retrain module 4 with this text model")
            with threadpool_limits(limits=4):
                previous_threads = torch.get_num_threads()
                try:
                    torch.set_num_threads(4)
                    popularity_score = text_model.assess(args.text_file.read_text(encoding="utf-8-sig"), args.topic)
                finally:
                    torch.set_num_threads(previous_threads)
        prediction = model.predict(args.topic, popularity_score, args.resource, args.source)
        print(json.dumps(dict(topic=args.topic, resource=args.resource, source=args.source,
                              popularity_score=popularity_score, **prediction.to_dict()), ensure_ascii=False))
        return
    if not dsn:
        parser.error("Set DATABASE_URL in .env")
    if args.folds < 2 or args.threads < 1:
        parser.error("At least two folds and positive threads required")
    if args.output.resolve() == args.text_model_file.resolve():
        parser.error("The two model artifact paths must differ")
    if os.getenv("EMBEDDING_MODEL", MODEL_NAME) != MODEL_NAME:
        parser.error("Cached DB vectors require the standard E5 model")
    dataset = await load_metric_dataset(dsn)
    progress(f"Training on {len(dataset.scores)} articles / {len(dataset.resources)} resources")
    encoder = TextTopicEncoder()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.threads)
        with threadpool_limits(limits=args.threads):
            features = encoder.transform(dataset.text.texts, dataset.text.topics, dataset.text.embeddings)
            scores, audit = cross_fitted_scores(features, dataset.scores, dataset.groups,
                folds=args.folds, seed=args.seed, threads=args.threads, on_progress=progress)
            inputs = dataset.inputs(np.arange(len(scores)), scores)
            model = SocialMetricsPredictor(args.seed, args.threads).fit(inputs, dataset.counts)
            model.set_resources(dataset.resources)
            progress("Training the final text neural network on all rows")
            text_model = train_text_head(encoder, features, dataset.scores, seed=args.seed, threads=args.threads)
            text_model.metadata.update(dataset=dataset.text.report,
                dataset_fingerprint=dataset.text.fingerprint, purpose="production; all labeled rows")
            text_model.save(args.text_model_file)
            model.metadata.update(dataset=dataset.text.report, dataset_fingerprint=dataset.fingerprint,
                crossfit_folds=args.folds, purpose="production; all labeled rows",
                text_model_sha256=hashlib.sha256(args.text_model_file.read_bytes()).hexdigest())
            model.save(args.output)
            manifest = dict(dataset_fingerprint=dataset.fingerprint, metadata=model.metadata,
                text_model=str(args.text_model_file.resolve()), social_model=str(args.output.resolve()),
                social_model_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                article_ids=dataset.text.article_ids, crossfit=audit, out_of_fold_scores=scores.tolist())
            args.output.with_suffix(".training.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            progress(f"Saved {args.output} and {args.text_model_file}")
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run())

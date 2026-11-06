"""Train models from PostgreSQL or predict for a local text file."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from threadpoolctl import threadpool_limits

from src.database.connection import initialize_database
from src.database.model_embedding import MODEL_NAME

from .assessment import AssesmentConformPPtextBased
from .dataset import load_training_dataset
from .features import TextTopicEncoder


async def run():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    parser = argparse.ArgumentParser(description="Train or predict popularity from text and topic")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--model", choices=["neural_network"], default="neural_network")
    train.add_argument("--output", type=Path, default=Path("artifacts/text_models"))
    train.add_argument("--refresh-scores", action="store_true")
    train.add_argument("--reembed", action="store_true", help="Recompute text embeddings locally")
    train.add_argument("--seed", type=int, default=42)
    predict = commands.add_parser("predict")
    predict.add_argument("--model-file", type=Path, required=True)
    predict.add_argument("--text-file", type=Path, required=True)
    predict.add_argument("--topic", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.command == "predict":
            model = AssesmentConformPPtextBased.load(args.model_file)
            text = args.text_file.read_text(encoding="utf-8-sig")
            print(json.dumps(dict(model=model.model.name, topic=args.topic,
                                  score=model.assess(text, args.topic)), ensure_ascii=False))
            return
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise SystemExit("Set DATABASE_URL in .env")
        if args.refresh_scores:
            print(f"Reassessed {await initialize_database(dsn)} articles", flush=True)
        dataset = await load_training_dataset(dsn)
        print(json.dumps(dataset.report, ensure_ascii=False), flush=True)
        embedding_model = os.getenv("EMBEDDING_MODEL", MODEL_NAME)
        if not args.reembed and embedding_model != MODEL_NAME:
            raise SystemExit("Use --reembed for a custom EMBEDDING_MODEL; DB vectors use default E5")
        encoder = TextTopicEncoder(embedding_model)
        features = encoder.transform(dataset.texts, dataset.topics,
                                     None if args.reembed else dataset.embeddings)
        names = [args.model]
        for name in names:
            print(f"Training {name} on {len(dataset.scores)} articles", flush=True)
            model = AssesmentConformPPtextBased(name, encoder, args.seed)
            model.train_features(features, dataset.scores)
            model.metadata.update(dataset_fingerprint=dataset.fingerprint, dataset=dataset.report,
                                  purpose="production; fitted on all labeled rows")
            destination = args.output / f"{name}.joblib"
            model.save(destination)
            print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run())

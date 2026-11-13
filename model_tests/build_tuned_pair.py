"""Build core-only artifacts from the validated frozen configurations.

Held-out fresh articles stay excluded. This does not overwrite active files;
copy the verified pair from the requested output after reviewing the audit.
"""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import joblib
import numpy as np

from src.social_metrics_prediction import SocialMetricsPredictor,ResourceInfo
from src.social_metrics_prediction.dataset import load_metric_dataset
from .compare_text_models import write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def main():
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.directory/'fresh_confirmation.json').read_text(encoding='utf8'))
    latency=json.loads((args.directory/'fresh_evaluation_latency.json').read_text(encoding='utf8'))
    assert report['improves'] and latency['status']=='passed'
    assert report['configuration']==dict(family='catboost',epochs=150,alpha=20.,blend=.25)
    inputs=json.loads((args.directory/'fresh_training_inputs.json').read_text(encoding='utf8'))
    snapshot=json.loads((args.directory/'fresh_snapshot.json').read_text(encoding='utf8'))
    split=json.loads((args.directory/'fresh_split_frozen.json').read_text(encoding='utf8'))
    text=joblib.load(args.directory/'confirmation/text_evaluation.joblib')
    text.metadata.update(model_parameters=text.model.estimator.get_params(),
        purpose='frozen evaluation winner; old held-out 2039 and all fresh texts excluded',
        dataset=dict(database_rows=10182,training_rows=8143,heldout_rows=2039,resource_count=91),
        evaluation='docs/experiments/2026-11-11-tuning/README.md')
    assert text.metadata['training_rows']==8143
    model=SocialMetricsPredictor(seed=42,threads=4).fit(inputs['rows'],np.asarray(inputs['counts']))
    with np.load(args.directory/'fresh_test_predictions.npz') as predictions:
        np.testing.assert_array_equal(model.predict_rows(inputs['validation']),predictions['model'])
    dataset=await load_metric_dataset(os.environ['DATABASE_URL'])
    resources={r.id:r for r in dataset.resources}
    trained={r['resource_id'] for r in inputs['rows']}
    for row in snapshot['articles']:
        if row['resource_id'] in trained:
            resources[row['resource_id']]=ResourceInfo(row['resource_id'],row['resource_name'],int(row['audience']),row['source'])
    model.set_resources(list(resources.values()))
    assert len(resources)==91+report['fresh_channels']
    training_fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()
    model.metadata.update(purpose='validated CatBoost correction; fresh held-out 282 articles excluded',
        score_input='nested OOF scores for old training rows; held-out head scores for old test and fresh articles',
        dataset_fingerprint=training_fingerprint,
        dataset=dict(old_rows=10182,fresh_training_rows=report['fresh_training_rows'],
                     training_rows=len(inputs['rows']),resource_count=len(resources),fresh_heldout_rows=report['fresh_test_rows']),
        evaluation='docs/experiments/2026-11-11-tuning/README.md')
    args.output.mkdir(parents=True,exist_ok=True)
    text_path=args.output/'neural_network.joblib';social_path=args.output/'model.joblib'
    text.save(text_path)
    model.metadata['text_model_sha256']=sha256(text_path)
    model.save(social_path)
    restored=SocialMetricsPredictor.load(social_path)
    np.testing.assert_array_equal(restored.predict_rows(inputs['validation']),model.predict_rows(inputs['validation']))
    manifest=dict(status='passed',built_at=datetime.now(timezone.utc).isoformat(),
        text_model_sha256=sha256(text_path),social_model_sha256=sha256(social_path),
        text_model_rows=8143,social_model_rows=len(inputs['rows']),resources=len(resources),
        training_fingerprint=training_fingerprint,fresh_heldout_excluded=True,
        production_predictions_match_confirmed_predictions=True,
        model_metadata=model.metadata,text_metadata=text.metadata,
        resource_catalog=[asdict(r) for r in resources.values()])
    write_json(args.output/'model.training.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','text_model_rows','social_model_rows','resources','production_predictions_match_confirmed_predictions']}),flush=True)


if __name__=='__main__':
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform=='win32' else None) as runner:
        runner.run(main())

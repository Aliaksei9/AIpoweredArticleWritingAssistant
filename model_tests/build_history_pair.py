"""Build the validated history predictor without changing the text model."""
import argparse
import asyncio
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import numpy as np

from src.social_metrics_prediction import SocialMetricsPredictor,ResourceInfo
from src.social_metrics_prediction.selected_model import MODEL_CONFIGURATION
from .compare_text_models import write_json


async def main():
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads((args.directory/'development.json').read_text(encoding='utf8'))
    report=json.loads((args.directory/'confirmation.json').read_text(encoding='utf8'))
    protocol=json.loads((args.directory/'control_protocol.json').read_text(encoding='utf8'))
    control=json.loads((args.directory/'control_snapshot.json').read_text(encoding='utf8'))
    assert report['improvement_percent']>0 and report['improvement_vs_previous_profile_percent']>0
    assert MODEL_CONFIGURATION==report['configuration']
    text_path=root/'artifacts/text_models/neural_network.joblib'
    text_hash=hashlib.sha256(text_path.read_bytes()).hexdigest()
    assert text_hash==protocol['text_model_sha256']
    model=SocialMetricsPredictor().fit(data['rows'],np.asarray(data['counts']))
    assert all(m.channels_without_dates_==0 for m in model.estimators_)
    rows=[{key:r[key] for key in ['source','resource_id','topic','score','audience']} for r in control['articles']]
    with np.load(args.directory/'control_predictions.npz') as saved:
        np.testing.assert_array_equal(model.predict_rows(rows),saved['candidate'])
    previous=SocialMetricsPredictor.load(root/'artifacts/model_history_backups/20261111/model.joblib')
    canonical={r['resource_id'] for r in protocol['channels']}
    resources={r.id:r for r in previous.resources if r.id in canonical}
    old_snapshot=json.loads((root/'model_tests/results/tuning_20261111/fresh_snapshot.json').read_text(encoding='utf8'))
    mapping={r['username'].lower():r for r in protocol['channels']}
    for row in old_snapshot['articles']:
        resource_id=mapping[row['username'].lower()]['resource_id']
        resources[resource_id]=ResourceInfo(resource_id,row['resource_name'],row['audience'],'TG')
    model.set_resources(list(resources.values()))
    assert len(model.resources)==146
    digest=hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()
    model.metadata.update(purpose='frozen chronological winner; new control publications excluded',
        dataset_fingerprint=digest,text_model_sha256=text_hash,
        dataset=dict(training_rows=len(data['rows']),resource_count=len(resources),
                     old_rows=data['old_rows'],additional_rows=data['fresh_rows'],heldout_rows=len(rows)),
        score_input='unchanged text/topic neural network; nested OOF old training inputs; old held-out and additional texts predicted by frozen head',
        evaluation='docs/experiments/2026-11-11-history/README.md')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'model.joblib'
    model.save(path)
    restored=SocialMetricsPredictor.load(path)
    np.testing.assert_array_equal(restored.predict_rows(rows),model.predict_rows(rows))
    manifest=dict(status='passed',built_at=datetime.now(timezone.utc).isoformat(),
        text_model_sha256=text_hash,social_model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        training_rows=len(data['rows']),resources=len(model.resources),control_rows_excluded=len(rows),
        publication_dates_known=model.metadata['publication_dates_known'],
        production_predictions_match_confirmed_predictions=True,metadata=model.metadata)
    write_json(args.output/'model.training.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='metadata'}),flush=True)


if __name__=='__main__':
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform=='win32' else None) as runner:
        runner.run(main())

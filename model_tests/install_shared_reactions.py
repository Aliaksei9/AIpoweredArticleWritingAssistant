"""Install a fully validated reaction-only change, keeping the other models intact."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
from src.social_metrics_prediction import SocialMetricsPredictor
from src.social_metrics_prediction.unified_features import INPUT_FIELDS
from .compare_text_models import write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root=Path(__file__).resolve().parents[1]
    report=json.loads(args.report.read_text(encoding='utf8'))
    latency=json.loads(args.latency.read_text(encoding='utf8'))
    training=json.loads(args.training.read_text(encoding='utf8'))
    active=root/'artifacts/social_metrics/model.joblib'
    manifest_path=root/'artifacts/model_bundle.json'
    text=root/'artifacts/text_models/neural_network.joblib'
    assert report['every_row_tested_once'] and report['one_row_per_article']
    assert report['gains_percent']['reactions']>=1.
    assert latency['within_ten_seconds'] and sha(args.social)==latency['social_sha256']
    assert sha(text)==latency['text_sha256']==training['text_sha256']
    assert sha(active)==training['previous_social_sha256']
    previous=SocialMetricsPredictor.load(active);candidate=SocialMetricsPredictor.load(args.social)
    assert report['rows']==candidate.metadata['training_rows']
    assert tuple(candidate.shared_input_fields)==INPUT_FIELDS
    assert candidate.metadata['text_model_sha256']==sha(text)
    assert len({id(head.model) for head in candidate.estimators_})==3
    assert joblib.hash(candidate.shared_encoder_)==joblib.hash(previous.shared_encoder_)
    for i,name in [(0,'views'),(2,'comments')]:
        assert report['candidate'][name]==report['previous_models_same_rows'][name]
        assert joblib.hash(candidate.estimators_[i])==joblib.hash(previous.estimators_[i])
    assert args.social.stat().st_size<100*1024**2
    before=json.loads(manifest_path.read_text(encoding='utf8'))
    assert before['social_model']['sha256']==sha(active)
    backup=root/'artifacts/model_history_backups/20261112_reactions_percent'
    backup.mkdir(parents=True,exist_ok=True)
    for source,name in [(active,'social.joblib'),(manifest_path,'model_bundle.json')]:
        target=backup/name
        if target.exists():assert sha(target)==sha(source)
        else:shutil.copy2(source,target)
    manifest=dict(before)
    manifest.update(packaged_on=datetime.now(timezone.utc).date().isoformat(),
        purpose='independent reaction refinement; complete shared inputs; one current article row',
        evaluation='docs/experiments/2026-11-12-reactions/README.md')
    manifest['social_model']=dict(before['social_model'],sha256=sha(args.social),
        bytes=args.social.stat().st_size,configuration=candidate.metadata['selection']['configuration'])
    manifest['gains_to_channel_mean_mse_percent']=report['gains_percent']
    try:
        temporary=active.with_suffix('.installing');shutil.copy2(args.social,temporary);temporary.replace(active)
        write_json(manifest_path,manifest)
    except BaseException:
        shutil.copy2(backup/'social.joblib',active)
        shutil.copy2(backup/'model_bundle.json',manifest_path)
        raise
    print(json.dumps(dict(status='installed',reaction_gain_percent=report['gains_percent']['reactions'],
        unchanged=['text','views','comments'],social_sha256=sha(active))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('report','latency','training','social'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())

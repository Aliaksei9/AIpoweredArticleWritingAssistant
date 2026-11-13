"""Install only a complete, tested shared-input pair with rollback copies."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from src.social_metrics_prediction import SocialMetricsPredictor
from src.social_metrics_prediction.unified_features import INPUT_FIELDS
from .compare_text_models import write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root=Path(__file__).resolve().parents[1]
    report=json.loads(args.report.read_text(encoding='utf8'));latency=json.loads(args.latency.read_text(encoding='utf8'))
    assert report['one_row_per_article'] and report['every_row_tested_once'] and latency['within_ten_seconds']
    assert sha(args.text)==latency['text_sha256'] and sha(args.social)==latency['social_sha256']
    text=AssesmentConformPPtextBased.load(args.text);social=SocialMetricsPredictor.load(args.social)
    assert social.feature_mode=='unified' and tuple(social.shared_input_fields)==INPUT_FIELDS
    assert social.metadata['text_model_sha256']==sha(args.text)
    assert all(head.model.features.shape[1]==social.shared_encoder_.n_features_in_ if hasattr(head.model,'features') else head.model.model.coef_.shape[1]==social.shared_encoder_.n_features_in_ for head in social.estimators_)
    paths=[root/'artifacts/text_models/neural_network.joblib',root/'artifacts/social_metrics/model.joblib']
    before=json.loads((root/'artifacts/model_bundle.json').read_text(encoding='utf8'))
    assert sha(paths[0])==before['text_model']['sha256'] and sha(paths[1])==before['social_model']['sha256']
    backup=root/'artifacts/model_history_backups/20261112_single_snapshot';backup.mkdir(parents=True,exist_ok=True)
    for source,name in [(paths[0],'text.joblib'),(paths[1],'social.joblib'),(root/'artifacts/model_bundle.json','model_bundle.json')]:
        target=backup/name
        if target.exists():assert sha(source)==sha(target)
        else:shutil.copy2(source,target)
    manifest=dict(packaged_on=datetime.now(timezone.utc).date().isoformat(),purpose='same complete input schema for all metrics; no repeated measurements',
        target_formula_version=text.metadata['target_formula_version'],
        training=dict(text_rows=text.metadata['training_rows'],social_rows=social.metadata['training_rows'],one_row_per_article=True,
            resources=len(social.resources),sources=sorted({r.source for r in social.resources}),text_inputs=['text','topic'],social_inputs=list(INPUT_FIELDS)),
        text_model=dict(path='artifacts/text_models/neural_network.joblib',sha256=sha(args.text),bytes=args.text.stat().st_size,family='neural_network',strategy='additive_ensemble'),
        social_model=dict(path='artifacts/social_metrics/model.joblib',sha256=sha(args.social),bytes=args.social.stat().st_size,
            family='independent_models_shared_inputs',configuration=social.metadata['selection']['configuration'],feature_dimensions=social.shared_encoder_.n_features_in_),
        evaluation='docs/experiments/2026-11-12-unified-inputs/README.md',validation_environment=before.get('validation_environment'))
    try:
        for source,target in zip([args.text,args.social],paths):
            temp=target.with_suffix('.installing');shutil.copy2(source,temp);temp.replace(target)
        write_json(root/'artifacts/model_bundle.json',manifest)
    except BaseException:
        for source,target in [(backup/'text.joblib',paths[0]),(backup/'social.joblib',paths[1]),(backup/'model_bundle.json',root/'artifacts/model_bundle.json')]:shutil.copy2(source,target)
        raise
    print(json.dumps(dict(status='installed',shared_inputs=list(INPUT_FIELDS),rows=social.metadata['training_rows'],backup=str(backup.relative_to(root)))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report','latency','text','social'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())

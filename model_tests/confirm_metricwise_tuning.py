"""Evaluate per-count settings frozen from development CV before test access."""
import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np

from .compare_social_metrics import evaluate,normalized_mse
from .compare_text_models import write_json
from .tuning_candidates import MetricWiseResidualPredictor,SOCIAL_CONFIGS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=Path('model_tests/results/expansion_20261111'))
    args=parser.parse_args()
    directory=args.directory/'confirmation'
    report=json.loads((directory/'confirmation.json').read_text(encoding='utf8'))
    frozen=json.loads((args.directory/'metricwise_choices_frozen.json').read_text(encoding='utf8'))
    screening=json.loads((args.directory/'screening.json').read_text(encoding='utf8'))
    inputs=json.loads((directory/'nested_inputs.json').read_text(encoding='utf8'))
    splits=json.loads((args.previous/'social_metrics/splits.json').read_text(encoding='utf8'))
    with np.load(args.previous/'social_metrics/dataset.npz') as saved:counts=saved['counts']
    configurations=[]
    for entry in frozen['metrics']:
        name=entry['configuration']
        if name=='resource_mean':configurations.append(None)
        else:
            config=screening['social'][name]
            configurations.append(dict(SOCIAL_CONFIGS[config['config']],blend=config['blend']))
    factory=lambda:MetricWiseResidualPredictor(configurations)
    errors=[];per_metric=[]
    for fold,rows in zip(splits['cv'],inputs['cv']):
        a,b=np.asarray(fold['train_indices']),np.asarray(fold['validation_indices'])
        model=factory().fit(rows['train'],counts[a])
        predicted=model.predict_rows(rows['validation'])
        errors.append(normalized_mse(counts[b],predicted,counts[a]))
        per_metric.append(evaluate(counts[b],predicted))
    result=dict(cv_mse=float(np.mean(errors)),cv_folds=errors,per_metric=per_metric,configurations=configurations)
    assert 'metricwise_cv' not in report['results']
    report['results']['metricwise_cv']=result
    chosen=min(report['results'],key=lambda n:report['results'][n]['cv_mse'])
    write_json(directory/'extended_choices_frozen.json',dict(social_selected=chosen,metricwise=frozen,
        cv={n:v['cv_mse'] for n,v in report['results'].items()}))
    a,b=np.asarray(splits['train_indices']),np.asarray(splits['test_indices'])
    model=factory().fit(inputs['heldout']['train'],counts[a])
    predicted=model.predict_rows(inputs['heldout']['test'])
    result.update(test=evaluate(counts[b],predicted),test_normalized_mse=normalized_mse(counts[b],predicted,counts[a]))
    if chosen=='metricwise_cv':joblib.dump(model,directory/'social_evaluation.joblib',compress=3)
    with np.load(directory/'test_predictions.npz') as saved:predictions={k:saved[k] for k in saved.files}
    predictions['metricwise_cv']=predicted
    np.savez_compressed(directory/'test_predictions.npz',**predictions)
    shutil.copy2(directory/'confirmation.json',directory/'confirmation_before_metricwise.json')
    report['social_selected']=chosen
    report['metricwise_choices']=frozen
    report['social_improves_cv']=report['results'][chosen]['cv_mse']<report['results']['resource_mean']['cv_mse']
    report['social_improves_test']=report['results'][chosen]['test_normalized_mse']<report['results']['resource_mean']['test_normalized_mse']
    write_json(directory/'confirmation.json',report)
    print(json.dumps(dict(social_selected=chosen,metricwise=result),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

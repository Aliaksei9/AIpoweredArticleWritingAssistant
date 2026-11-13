"""Tune on development folds only; release held-out labels after freezing choices.

Stage one reuses the previous leak-free upstream scores for social screening.
Confirmation later rebuilds upstream scores using nested text configuration
selection. The old test is a historical holdout, not a new independent sample.
"""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from .benchmark_utils import metrics
from .compare_social_metrics import normalized_mse
from .compare_text_models import write_json
from .social_metric_candidates import ResourceMeanBaseline
from .tuning_candidates import TEXT_CONFIGS,SOCIAL_CONFIGS,ResidualRatePredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=Path('model_tests/results/expansion_20261111'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'screening.json').exists():
        raise ValueError('Use a new output folder')
    with np.load(args.previous/'social_metrics/dataset.npz') as saved:
        features=saved['text_features']; labels=saved['metric_scores']; groups=saved['groups']; counts=saved['counts']
    splits=json.loads((args.previous/'social_metrics/splits.json').read_text(encoding='utf8'))
    inputs=json.loads((args.previous/'social_metrics/inputs.json').read_text(encoding='utf8'))
    train=np.asarray(splits['train_indices'])
    test=set(splits['test_indices'])
    for fold in splits['cv']:
        assert not test.intersection(fold['train_indices']+fold['validation_indices'])
    report=dict(status='running',protocol='development CV only; historical test not accessed',
        previous=str(args.previous),text={},social={},seed=42,threads=4,
        text_configurations=TEXT_CONFIGS,social_configurations=SOCIAL_CONFIGS,blend_grid=[.1,.25,.5,1.])
    write_json(args.output/'screening.json',report)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for name,parameters in TEXT_CONFIGS.items():
            errors=[]; baseline_errors=[]
            started=time.perf_counter()
            for i,fold in enumerate(splits['cv']):
                fitting=np.asarray(fold['train_indices']); validation=np.asarray(fold['validation_indices'])
                model=AssesmentConformPPtextBased(seed=42,num_threads=4,**parameters)
                model.train_features(features[fitting],labels[fitting])
                errors.append(metrics(labels[validation],model.model.predict(features[validation]))['mse'])
                keys=[(r['source'],r['resource_id']) for r in inputs['cv'][i]['train']]
                means={k:labels[fitting][[j for j,key in enumerate(keys) if k==key]].mean() for k in set(keys)}
                baseline=np.asarray([means.get((r['source'],r['resource_id']),labels[fitting].mean()) for r in inputs['cv'][i]['validation']])
                baseline_errors.append(metrics(labels[validation],baseline)['mse'])
            report['text'][name]=dict(cv_mse=float(np.mean(errors)),folds=errors,channel_mean_cv_mse=float(np.mean(baseline_errors)),fit_seconds=time.perf_counter()-started)
            write_json(args.output/'screening.json',report)
            print('TEXT',name,json.dumps(report['text'][name]),flush=True)
        for name,parameters in SOCIAL_CONFIGS.items():
            errors={str(b):[] for b in report['blend_grid']}
            raw_metrics={str(b):[] for b in report['blend_grid']}
            baseline_errors=[]
            started=time.perf_counter()
            for i,fold in enumerate(splits['cv']):
                fitting=np.asarray(fold['train_indices']); validation=np.asarray(fold['validation_indices'])
                rows=inputs['cv'][i]
                model=ResidualRatePredictor(seed=42,threads=4,**parameters).fit(rows['train'],counts[fitting])
                baseline=ResourceMeanBaseline().fit(rows['train'],counts[fitting]).predict_rows(rows['validation'])
                baseline_errors.append(normalized_mse(counts[validation],baseline,counts[fitting]))
                for blend in report['blend_grid']:
                    model.blend=blend
                    predicted=model.predict_rows(rows['validation'])
                    errors[str(blend)].append(normalized_mse(counts[validation],predicted,counts[fitting]))
                    raw_metrics[str(blend)].append(np.mean((counts[validation]-predicted)**2,axis=0).tolist())
            for blend in report['blend_grid']:
                key=f'{name}_blend_{blend:g}'
                report['social'][key]=dict(config=name,blend=blend,cv_mse=float(np.mean(errors[str(blend)])),
                    folds=errors[str(blend)],raw_mse_folds=raw_metrics[str(blend)],baseline_cv_mse=float(np.mean(baseline_errors)),
                    fit_seconds=time.perf_counter()-started)
            write_json(args.output/'screening.json',report)
            best=min((v for v in report['social'].values() if v['config']==name),key=lambda v:v['cv_mse'])
            print('SOCIAL',name,json.dumps(best),flush=True)
    report.update(status='complete',text_selected=min(report['text'],key=lambda n:report['text'][n]['cv_mse']),
        social_selected=min(report['social'],key=lambda n:report['social'][n]['cv_mse']))
    write_json(args.output/'screening.json',report)
    previous_report=json.loads((args.previous/'social_metrics/report.json').read_text(encoding='utf8'))
    scales=np.asarray([np.maximum(np.var(counts[f['train_indices']],axis=0),1.) for f in splits['cv']])
    metricwise=[]
    for metric_index,metric in enumerate(['views','reactions','comments']):
        baseline=np.asarray(previous_report['results']['resource_mean']['cv_metrics'][metric]['mse']['samples'])
        options=[('resource_mean',float(np.mean(baseline/scales[:,metric_index])))]
        for name,value in report['social'].items():
            error=np.asarray(value['raw_mse_folds'])[:,metric_index]
            options.append((name,float(np.mean(error/scales[:,metric_index]))))
        chosen=min(options,key=lambda pair:pair[1])
        metricwise.append(dict(metric=metric,configuration=chosen[0],cv_normalized_mse=chosen[1],
                               baseline_cv_normalized_mse=options[0][1]))
    write_json(args.output/'metricwise_choices_frozen.json',dict(
        protocol='per-metric CV choice frozen before any new held-out comparison',metrics=metricwise))
    print(json.dumps({k:report[k] for k in ['status','text_selected','social_selected']}),flush=True)


if __name__=='__main__':
    main()

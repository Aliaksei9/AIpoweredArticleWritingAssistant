"""Temporal development selection before examining a reserved publication window."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .compare_social_metrics import evaluate,normalized_mse
from .compare_text_models import write_json
from .history_candidates import ConditionalHistoryPredictor,configurations
from .tuning_candidates import ResidualRatePredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    output=args.directory
    if (output/'choices_frozen.json').exists():raise ValueError('Study already frozen')
    data=json.loads((output/'development.json').read_text(encoding='utf8'))
    rows=data['rows'];counts=np.asarray(data['counts'])
    configs=configurations()
    report=dict(status='running',configurations=configs,results={},text_model='unchanged text/topic neural network')
    prepared=[]
    for fold in data['folds']:
        a,b=np.asarray(fold['train']),np.asarray(fold['validation'])
        train=[rows[i] for i in a];validation=[rows[i] for i in b]
        fresh_train=a[a>=data['old_rows']]
        scale=counts[fresh_train]
        baseline=ConditionalHistoryPredictor().fit(train,counts[a]).predict_rows(validation)
        prepared.append(dict(train=train,validation=validation,fit=a,test=b,scale=scale,
            baseline_score=normalized_mse(counts[b],baseline,scale),baseline=baseline))
    for name in [*configs,'current_catboost']:
        scores=[];ratios=[];per_metric=[]
        for fold in prepared:
            model=(ResidualRatePredictor(family='catboost',epochs=150,alpha=20,blend=.25)
                   if name=='current_catboost' else ConditionalHistoryPredictor(**configs[name]))
            model.fit(fold['train'],counts[fold['fit']])
            predicted=model.predict_rows(fold['validation'])
            score=normalized_mse(counts[fold['test']],predicted,fold['scale'])
            scores.append(score);ratios.append(score/fold['baseline_score'])
            per_metric.append(evaluate(counts[fold['test']],predicted))
        report['results'][name]=dict(cv_normalized_mse=scores,cv_ratios=ratios,
            mean_ratio=float(np.mean(ratios)),worst_ratio=float(np.max(ratios)),metrics=per_metric)
        write_json(output/'screening.json',report)
        if len(report['results'])%10==0 or name=='current_catboost':
            best=min(report['results'],key=lambda n:report['results'][n]['mean_ratio'])
            print(json.dumps(dict(tried=len(report['results']),best=best,ratio=report['results'][best]['mean_ratio'])),flush=True)
    selected=min(report['results'],key=lambda n:report['results'][n]['mean_ratio'])
    report.update(status='complete',selected=selected)
    write_json(output/'screening.json',report)
    write_json(output/'choices_frozen.json',dict(selected=selected,
        configuration=configs.get(selected),cv=report['results'][selected],
        protocol='lowest mean ratio to channel mean over three forward chronological development folds; new holdout not accessed'))
    model=(ResidualRatePredictor(family='catboost',epochs=150,alpha=20,blend=.25)
           if selected=='current_catboost' else ConditionalHistoryPredictor(**configs[selected]))
    model.fit(rows,counts)
    joblib.dump(model,output/'selected_evaluation.joblib',compress=3)
    print(json.dumps(dict(selected=selected,configuration=configs.get(selected),cv=report['results'][selected]['cv_ratios'])),flush=True)


if __name__=='__main__':main()

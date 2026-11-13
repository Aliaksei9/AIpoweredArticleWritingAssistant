"""Nested whole-corpus evaluation with three independently selected models."""
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
from src.social_metrics_prediction.training import cross_fitted_scores
from src.database.embedding_config import EMBEDDING_SIGNATURE
from .compare_social_metrics import evaluate
from .compare_text_models import write_json
from .history_candidates import ConditionalHistoryPredictor
from .independent_metric_candidates import candidate_grid,MetricModel,ViewsModel,ReactionsModel,CommentsModel


TEXT_CONFIGS={
    'current':dict(hidden_layers=(256,128),dropout=.5,epochs=60,learning_rate=.0005,weight_decay=.1),
    'compact':dict(hidden_layers=(128,),dropout=.5,epochs=25,learning_rate=.0005,weight_decay=.1),
    'wide_short':dict(hidden_layers=(256,64),dropout=.5,epochs=30,learning_rate=.001,weight_decay=.1),
}
METRICS=['views','reactions','comments']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();directory=args.directory
    if (directory/'nested_report.json').exists():raise ValueError('Evaluation already completed')
    audit=json.loads((directory/'audit.json').read_text(encoding='utf8'))
    assert EMBEDDING_SIGNATURE==audit['fixed_embedding_model'], 'Embedding profile differs from frozen experiment corpus'
    data=np.load(directory/'corpus.npz');X=data['features'];labels=data['labels'];counts=data['counts'];groups=data['groups']
    rows=json.loads((directory/'rows.json').read_text(encoding='utf8'))
    grid=candidate_grid();classes=[ViewsModel,ReactionsModel,CommentsModel]
    predictions=np.full_like(counts,np.nan);baseline=np.full_like(counts,np.nan)
    text_predictions=np.full(len(labels),np.nan);hits=np.zeros(len(labels),dtype=int)
    report=dict(status='running',rows=len(rows),folds=[],target_reduction_percent=5.,text_configurations=TEXT_CONFIGS,
        metric_configuration_count=len(grid),protocol='five outer exact-text-group folds; text and per-metric tuning inside training data only',
        production_weights_modified=False)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for fold,(train,test) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=20261111).split(X,groups=groups),1):
            print(f'OUTER {fold}/5: {len(train)} train / {len(test)} test',flush=True)
            assert not set(groups[train])&set(groups[test])
            path=directory/f'outer_{fold}';path.mkdir(exist_ok=True)
            if (path/'prepared.joblib').exists():prepared=joblib.load(path/'prepared.joblib')
            else:
                errors={}
                inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=42).split(train,groups=groups[train]))
                for name,config in TEXT_CONFIGS.items():
                    values=[]
                    for a,b in inner:
                        head=AssesmentConformPPtextBased(seed=42,num_threads=4,**config)
                        head.train_features(X[train[a]],labels[train[a]])
                        values.append(float(np.mean((head.model.predict(X[train[b]])-labels[train[b]])**2)))
                    errors[name]=float(np.mean(values))
                    print(f'  text {name}: inner MSE={errors[name]:.4f}',flush=True)
                chosen=min(errors,key=errors.get);config=TEXT_CONFIGS[chosen]
                scores,audit=cross_fitted_scores(X[train],labels[train],groups[train],folds=3,
                    factory=lambda:AssesmentConformPPtextBased(seed=42,num_threads=4,**config))
                head=AssesmentConformPPtextBased(seed=42,num_threads=4,**config).train_features(X[train],labels[train])
                test_scores=head.model.predict(X[test])
                fit_rows=[dict(rows[i],score=float(s)) for i,s in zip(train,scores)]
                test_rows=[dict(rows[i],score=float(s)) for i,s in zip(test,test_scores)]
                prepared=dict(train=train,test=test,fit_rows=fit_rows,test_rows=test_rows,
                    text_selected=chosen,text_errors=errors,test_scores=test_scores,head=head,audit=audit)
                joblib.dump(prepared,path/'prepared.joblib',compress=3)
            assert np.array_equal(train,prepared['train']) and np.array_equal(test,prepared['test'])
            fit_rows=prepared['fit_rows'];test_rows=prepared['test_rows']
            text_predictions[test]=prepared['test_scores']
            validation_folds=list(GroupKFold(n_splits=2,shuffle=True,random_state=47).split(train,groups=groups[train]))
            scores={name:np.zeros(3) for name in grid}
            for name,config in grid.items():
                for a,b in validation_folds:
                    fitting=[fit_rows[i] for i in a];validation=[fit_rows[i] for i in b]
                    if 'family' not in config:
                        model=ConditionalHistoryPredictor(**config).fit(fitting,counts[train[a]])
                        predicted=model.predict_rows(validation)
                    else:
                        predicted=np.column_stack([MetricModel(i,config).fit(fitting,counts[train[a]]).predict(validation) for i in range(3)])
                    scores[name]+=np.mean((counts[train[b]]-predicted)**2,axis=0)/2
                if list(grid).index(name)%25==0:print(f'  social configurations {list(grid).index(name)+1}/{len(grid)}',flush=True)
            chosen=[min(scores,key=lambda name:scores[name][i]) for i in range(3)]
            base=ConditionalHistoryPredictor().fit(fit_rows,counts[train])
            baseline[test]=base.predict_rows(test_rows)
            for i,cls in enumerate(classes):
                model=cls(grid[chosen[i]]).fit(fit_rows,counts[train])
                predictions[test,i]=model.predict(test_rows)
                joblib.dump(model,path/(METRICS[i]+'.joblib'),compress=3)
            hits[test]+=1
            entry=dict(fold=fold,train_rows=len(train),test_rows=len(test),text_selected=prepared['text_selected'],
                text_mse=float(np.mean((text_predictions[test]-labels[test])**2)),
                selected=dict(zip(METRICS,chosen)),baseline=evaluate(counts[test],baseline[test]),candidate=evaluate(counts[test],predictions[test]))
            report['folds'].append(entry)
            write_json(path/'selection.json',dict(chosen=entry['selected'],inner_mse={n:v.tolist() for n,v in scores.items()}))
            write_json(directory/'progress.json',report)
            print('OUTER COMPLETE '+json.dumps(entry['selected']),flush=True)
    assert np.all(hits==1) and np.isfinite(predictions).all()
    results=dict(baseline=evaluate(counts,baseline),candidate=evaluate(counts,predictions))
    gains={m:(1-results['candidate'][m]['mse']/results['baseline'][m]['mse'])*100 for m in METRICS}
    report.update(status='complete',results=results,gains_percent=gains,
        text_oof_mse=float(np.mean((text_predictions-labels)**2)),every_row_tested_once=True,
        each_metric_meets_five_percent={m:bool(gains[m]>=5) for m in METRICS},
        promotion_allowed=bool(all(gains[m]>=5 for m in METRICS)),
        limitations=['Historical corpus used in earlier research; this is nested whole-corpus validation, not a newly collected temporal test',
                     'Random exact-text groups permit later dates in training for earlier test posts; future-time quality needs a separate check',
                     'Inner social tuning uses training-only upstream OOF scores; outer test labels are excluded from all fitting and selection'])
    np.savez_compressed(directory/'nested_predictions.npz',actual=counts,candidate=predictions,baseline=baseline,
                        actual_text_scores=labels,predicted_text_scores=text_predictions)
    write_json(directory/'nested_report.json',report)
    print(json.dumps({k:report[k] for k in ['gains_percent','text_oof_mse','promotion_allowed']}),flush=True)


if __name__=='__main__':main()

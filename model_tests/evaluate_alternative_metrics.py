"""Whole-corpus nested comparison of existing versus richer metric inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .alternative_metric_models import (AlternativeFeatures,AlternativeRegressor,SemanticNeighbour,
    AlternativeMetricHead,ReferenceMetricHead,configurations)
from .compare_social_metrics import evaluate
from .compare_text_models import write_json


METRICS=('views','reactions','comments')
STRENGTHS=(.25,.5,1.,2.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker-fold',type=int,choices=range(1,6),help='Fit one outer fold; never report partial accuracy')
    parser.add_argument('--collect-only',action='store_true',help='Verify and aggregate all five saved outer folds')
    args=parser.parse_args()
    if (args.output/'report.json').exists():raise ValueError('Completed study already exists')
    audit=json.loads((args.output/'data_audit.json').read_text(encoding='utf8'))
    corpus_hash=hashlib.sha256((args.corpus/'corpus.npz').read_bytes()).hexdigest()
    assert corpus_hash==audit['corpus_sha256']
    data=np.load(args.corpus/'corpus.npz');x=data['features'];y=data['counts'];groups=data['groups']
    texts=np.asarray(json.loads((args.output/'raw_texts.json').read_text(encoding='utf8')),dtype=object)
    old=np.load(args.previous/'predictions.npz');np.testing.assert_array_equal(old['actual'],y)
    previous_prediction=old['predicted'];baseline=old['baseline']
    predicted={name:np.full_like(y,np.nan) for name in ('same_inputs','richer_inputs')}
    grid=configurations();hits=np.zeros(len(y),int);started=time.perf_counter()
    report=dict(rows=len(y),groups=len(set(groups)),folds=[],corpus_sha256=corpus_hash,
        base_configuration_count=len(grid),strengths=STRENGTHS,target_reduction_percent=2.,
        protocol='Five frozen outer exact-text-group folds; two inner folds select each metric separately; same-input and direct-text-input pipelines evaluated separately',
        production_weights_modified=False,validation_counts_unchanged=True)
    with threadpool_limits(limits=4):
        for fold in range(1,6):
            if args.worker_fold and fold!=args.worker_fold:continue
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            prepared=joblib.load(args.previous/f'fold_{fold}'/'prepared.joblib')
            train,test=prepared['train'],prepared['test'];fit_rows=prepared['fit_rows'];test_rows=prepared['test_rows']
            assert not set(groups[train])&set(groups[test])
            selection=json.loads((args.previous/f'fold_{fold}'/'selection.json').read_text(encoding='utf8'))
            reference=joblib.load(args.previous/f'fold_{fold}'/'social_heads.joblib')
            if args.collect_only:
                saved=json.loads((path/'selection.json').read_text(encoding='utf8'))
                for scope in predicted:
                    heads=joblib.load(path/f'{scope}.joblib')
                    predicted[scope][test]=np.column_stack([heads[m].predict(test_rows,x[test],texts[test]) for m in METRICS])
                hits[test]+=1;report['folds'].append(dict(fold=fold,selected=saved['selected']))
                continue
            scores={'reference':np.asarray([selection['inner_mse'][selection['selected'][m]][i] for i,m in enumerate(METRICS)])}
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=47).split(train,groups=groups[train]))
            features={}
            for number,(name,config) in enumerate(grid.items(),1):
                errors={s:np.zeros(3) for s in STRENGTHS};n=0
                for j,(a,b) in enumerate(inner):
                    rows_a=[fit_rows[i] for i in a];rows_b=[fit_rows[i] for i in b]
                    if config['family']=='neighbour':
                        model=SemanticNeighbour(config).fit(rows_a,y[train[a]],x[train[a]],texts[train[a]])
                    else:
                        key=j,config['features']
                        if key not in features:
                            builder=AlternativeFeatures(config['features']).fit(rows_a,x[train[a]],texts[train[a]])
                            features[key]=(builder,builder.transform(rows_a,x[train[a]],texts[train[a]]))
                        builder,matrix=features[key]
                        model=AlternativeRegressor(config).fit(rows_a,y[train[a]],x[train[a]],texts[train[a]],
                            feature_builder=builder,feature_matrix=matrix)
                    prior=model._prior(rows_b);raw=model.raw(rows_b,x[train[b]],texts[train[b]])
                    for strength in STRENGTHS:
                        value=np.rint(np.maximum(0,prior+strength*(raw-prior)))
                        errors[strength]+=np.sum((value-y[train[b]])**2,axis=0)
                    n+=len(b)
                for strength,error in errors.items():scores[f'{name}@{strength:g}']=error/n
                print(f'alternatives fold {fold}/5 {number}/{len(grid)} {name}',flush=True)
            final_models={};final_features={};selected={};entry=dict(fold=fold)
            for scope in predicted:
                eligible=[key for key in scores if key=='reference' or scope=='richer_inputs' or
                          grid[key.split('@')[0]].get('features')=='native']
                chosen={m:min(eligible,key=lambda key:scores[key][i]) for i,m in enumerate(METRICS)}
                heads={}
                for i,m in enumerate(METRICS):
                    choice=chosen[m]
                    if choice=='reference':head=ReferenceMetricHead(reference[m])
                    else:
                        name,strength=choice.split('@');config=grid[name]
                        if name not in final_models:
                            if config['family']=='neighbour':
                                final_models[name]=SemanticNeighbour(config).fit(fit_rows,y[train],x[train],texts[train])
                            else:
                                mode=config['features']
                                if mode not in final_features:
                                    builder=AlternativeFeatures(mode).fit(fit_rows,x[train],texts[train])
                                    final_features[mode]=(builder,builder.transform(fit_rows,x[train],texts[train]))
                                builder,matrix=final_features[mode]
                                final_models[name]=AlternativeRegressor(config).fit(fit_rows,y[train],x[train],texts[train],
                                    feature_builder=builder,feature_matrix=matrix)
                        head=AlternativeMetricHead(final_models[name],i,float(strength))
                    predicted[scope][test,i]=head.predict(test_rows,x[test],texts[test]);heads[m]=head
                selected[scope]=chosen;joblib.dump(heads,path/f'{scope}.joblib',compress=3)
            hits[test]+=1;entry['selected']=selected;report['folds'].append(entry)
            write_json(path/'selection.json',dict(selected=selected,inner_mse={n:v.tolist() for n,v in scores.items()}))
            write_json(path/'worker_progress.json' if args.worker_fold else args.output/'progress.json',report)
            print('FOLD COMPLETE '+json.dumps(selected),flush=True)
    if args.worker_fold:
        write_json(args.output/f'fold_{args.worker_fold}'/'worker_manifest.json',
            dict(status='fold_training_complete',corpus_sha256=corpus_hash,fold=args.worker_fold,full_accuracy_evaluated=False))
        return
    assert np.all(hits==1) and all(np.isfinite(p).all() for p in predicted.values())
    base=evaluate(y,baseline);prior_metrics=evaluate(y,previous_prediction)
    report.update(status='complete',every_row_tested_once=True,baseline=base,previous=prior_metrics,
        results={scope:evaluate(y,p) for scope,p in predicted.items()},elapsed_seconds=time.perf_counter()-started)
    report['gains_percent']={scope:{m:100*(1-report['results'][scope][m]['mse']/base[m]['mse']) for m in METRICS} for scope in predicted}
    report['gains_vs_previous_percent']={scope:{m:100*(1-report['results'][scope][m]['mse']/prior_metrics[m]['mse']) for m in METRICS} for scope in predicted}
    report['passes_accuracy_gate']={scope:all(v>=2 for v in values.values()) for scope,values in report['gains_percent'].items()}
    report['limitations']=['Historical corpus repeatedly used in research; this is not a new temporal confirmation',
        'Richer-input alternatives require direct text or text embeddings in module 4, beyond its current scalar-score interface',
        'Inner social selection reuses training-only upstream OOF predictions; outer test labels are excluded from fitting and selection']
    np.savez_compressed(args.output/'predictions.npz',actual=y,baseline=baseline,previous=previous_prediction,**predicted)
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(gains_percent=report['gains_percent'],passes_accuracy_gate=report['passes_accuracy_gate'])),flush=True)


if __name__=='__main__':main()

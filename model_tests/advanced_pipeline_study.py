"""Rebuild upstream OOF scores and select three independent social heads."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import torch

from .advanced_neural_study import fit_selection
from .compare_social_metrics import evaluate
from .compare_text_models import write_json
from .conditional_kernel_metrics import KernelChannelPredictor,SingleMetricHead,configurations as kernel_configurations
from .history_candidates import ConditionalHistoryPredictor,configurations as history_configurations
from .independent_metric_candidates import RateRegressor


METRICS=('views','reactions','comments')


def grid():
    result={n:dict(kind='history',parameters=p) for n,p in history_configurations().items()}
    result.update({n:dict(kind='kernel',parameters=p) for n,p in kernel_configurations().items()})
    for alpha in (1.,100.,10000.):
        for relative in (False,True):
            result[f'ridge_{alpha:g}_{relative}']=dict(kind='ridge',parameters=dict(alpha=alpha,relative=relative))
    return result


def predict_config(configuration, training, counts, validation):
    kind=configuration['kind'];p=configuration['parameters']
    if kind=='ridge':return np.column_stack([RateRegressor(i,**p).fit(training,counts).predict(validation) for i in range(3)])
    cls=ConditionalHistoryPredictor if kind=='history' else KernelChannelPredictor
    return cls(**p).fit(training,counts).predict_rows(validation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--text-study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true',help='Prepare training-only upstream folds without reporting accuracy')
    parser.add_argument('--limit-folds',type=int,default=5,choices=range(1,6))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.limit_folds!=5 and not args.prepare_only:raise ValueError('Accuracy evaluation must cover every outer fold')
    if (args.output/'report.json').exists():raise ValueError('Completed pipeline study already exists')
    text_path=args.text_study/'report.json'
    if args.prepare_only and not text_path.exists():text_path=args.text_study/'progress.json'
    text_report=json.loads(text_path.read_text(encoding='utf8'))
    corpus_hash=hashlib.sha256((args.corpus/'corpus.npz').read_bytes()).hexdigest()
    assert text_report['corpus_sha256']==corpus_hash
    data=np.load(args.corpus/'corpus.npz');x=data['features'];y=data['labels'];counts=data['counts'];groups=data['groups']
    rows=json.loads((args.corpus/'rows.json').read_text(encoding='utf8'))
    configurations=grid();prediction=np.full_like(counts,np.nan);baseline=np.full_like(counts,np.nan)
    hits=np.zeros(len(rows),int);started=time.perf_counter()
    report=dict(rows=len(rows),groups=len(set(groups)),folds=[],configuration_count=len(configurations),
        protocol='Five frozen outer exact-text-group folds; selected text heads refitted in three training-only OOF folds; independent inner social selection',
        production_weights_modified=False,corpus_sha256=corpus_hash,all_validation_targets_uncapped=True)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for fold,(train,test) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=20261111).split(x,groups=groups),1):
            if fold>args.limit_folds:break
            assert not set(groups[train])&set(groups[test])
            selection=text_report['folds'][fold-1]['selected']
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            if (path/'prepared.joblib').exists():
                prepared=joblib.load(path/'prepared.joblib')
                assert prepared['corpus_sha256']==corpus_hash and prepared['selection']==selection
                np.testing.assert_array_equal(prepared['train'],train);np.testing.assert_array_equal(prepared['test'],test)
            else:
                scores=np.full(len(train),np.nan);seen=np.zeros(len(train),int);audit=[]
                for j,(a,b) in enumerate(GroupKFold(n_splits=3,shuffle=True,random_state=42).split(train,groups=groups[train]),1):
                    assert not set(groups[train[a]])&set(groups[train[b]])
                    model=fit_selection(selection,x,y,train[a],path/f'upstream_{j}',corpus_hash)
                    scores[b]=model.predict(x[train[b]]);seen[b]+=1
                    audit.append(dict(fold=j,training_rows=len(a),validation_rows=len(b),exact_text_overlap=0))
                    print(f'pipeline fold {fold}/5 upstream OOF {j}/3 complete',flush=True)
                assert np.all(seen==1) and np.isfinite(scores).all()
                test_head=joblib.load(args.text_study/f'fold_{fold}'/'predictor.joblib')
                test_scores=test_head.predict(x[test])
                prepared=dict(train=train,test=test,selection=selection,corpus_sha256=corpus_hash,audit=audit,
                    fit_rows=[dict(rows[i],score=float(s)) for i,s in zip(train,scores)],
                    test_rows=[dict(rows[i],score=float(s)) for i,s in zip(test,test_scores)])
                joblib.dump(prepared,path/'prepared.joblib',compress=3)
            if args.prepare_only:
                print(f'pipeline fold {fold}/5 training preparation cached; accuracy not evaluated',flush=True)
                continue
            fitting=prepared['fit_rows'];validation=prepared['test_rows']
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=47).split(train,groups=groups[train]))
            scores={}
            for number,(name,configuration) in enumerate(configurations.items(),1):
                sse=np.zeros(3);n=0
                for a,b in inner:
                    p=predict_config(configuration,[fitting[i] for i in a],counts[train[a]],[fitting[i] for i in b])
                    sse+=np.sum((p-counts[train[b]])**2,axis=0);n+=len(b)
                scores[name]=sse/n
                if number%30==0:print(f'pipeline fold {fold}/5 social {number}/{len(configurations)}',flush=True)
            chosen={m:min(scores,key=lambda n:scores[n][i]) for i,m in enumerate(METRICS)}
            heads={}
            for i,m in enumerate(METRICS):
                head=SingleMetricHead(i,configurations[chosen[m]]).fit(fitting,counts[train])
                prediction[test,i]=head.predict(validation);heads[m]=head
            baseline[test]=ConditionalHistoryPredictor().fit(fitting,counts[train]).predict_rows(validation)
            hits[test]+=1;joblib.dump(heads,path/'social_heads.joblib',compress=3)
            entry=dict(fold=fold,text_selection=selection,selected=chosen,
                baseline=evaluate(counts[test],baseline[test]),candidate=evaluate(counts[test],prediction[test]),
                upstream_audit=prepared['audit'])
            report['folds'].append(entry)
            write_json(path/'selection.json',dict(selected=chosen,inner_mse={n:v.tolist() for n,v in scores.items()}))
            write_json(args.output/'progress.json',report)
            print('pipeline fold '+str(fold)+'/5 selected '+json.dumps(chosen),flush=True)
    if args.prepare_only:
        write_json(args.output/'preparation.json',dict(status='training_preparation_only',prepared_folds=args.limit_folds,accuracy_evaluated=False))
        return
    assert np.all(hits==1) and np.isfinite(prediction).all()
    base=evaluate(counts,baseline);candidate=evaluate(counts,prediction)
    gains={m:100*(1-candidate[m]['mse']/base[m]['mse']) for m in METRICS}
    report.update(status='complete',baseline=base,candidate=candidate,gains_percent=gains,
        every_row_tested_once=True,passes_accuracy_gate=bool(all(v>=5. for v in gains.values())),
        elapsed_seconds=time.perf_counter()-started,
        limitations=['Historical corpus reused in research; a new temporal confirmation is required',
                     'Social inner selection reuses training-only upstream OOF scores; all outer test labels are excluded from fitting and selection'])
    np.savez_compressed(args.output/'predictions.npz',actual=counts,predicted=prediction,baseline=baseline)
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(gains_percent=gains,passes_accuracy_gate=report['passes_accuracy_gate'])),flush=True)


if __name__=='__main__':main()

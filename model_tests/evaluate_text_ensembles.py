"""Nested full-corpus comparison of regularized text-only neural networks."""
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
from src.database.embedding_config import EMBEDDING_SIGNATURE
from .compare_text_models import write_json


CONFIGURATIONS={
    'compact':dict(hidden_layers=(128,),dropout=.5,epochs=25,learning_rate=.0005,weight_decay=.1),
    'narrow':dict(hidden_layers=(64,),dropout=.3,epochs=35,learning_rate=.0003,weight_decay=.3),
    'bottleneck':dict(hidden_layers=(32,),dropout=.2,epochs=25,learning_rate=.0005,weight_decay=.3),
    'wide_regularized':dict(hidden_layers=(512,128),dropout=.65,epochs=25,learning_rate=.0005,weight_decay=.2),
}
SEEDS=(42,73)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Completed experiment already exists')
    audit=json.loads((args.corpus/'audit.json').read_text(encoding='utf8'))
    assert audit['fixed_embedding_model']==EMBEDDING_SIGNATURE
    data=np.load(args.corpus/'corpus.npz');x=data['features'];y=data['labels'];groups=data['groups']
    prediction=np.full(len(y),np.nan);hits=np.zeros(len(y),int)
    report=dict(rows=len(y),groups=len(set(groups)),configurations=CONFIGURATIONS,seeds=SEEDS,folds=[],
        protocol='Five outer exact-text-group folds; two inner folds choose architecture and one/two seeds using training rows only',
        prediction_inputs=['text','topic'],production_weights_modified=False)
    started=time.perf_counter();torch.set_num_threads(4)

    def fitted(path,indices,config,seed):
        if path.exists():return joblib.load(path)
        head=AssesmentConformPPtextBased(seed=seed,num_threads=4,**config).train_features(x[indices],y[indices])
        joblib.dump(head,path,compress=3);return head

    with threadpool_limits(limits=4):
        for fold,(train,test) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=20261111).split(x,groups=groups),1):
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            assert not set(groups[train])&set(groups[test])
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=91).split(train,groups=groups[train]))
            scores={}
            for name,config in CONFIGURATIONS.items():
                sse={1:0.,2:0.};n=0
                for j,(a,b) in enumerate(inner,1):
                    values=[]
                    for seed in SEEDS:
                        model=fitted(path/f'inner_{j}_{name}_{seed}.joblib',train[a],config,seed)
                        values.append(model.model.predict(x[train[b]]))
                    for size in (1,2):sse[size]+=float(np.sum((np.mean(values[:size],axis=0)-y[train[b]])**2))
                    n+=len(b)
                for size in (1,2):scores[f'{name}:{size}']=sse[size]/n
                print(f'text fold {fold}/5 {name}: single {sse[1]/n:.5f}, ensemble {sse[2]/n:.5f}',flush=True)
            selected=min(scores,key=scores.get);name,size=selected.split(':');size=int(size)
            values=[];models=[]
            for seed in SEEDS[:size]:
                model=fitted(path/f'selected_{name}_{seed}.joblib',train,CONFIGURATIONS[name],seed)
                models.append(model);values.append(model.model.predict(x[test]))
            prediction[test]=np.mean(values,axis=0);hits[test]+=1
            # Warm regression timing excludes SBERT and startup; end-to-end timing is separate.
            for model in models:model.model.predict(x[test[:1]])
            times=[]
            for _ in range(30):
                t=time.perf_counter()
                for model in models:model.model.predict(x[test[:1]])
                times.append(time.perf_counter()-t)
            report['folds'].append(dict(fold=fold,selected=selected,inner_mse=scores,
                outer_mse=float(np.mean((prediction[test]-y[test])**2)),
                warm_head_max_seconds=max(times),warm_head_mean_seconds=float(np.mean(times))))
            write_json(args.output/'progress.json',report)
            print(f'text fold {fold}/5 selected {selected}; outer MSE {report["folds"][-1]["outer_mse"]:.5f}',flush=True)
    assert np.all(hits==1) and np.isfinite(prediction).all()
    baseline=json.loads((args.corpus/'text_baseline.json').read_text(encoding='utf8'))['mse']
    previous=json.loads((args.corpus/'nested_report.json').read_text(encoding='utf8'))['text_oof_mse']
    mse=float(np.mean((prediction-y)**2))
    report.update(status='complete',mse=mse,baseline_mse=baseline,previous_nested_mse=previous,
        reduction_vs_baseline_percent=100*(1-mse/baseline),reduction_vs_previous_nested_percent=100*(1-mse/previous),
        every_row_tested_once=True,elapsed_seconds=time.perf_counter()-started,
        limitations=['The historical corpus has been used in earlier research; a new future-time confirmation is still required',
                     'Text-only improvement does not establish that the downstream social models meet their 5 percent requirement'])
    np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=prediction)
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(mse=mse,reduction_vs_baseline_percent=report['reduction_vs_baseline_percent'])),flush=True)


if __name__=='__main__':main()

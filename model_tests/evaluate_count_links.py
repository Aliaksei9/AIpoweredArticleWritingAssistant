"""Whole-corpus nested selection of positive-link social metric heads."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.database.embedding_config import EMBEDDING_SIGNATURE
from .compare_social_metrics import evaluate
from .compare_text_models import write_json
from .count_link_candidates import CalibratedCountLink,configurations
from .history_candidates import ConditionalHistoryPredictor


METRICS=('views','reactions','comments')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Completed experiment already exists')
    audit=json.loads((args.corpus/'audit.json').read_text(encoding='utf8'))
    assert audit['fixed_embedding_model']==EMBEDDING_SIGNATURE
    data=np.load(args.corpus/'corpus.npz');counts=data['counts'];groups=data['groups']
    prediction=np.full_like(counts,np.nan);baseline=np.full_like(counts,np.nan)
    hits=np.zeros(len(counts),dtype=int);grid=configurations()
    report=dict(rows=len(counts),groups=len(set(groups)),folds=[],configuration_count=len(grid),
                protocol='Five frozen outer exact-text-group folds; independent inner selection for each metric; outer labels excluded from fitting',
                production_weights_modified=False,full_uncapped_targets=True)
    started=time.perf_counter()
    with threadpool_limits(limits=4):
        for fold in range(1,6):
            prepared=joblib.load(args.corpus/f'outer_{fold}'/'prepared.joblib')
            train,test=prepared['train'],prepared['test']
            assert not set(groups[train])&set(groups[test])
            rows=prepared['fit_rows'];validation=prepared['test_rows']
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=47).split(train,groups=groups[train]))
            chosen={};errors={};timings={}
            for metric,name in enumerate(METRICS):
                mse={}
                for key,config in grid.items():
                    sse=0.;n=0
                    for a,b in inner:
                        model=CalibratedCountLink(metric,**config).fit([rows[i] for i in a],counts[train[a]])
                        predicted=model.predict([rows[i] for i in b])
                        sse+=float(np.sum((predicted-counts[train[b],metric])**2));n+=len(b)
                    mse[key]=sse/n
                # Channel mean is an explicit candidate to avoid forcing a harmful departure.
                sse=0.;n=0
                for a,b in inner:
                    mean=ConditionalHistoryPredictor().fit([rows[i] for i in a],counts[train[a]])
                    value=mean.predict_rows([rows[i] for i in b])[:,metric]
                    sse+=float(np.sum((value-counts[train[b],metric])**2));n+=len(b)
                mse['channel_mean']=sse/n
                key=min(mse,key=mse.get);chosen[name]=key;errors[name]=mse
                if key=='channel_mean':
                    model=ConditionalHistoryPredictor().fit(rows,counts[train])
                    prediction[test,metric]=model.predict_rows(validation)[:,metric]
                else:
                    model=CalibratedCountLink(metric,**grid[key]).fit(rows,counts[train])
                    t=time.perf_counter();prediction[test,metric]=model.predict(validation)
                    timings[name]=(time.perf_counter()-t)/len(test)
                joblib.dump(model,args.output/f'fold_{fold}_{name}.joblib',compress=3)
                print(f'fold {fold}/5 {name}: {key}',flush=True)
            baseline[test]=ConditionalHistoryPredictor().fit(rows,counts[train]).predict_rows(validation)
            hits[test]+=1
            report['folds'].append(dict(fold=fold,selected=chosen,inner_mse=errors,seconds_per_row=timings))
            write_json(args.output/'progress.json',report)
    assert np.all(hits==1) and np.isfinite(prediction).all()
    base=evaluate(counts,baseline);candidate=evaluate(counts,prediction)
    gains={m:100.*(1.-candidate[m]['mse']/base[m]['mse']) for m in METRICS}
    report.update(status='complete',baseline=base,candidate=candidate,gains_percent=gains,
                  each_row_tested_once=True,promotion_allowed=bool(all(v>=5. for v in gains.values())),
                  elapsed_seconds=time.perf_counter()-started,
                  limitations=['Historical corpus previously used for research; this is not a new future-time holdout',
                               'Upstream score features are frozen training-only OOF predictions; inner social selection reuses these outer-training predictions'])
    np.savez_compressed(args.output/'predictions.npz',actual=counts,baseline=baseline,candidate=prediction)
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(gains_percent=gains,promotion_allowed=report['promotion_allowed'])),flush=True)


if __name__=='__main__':main()

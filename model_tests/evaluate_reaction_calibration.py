"""Full validation of horizon forecasts plus OOF reaction calibration."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .alternative_metric_models import AlternativeMetricHead
from .calibrated_reactions import CalibratedReactionHead,calibration_coefficient
from .compare_social_metrics import evaluate
from .compare_text_models import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','previous','calendar','texts','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Completed calibration exists')
    data=np.load(args.corpus/'corpus.npz');x=data['features'];y=data['counts'];groups=data['groups']
    texts=np.asarray(json.loads(args.texts.read_text(encoding='utf8')),dtype=object)
    current=np.load(args.calendar/'predictions.npz');prediction=current['predicted'].copy();old=np.load(args.previous/'predictions.npz')
    hits=np.zeros(len(y),int);report=dict(rows=len(y),folds=[],production_weights_modified=False,
        protocol='Five frozen outer text-group folds; a reaction coefficient is fitted on training-only inner OOF predictions')
    with threadpool_limits(limits=4):
        for fold in range(1,6):
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            prepared=joblib.load(args.previous/f'fold_{fold}'/'prepared.joblib');train,test=prepared['train'],prepared['test']
            rows=prepared['fit_rows'];validation=prepared['test_rows']
            assert not set(groups[train])&set(groups[test])
            heads=joblib.load(args.calendar/f'fold_{fold}'/'heads.joblib');head=heads['reactions']
            oof=np.full(len(train),np.nan);prior=np.full(len(train),np.nan)
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=47).split(train,groups=groups[train]))
            for a,b in inner:
                fitting=[rows[i] for i in a];validating=[rows[i] for i in b]
                model=type(head.model)(head.model.configuration).fit(fitting,y[train[a]],x[train[a]],texts[train[a]])
                fitted=AlternativeMetricHead(model,1,head.strength)
                oof[b]=fitted.predict(validating,x[train[b]],texts[train[b]]);prior[b]=model._prior(validating)[:,1]
            assert np.isfinite(oof).all()
            base_error=float(np.sum((oof-y[train,1])**2));calibrated_error=0.
            for a,b in inner:
                coefficient=calibration_coefficient(y[train[a],1],prior[a],oof[a])
                value=np.rint(np.maximum(0,prior[b]+coefficient*(oof[b]-prior[b])))
                calibrated_error+=float(np.sum((value-y[train[b],1])**2))
            coefficient=calibration_coefficient(y[train,1],prior,oof) if calibrated_error<base_error else 1.
            heads['reactions']=CalibratedReactionHead(head,coefficient)
            prediction[test,1]=heads['reactions'].predict(validation,x[test],texts[test]);hits[test]+=1
            joblib.dump(heads,path/'heads.joblib',compress=3)
            report['folds'].append(dict(fold=fold,coefficient=coefficient,base_inner_sse=base_error,calibrated_inner_sse=calibrated_error))
            print('CALIBRATION FOLD '+str(fold)+' coefficient '+str(coefficient),flush=True)
    assert np.all(hits==1)
    baseline=evaluate(y,old['baseline']);candidate=evaluate(y,prediction)
    gains={m:100*(1-candidate[m]['mse']/baseline[m]['mse']) for m in ('views','reactions','comments')}
    report.update(status='complete',every_row_tested_once=True,baseline=baseline,candidate=candidate,gains_percent=gains,
                  passes_accuracy_gate=bool(all(v>=2 for v in gains.values())))
    np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=prediction)
    write_json(args.output/'report.json',report);print(json.dumps(gains),flush=True)


if __name__=='__main__':main()

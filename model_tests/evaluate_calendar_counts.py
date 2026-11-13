"""Full nested validation of publication-time and nominal horizon features."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .alternative_metric_models import predict_heads
from .calendar_count_models import CalendarCountRegressor,CalendarMetricHead
from .compare_social_metrics import evaluate
from .compare_text_models import write_json


METRICS=('views','reactions','comments')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','previous','classical','texts','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Completed calendar study exists')
    data=np.load(args.corpus/'corpus.npz');x=data['features'];y=data['counts'];groups=data['groups']
    texts=np.asarray(json.loads(args.texts.read_text(encoding='utf8')),dtype=object)
    old=np.load(args.previous/'predictions.npz');prediction=np.full_like(y,np.nan);hits=np.zeros(len(y),int)
    older=json.loads(Path('model_tests/results/expansion_20261111/collection.json').read_text(encoding='utf8'))['as_of']
    control=json.loads(Path('model_tests/results/history_20261111/control_protocol.json').read_text(encoding='utf8'))['created_at']
    asofs=[older]*11493+[control]*216
    def augment(rows,indices):
        return [dict(r,horizon_days=(datetime.fromisoformat(asofs[i])-datetime.fromisoformat(r['published_at'])).total_seconds()/86400.) for r,i in zip(rows,indices)]
    report=dict(rows=len(y),folds=[],production_weights_modified=False,
        protocol='Five frozen outer text-group folds; training-only selection of calendar features against the previous direct-text pipeline',
        limitations=['Calendar/horizon is an extra input, unavailable to the current module-4 signature',
            'Nominal age and publication-date trend are confounded in a single cross-sectional measurement; this is not evidence of a causal maturation curve'])
    with threadpool_limits(limits=4):
        for fold in range(1,6):
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            prepared=joblib.load(args.previous/f'fold_{fold}'/'prepared.joblib');train,test=prepared['train'],prepared['test']
            rows=augment(prepared['fit_rows'],train);validation=augment(prepared['test_rows'],test)
            choice=json.loads((args.classical/f'fold_{fold}'/'selection.json').read_text(encoding='utf8'))
            scores={'reference':np.asarray([choice['inner_mse'][choice['selected']['richer_inputs'][m]][i] for i,m in enumerate(METRICS)])}
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=47).split(train,groups=groups[train]))
            for alpha in (10.,1000.,100000.):
                errors={s:np.zeros(3) for s in (.25,.5,1.)};n=0
                for a,b in inner:
                    r_a=[rows[i] for i in a];r_b=[rows[i] for i in b]
                    model=CalendarCountRegressor(alpha).fit(r_a,y[train[a]])
                    prior=model._prior(r_b);value=model.predict(r_b)
                    for strength in errors:
                        predicted=np.rint(np.maximum(0,prior+strength*(value-prior)))
                        errors[strength]+=np.sum((predicted-y[train[b]])**2,axis=0)
                    n+=len(b)
                for strength,error in errors.items():scores[f'{alpha:g}@{strength:g}']=error/n
            chosen={m:min(scores,key=lambda n:scores[n][i]) for i,m in enumerate(METRICS)}
            heads={};models={};reference=joblib.load(args.classical/f'fold_{fold}'/'richer_inputs.joblib')
            for i,m in enumerate(METRICS):
                if chosen[m]=='reference':heads[m]=reference[m]
                else:
                    alpha,strength=chosen[m].split('@');alpha=float(alpha)
                    if alpha not in models:models[alpha]=CalendarCountRegressor(alpha).fit(rows,y[train])
                    heads[m]=CalendarMetricHead(models[alpha],i,float(strength))
                prediction[test,i]=heads[m].predict(validation,x[test],texts[test])
            hits[test]+=1;joblib.dump(heads,path/'heads.joblib',compress=3)
            report['folds'].append(dict(fold=fold,selected=chosen));write_json(args.output/'progress.json',report)
            print('CALENDAR FOLD COMPLETE '+str(fold)+' '+json.dumps(chosen),flush=True)
    assert np.all(hits==1) and np.isfinite(prediction).all()
    baseline=evaluate(y,old['baseline']);candidate=evaluate(y,prediction)
    report.update(status='complete',every_row_tested_once=True,baseline=baseline,candidate=candidate,
        gains_percent={m:100*(1-candidate[m]['mse']/baseline[m]['mse']) for m in METRICS})
    np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=prediction)
    write_json(args.output/'report.json',report);print(json.dumps(report['gains_percent']),flush=True)


if __name__=='__main__':main()

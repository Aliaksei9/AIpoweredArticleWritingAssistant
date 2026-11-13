"""Whole-corpus nested test of cross-resource semantic transfer."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .alternative_metric_models import AlternativeMetricHead,ReferenceMetricHead
from .compare_social_metrics import evaluate
from .compare_text_models import write_json
from .cross_resource_transfer import CrossResourceTransfer,configurations


METRICS=('views','reactions','comments');STRENGTHS=(.1,.25,.5,1.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--texts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Completed transfer experiment exists')
    data=np.load(args.corpus/'corpus.npz');x=data['features'];y=data['counts'];groups=data['groups']
    texts=np.asarray(json.loads(args.texts.read_text(encoding='utf8')),dtype=object)
    old=np.load(args.previous/'predictions.npz');np.testing.assert_array_equal(old['actual'],y)
    prediction=np.full_like(y,np.nan);hits=np.zeros(len(y),int);grid=configurations()
    report=dict(rows=len(y),folds=[],configuration_count=len(grid)*len(STRENGTHS),production_weights_modified=False,
                protocol='Five frozen outer exact-text-group folds; two training-only inner folds choose each metric and transfer strength')
    with threadpool_limits(limits=4):
        for fold in range(1,6):
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            prepared=joblib.load(args.previous/f'fold_{fold}'/'prepared.joblib')
            train,test=prepared['train'],prepared['test'];rows=prepared['fit_rows'];validation=prepared['test_rows']
            assert not set(groups[train])&set(groups[test])
            previous_choice=json.loads((args.previous/f'fold_{fold}'/'selection.json').read_text(encoding='utf8'))
            scores={'reference':np.asarray([previous_choice['inner_mse'][previous_choice['selected'][m]][i] for i,m in enumerate(METRICS)])}
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=47).split(train,groups=groups[train]))
            cache=[]
            for a,b in inner:
                fitting=[rows[i] for i in a];validating=[rows[i] for i in b]
                model=CrossResourceTransfer(next(iter(grid.values()))).fit(fitting,y[train[a]],x[train[a]],texts[train[a]])
                cache.append((model,{exclude:model.retrieval(validating,x[train[b]],exclude=exclude) for exclude in (False,True)}))
            for name,config in grid.items():
                errors={s:np.zeros(3) for s in STRENGTHS};n=0
                for j,(a,b) in enumerate(inner):
                    model,retrieved=cache[j];model.configuration=config
                    validating=[rows[i] for i in b];prior=model._prior(validating)
                    raw=model.raw(validating,x[train[b]],texts[train[b]],retrieved=retrieved[config['exclude_resource']])
                    for strength in STRENGTHS:
                        value=np.rint(np.maximum(0,prior+strength*(raw-prior)))
                        errors[strength]+=np.sum((value-y[train[b]])**2,axis=0)
                    n+=len(b)
                for strength,error in errors.items():scores[f'{name}@{strength:g}']=error/n
            chosen={m:min(scores,key=lambda n:scores[n][i]) for i,m in enumerate(METRICS)}
            heads={};models={};reference=joblib.load(args.previous/f'fold_{fold}'/'social_heads.joblib')
            for i,m in enumerate(METRICS):
                if chosen[m]=='reference':head=ReferenceMetricHead(reference[m])
                else:
                    name,strength=chosen[m].split('@')
                    if name not in models:models[name]=CrossResourceTransfer(grid[name]).fit(rows,y[train],x[train],texts[train])
                    head=AlternativeMetricHead(models[name],i,float(strength))
                prediction[test,i]=head.predict(validation,x[test],texts[test]);heads[m]=head
            hits[test]+=1;joblib.dump(heads,path/'heads.joblib',compress=3)
            write_json(path/'selection.json',dict(selected=chosen,inner_mse={n:v.tolist() for n,v in scores.items()}))
            report['folds'].append(dict(fold=fold,selected=chosen));write_json(args.output/'progress.json',report)
            print('TRANSFER FOLD COMPLETE '+str(fold)+' '+json.dumps(chosen),flush=True)
    assert np.all(hits==1) and np.isfinite(prediction).all()
    baseline=evaluate(y,old['baseline']);candidate=evaluate(y,prediction)
    report.update(status='complete',every_row_tested_once=True,baseline=baseline,candidate=candidate,
        gains_percent={m:100*(1-candidate[m]['mse']/baseline[m]['mse']) for m in METRICS})
    np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=prediction)
    write_json(args.output/'report.json',report);print(json.dumps(report['gains_percent']),flush=True)


if __name__=='__main__':main()

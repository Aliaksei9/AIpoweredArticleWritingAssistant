"""Full nested evaluation of three independent direct neural count heads."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import torch

from .alternative_metric_models import AlternativeMetricHead,ReferenceMetricHead
from .compare_social_metrics import evaluate
from .compare_text_models import write_json
from .direct_metric_neural import DirectMetricNeural,CONFIGURATIONS


METRICS=('views','reactions','comments');STRENGTHS=(.25,.5,1.,2.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True);parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--texts',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Completed experiment exists')
    data=np.load(args.corpus/'corpus.npz');x=data['features'];y=data['counts'];groups=data['groups']
    texts=np.asarray(json.loads(args.texts.read_text(encoding='utf8')),dtype=object)
    old=np.load(args.previous/'predictions.npz');np.testing.assert_array_equal(old['actual'],y)
    predictions=np.full_like(y,np.nan);hits=np.zeros(len(y),int)
    report=dict(rows=len(y),folds=[],protocol='Five frozen outer text-group folds; independent inner selection for each count head',
                target_reduction_percent=2.,production_weights_modified=False)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for fold in range(1,6):
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            prepared=joblib.load(args.previous/f'fold_{fold}'/'prepared.joblib')
            train,test=prepared['train'],prepared['test'];rows=prepared['fit_rows'];validation=prepared['test_rows']
            assert not set(groups[train])&set(groups[test])
            previous_choice=json.loads((args.previous/f'fold_{fold}'/'selection.json').read_text(encoding='utf8'))
            scores={'reference':np.asarray([previous_choice['inner_mse'][previous_choice['selected'][m]][i] for i,m in enumerate(METRICS)])}
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=47).split(train,groups=groups[train]))
            for name,config in CONFIGURATIONS.items():
                errors={s:np.zeros(3) for s in STRENGTHS};n=0
                for j,(a,b) in enumerate(inner):
                    cache=path/f'inner_{j}_{name}.joblib'
                    if cache.exists():model=joblib.load(cache)
                    else:
                        model=DirectMetricNeural(config).fit([rows[i] for i in a],y[train[a]],x[train[a]],texts[train[a]])
                        joblib.dump(model,cache,compress=3)
                    validation_rows=[rows[i] for i in b]
                    prior=model._prior(validation_rows);raw=model.raw(validation_rows,x[train[b]],texts[train[b]])
                    for strength in STRENGTHS:
                        p=np.rint(np.maximum(0,prior+strength*(raw-prior)))
                        errors[strength]+=np.sum((p-y[train[b]])**2,axis=0)
                    n+=len(b)
                for strength,error in errors.items():scores[f'{name}@{strength:g}']=error/n
                print(f'neural counts fold {fold}/5 {name} complete',flush=True)
            chosen={m:min(scores,key=lambda n:scores[n][i]) for i,m in enumerate(METRICS)}
            models={};heads={};reference=joblib.load(args.previous/f'fold_{fold}'/'social_heads.joblib')
            for i,m in enumerate(METRICS):
                name=chosen[m]
                if name=='reference':head=ReferenceMetricHead(reference[m])
                else:
                    key,strength=name.split('@')
                    if key not in models:
                        models[key]=DirectMetricNeural(CONFIGURATIONS[key]).fit(rows,y[train],x[train],texts[train])
                    head=AlternativeMetricHead(models[key],i,float(strength))
                predictions[test,i]=head.predict(validation,x[test],texts[test]);heads[m]=head
            hits[test]+=1;joblib.dump(heads,path/'heads.joblib',compress=3)
            entry=dict(fold=fold,selected=chosen,inner_mse={n:v.tolist() for n,v in scores.items()})
            report['folds'].append(dict(fold=fold,selected=chosen));write_json(path/'selection.json',entry)
            write_json(args.output/'progress.json',report);print('NEURAL FOLD COMPLETE '+json.dumps(chosen),flush=True)
    assert np.all(hits==1) and np.isfinite(predictions).all()
    baseline=evaluate(y,old['baseline']);candidate=evaluate(y,predictions)
    report.update(status='complete',every_row_tested_once=True,baseline=baseline,candidate=candidate,
        gains_percent={m:100*(1-candidate[m]['mse']/baseline[m]['mse']) for m in METRICS})
    np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=predictions)
    write_json(args.output/'report.json',report);print(json.dumps(report['gains_percent']),flush=True)


if __name__=='__main__':main()

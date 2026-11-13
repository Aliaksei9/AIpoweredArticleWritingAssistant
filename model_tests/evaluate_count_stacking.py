"""Full outer evaluation of a small corrector on OOF metric predictions."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .alternative_metric_models import (AlternativeRegressor,SemanticNeighbour,AlternativeMetricHead,
    ReferenceMetricHead,configurations,predict_heads)
from .conditional_kernel_metrics import SingleMetricHead
from .compare_social_metrics import evaluate
from .compare_text_models import write_json
from .history_candidates import ConditionalHistoryPredictor
from .residual_count_stacking import CountCorrection,StackedCountPipeline,StackedMetricHead


METRICS=('views','reactions','comments')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','previous','classical','texts','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Completed stacking study exists')
    data=np.load(args.corpus/'corpus.npz');x=data['features'];y=data['counts'];groups=data['groups']
    texts=np.asarray(json.loads(args.texts.read_text(encoding='utf8')),dtype=object)
    old=np.load(args.previous/'predictions.npz');base_outer=np.load(args.classical/'predictions.npz')['richer_inputs']
    prediction=np.full_like(y,np.nan);hits=np.zeros(len(y),int);grid=configurations()
    report=dict(rows=len(y),folds=[],production_weights_modified=False,
        protocol='Five frozen outer text-group folds; correction targets are residuals of training-only OOF count predictions')
    with threadpool_limits(limits=4):
        for fold in range(1,6):
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            prepared=joblib.load(args.previous/f'fold_{fold}'/'prepared.joblib')
            train,test=prepared['train'],prepared['test'];rows=prepared['fit_rows'];validation=prepared['test_rows']
            assert not set(groups[train])&set(groups[test])
            chosen=json.loads((args.classical/f'fold_{fold}'/'selection.json').read_text(encoding='utf8'))['selected']['richer_inputs']
            reference=joblib.load(args.previous/f'fold_{fold}'/'social_heads.joblib')
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=47).split(train,groups=groups[train]))
            base_oof=np.full((len(train),3),np.nan);priors=np.full_like(base_oof,np.nan)
            for a,b in inner:
                fitting=[rows[i] for i in a];validating=[rows[i] for i in b];models={};heads={}
                for i,m in enumerate(METRICS):
                    key=chosen[m]
                    if key=='reference':head=ReferenceMetricHead(SingleMetricHead(i,reference[m].configuration).fit(fitting,y[train[a]]))
                    else:
                        name,strength=key.split('@');config=grid[name]
                        if name not in models:
                            cls=SemanticNeighbour if config['family']=='neighbour' else AlternativeRegressor
                            models[name]=cls(config).fit(fitting,y[train[a]],x[train[a]],texts[train[a]])
                        head=AlternativeMetricHead(models[name],i,float(strength))
                    heads[m]=head
                base_oof[b]=predict_heads(heads,validating,x[train[b]],texts[train[b]])
                priors[b]=ConditionalHistoryPredictor().fit(fitting,y[train[a]]).predict_rows(validating)
            assert np.isfinite(base_oof).all()
            scores={'base':np.mean((base_oof-y[train])**2,axis=0)}
            for alpha in (10.,1000.,100000.):
                errors={s:np.zeros(3) for s in (.25,.5,1.)};n=0
                for a,b in inner:
                    model=CountCorrection(alpha).fit([rows[i] for i in a],y[train[a]],base_oof[a],priors[a])
                    raw=model.raw([rows[i] for i in b],base_oof[b])
                    for strength in errors:
                        value=np.rint(np.maximum(0,base_oof[b]+strength*(raw-base_oof[b])))
                        errors[strength]+=np.sum((value-y[train[b]])**2,axis=0)
                    n+=len(b)
                for strength,error in errors.items():scores[f'{alpha:g}@{strength:g}']=error/n
            choice={m:min(scores,key=lambda n:scores[n][i]) for i,m in enumerate(METRICS)}
            models={};corrections={};base_heads=joblib.load(args.classical/f'fold_{fold}'/'richer_inputs.joblib')
            for i,m in enumerate(METRICS):
                if choice[m]=='base':continue
                alpha,strength=choice[m].split('@');alpha=float(alpha)
                if alpha not in models:models[alpha]=CountCorrection(alpha).fit(rows,y[train],base_oof,priors)
                corrections[i]=(models[alpha],float(strength))
            pipeline=StackedCountPipeline(base_heads,corrections)
            prediction[test]=pipeline.predict(validation,x[test],texts[test]);hits[test]+=1
            heads={m:StackedMetricHead(pipeline,i) for i,m in enumerate(METRICS)}
            joblib.dump(heads,path/'heads.joblib',compress=3)
            report['folds'].append(dict(fold=fold,selected=choice));write_json(args.output/'progress.json',report)
            print('STACK FOLD COMPLETE '+str(fold)+' '+json.dumps(choice),flush=True)
    assert np.all(hits==1) and np.isfinite(prediction).all()
    baseline=evaluate(y,old['baseline']);candidate=evaluate(y,prediction)
    report.update(status='complete',every_row_tested_once=True,baseline=baseline,candidate=candidate,
        gains_percent={m:100*(1-candidate[m]['mse']/baseline[m]['mse']) for m in METRICS},
        limitations=['The corrector inner search reuses training-only OOF base predictions; outer tests are excluded from the entire fit',
            'Historical corpus used for research, not a fresh temporal confirmation'])
    np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=prediction,baseline=old['baseline'],previous=old['predicted'])
    write_json(args.output/'report.json',report);print(json.dumps(report['gains_percent']),flush=True)


if __name__=='__main__':main()

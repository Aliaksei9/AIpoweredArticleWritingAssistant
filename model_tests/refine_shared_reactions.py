"""Nested reaction-only tuning; identical full inputs, all unique articles tested."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.social_metrics_prediction.unified_features import INPUT_FIELDS
from src.social_metrics_prediction.unified_models import UnifiedRateModel, UnifiedMetricHead
from .compare_social_metrics import evaluate
from .compare_text_models import write_json


from src.social_metrics_prediction.reaction_selection import (
    reaction_configurations as grid, select_reaction_configuration as select,
)


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    data=joblib.load(args.corpus);x=data['features'];vectors=x[:,:x.shape[1]//2]
    y=data['counts'];groups=data['groups'];old=np.load(args.previous/'predictions.npz')
    assert len(set(data['article_ids']))==len(y)
    predictions=old['predicted'].copy();hits=np.zeros(len(y),int);selected=[]
    with threadpool_limits(limits=4):
        for fold in range(1,6):
            target=args.output/f'fold_{fold}.joblib'
            p=joblib.load(args.upstream/f'fold_{fold}/models.joblib');a,b=p['train'],p['test']
            assert not set(groups[a])&set(groups[b])
            if target.exists():result=joblib.load(target)
            else:
                configuration,audit=select(p['fit_rows'],y[a],vectors[a],groups[a],folds=args.inner_folds,
                    on_progress=lambda line:print(f'Outer {fold}: {line}',flush=True))
                encoder=joblib.load(args.previous/f'fold_{fold}/models.joblib')['encoder']
                matrix=encoder.transform(p['fit_rows'],vectors[a]);test=encoder.transform(p['test_rows'],vectors[b])
                params=dict(configuration);strength=params.pop('strength')
                model=UnifiedRateModel(**params).fit(matrix,p['fit_rows'],y[a])
                head=UnifiedMetricHead(model,1,strength)
                result=dict(configuration=configuration,audit=audit,head=head,
                    prediction=head.predict(test,p['test_rows']),train=a,test=b)
                joblib.dump(result,target,compress=3)
            predictions[b,1]=result['prediction'];hits[b]+=1
            selected.append(result['configuration'])
            print('REACTION FOLD '+str(fold)+' '+json.dumps(result['configuration']),flush=True)
    assert np.all(hits==1) and np.array_equal(predictions[:,[0,2]],old['predicted'][:,[0,2]])
    baseline=evaluate(y,old['baseline']);candidate=evaluate(y,predictions);previous=evaluate(y,old['predicted'])
    report=dict(status='complete',rows=len(y),one_row_per_article=True,every_row_tested_once=True,
        shared_input_fields=list(INPUT_FIELDS),baseline=baseline,candidate=candidate,
        previous_models_same_rows=previous,
        gains_percent={m:100*(1-candidate[m]['mse']/baseline[m]['mse']) for m in baseline},
        change_vs_previous_mse_percent={m:100*(candidate[m]['mse']/previous[m]['mse']-1) for m in baseline},
        fold_configurations=selected,production_weights_modified=False,
        text_mse=float(np.mean((old['text']-data['labels'])**2)),
        protocol=f'Five identical outer text-group folds, same cached upstream OOF predictions; reaction-only {args.inner_folds}-fold training-only tuning; views/comments unchanged.',
        limitations=['Reused development corpus; not independent future-date confirmation.',
                     '1527 supplementary articles have a batch collection timestamp.'])
    write_json(args.output/'report.json',report)
    np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=predictions,
        baseline=old['baseline'],previous=old['predicted'],text=old['text'])
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('corpus','upstream','previous','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--inner-folds',type=int,default=3)
    main(p.parse_args())

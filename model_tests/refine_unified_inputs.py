"""Refine complete-feature distances, reusing only training-safe upstream scores."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.social_metrics_prediction.unified_models import fit_heads
from src.social_metrics_prediction.unified_features import INPUT_FIELDS
from .compare_social_metrics import evaluate
from .compare_text_models import write_json


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    data=joblib.load(args.previous/'corpus.joblib');x=data['features'];y=data['counts'];groups=data['groups']
    original=np.load(args.previous/'predictions.npz');prediction=np.full_like(y,np.nan);hits=np.zeros(len(y),int)
    with threadpool_limits(limits=4):
        for fold in range(1,6):
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            p=joblib.load(args.previous/f'fold_{fold}/models.joblib');a,b=p['train'],p['test']
            encoder,heads,selection=fit_heads(p['fit_rows'],y[a],x[a,:x.shape[1]//2],groups[a])
            matrix=encoder.transform(p['test_rows'],x[b,:x.shape[1]//2])
            prediction[b]=np.column_stack([head.predict(matrix,p['test_rows']) for head in heads]);hits[b]+=1
            joblib.dump(dict(encoder=encoder,heads=heads,selection=selection,train=a,test=b),path/'models.joblib',compress=3)
            print('COMPLETE SHARED INPUT FOLD '+str(fold)+' '+json.dumps(selection['configuration']),flush=True)
        assert np.all(hits==1)
        base=evaluate(y,original['baseline']);candidate=evaluate(y,prediction);prior=evaluate(y,original['previous'])
        gains={m:100*(1-candidate[m]['mse']/base[m]['mse']) for m in base}
        report=dict(status='complete',rows=len(y),one_row_per_article=True,every_row_tested_once=True,shared_input_fields=list(INPUT_FIELDS),
            baseline=base,candidate=candidate,previous_models_same_rows=prior,gains_percent=gains,
            change_vs_previous_mse_percent={m:100*(candidate[m]['mse']/prior[m]['mse']-1) for m in base},
            text_mse=float(np.mean((original['text']-data['labels'])**2)),production_weights_modified=False,
            protocol='Identical 11740 unique latest articles, same five outer text groups and same nested OOF text scores for previous and unified heads; inner two-fold tuning uses complete shared features with strictly positive context weights.',
            limitations=['Repeated development corpus; future-date validation remains necessary.','1527 older supplementary rows use a batch timestamp.'])
        write_json(args.output/'report.json',report);np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=prediction,baseline=original['baseline'],previous=original['previous'],text=original['text'])
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

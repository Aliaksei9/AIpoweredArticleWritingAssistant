"""Check the fixed seven-day prediction path on every unique corpus article.

This is an equivalence check against the previous seven-day default. Targets
were measured at different ages, so it deliberately does not claim day-7 MSE.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.social_metrics_prediction import SocialMetricsPredictor
from src.social_metrics_prediction.forecast_context import forecast_context


def main(args):
    data=joblib.load(args.corpus)
    scores=np.load(args.scores)['text']
    assert len(set(data['article_ids']))==len(scores)==len(data['rows'])
    rows=[dict(row,score=float(score),**forecast_context(row['published_at']))
          for row,score in zip(data['rows'],scores)]
    assert all(row['horizon_days']==7. for row in rows)
    vectors=data['features'][:,:data['features'].shape[1]//2]
    model=SocialMetricsPredictor.load(args.model)
    with threadpool_limits(limits=4):prediction=model.predict_rows(rows,document_embeddings=vectors)
    fingerprint=hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    reference=args.output/'reference.npz'
    if args.snapshot:
        assert not reference.exists(),'Use a new output directory for a new baseline'
        np.savez_compressed(reference,predicted=prediction,article_ids=data['article_ids'],model_sha256=fingerprint)
        print(json.dumps(dict(status='baseline_saved',rows=len(rows),horizon_days=7.)),flush=True)
        return
    before=np.load(reference)
    assert before['model_sha256'].item()==fingerprint
    np.testing.assert_array_equal(before['article_ids'],data['article_ids'])
    np.testing.assert_array_equal(before['predicted'],prediction)
    report=dict(status='passed',rows=len(rows),one_row_per_article=True,
        horizon_days=7.,all_three_predictions_identical=True,weights_unchanged=True,
        social_model_sha256=fingerprint,
        scope='Full-corpus equivalence to the previous seven-day default, not day-7 accuracy evaluation.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('corpus','scores','model','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--snapshot',action='store_true')
    main(p.parse_args())

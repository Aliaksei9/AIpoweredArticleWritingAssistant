"""Whole-corpus behavior snapshots for changes that must preserve predictions."""
import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from src.database.embedding_config import EMBEDDING_SIGNATURE
from src.database.time_utils import utc_datetime
from src.social_metrics_prediction import SocialMetricsPredictor
from src.social_metrics_prediction.dataset import metric_dataset_from_rows
from src.social_metrics_prediction.forecast_context import forecast_context
from src.social_metrics_prediction.reaction_selection import reaction_configurations


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    data=joblib.load(args.corpus);scores=np.load(args.scores)['text']
    assert len(set(data['article_ids']))==len(data['rows'])==len(scores)
    text=AssesmentConformPPtextBased.load(args.text)
    social=SocialMetricsPredictor.load(args.social)
    dimensions=data['features'].shape[1]//2
    vectors=data['features'][:,:dimensions]
    rows=[dict(row,score=float(score)) for row,score in zip(data['rows'],scores)]
    seven=[dict(row,**forecast_context(row['published_at'])) for row in rows]
    assessor=AssesmentConformPPmetricsBased()
    labels=np.asarray([assessor.assess(*counts,row['audience']) for row,counts in zip(rows,data['counts'])])
    raw=[]
    for i,row in enumerate(rows):
        raw.append(dict(row,article_id=data['article_ids'][i],text=data['texts'][i],
            score=float(labels[i]),views=data['counts'][i,0],reactions=data['counts'][i,1],comments=data['counts'][i,2],
            embedding=vectors[i],embedding_model=EMBEDDING_SIGNATURE,
            collected_at=utc_datetime(row['published_at'])+timedelta(days=row['horizon_days'])))
    dataset=metric_dataset_from_rows(raw,data['resources'])
    assert dataset.text.article_ids==list(map(str,data['article_ids']))
    np.testing.assert_array_equal(dataset.counts,data['counts'])
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        values=dict(metric_labels=labels,text_scores=text.model.predict(data['features']),
            social_observed=social.predict_rows(rows,document_embeddings=vectors),
            social_seven=social.predict_rows(seven,document_embeddings=vectors))
    metadata=dict(rows=len(rows),text_sha256=sha(args.text),social_sha256=sha(args.social),
        dataset_fingerprint=dataset.fingerprint,
        reaction_grid_sha256=hashlib.sha256(json.dumps(reaction_configurations(),sort_keys=True).encode()).hexdigest())
    args.output.mkdir(parents=True,exist_ok=True)
    reference=args.output/'reference.npz';meta=args.output/'reference.json'
    if args.snapshot:
        assert not reference.exists(),'Use a fresh directory for each refactoring'
        np.savez_compressed(reference,**values)
        meta.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(status='baseline_saved',**metadata)),flush=True)
        return
    assert metadata==json.loads(meta.read_text(encoding='utf8'))
    before=np.load(reference)
    for key,value in values.items():np.testing.assert_array_equal(before[key],value,err_msg=key)
    report=dict(status='passed',**metadata,checks=list(values),
        every_value_identical=True,one_row_per_article=True,
        scope='Behavior equivalence, not a new model-selection experiment.')
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('corpus','scores','text','social','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--snapshot',action='store_true')
    main(p.parse_args())

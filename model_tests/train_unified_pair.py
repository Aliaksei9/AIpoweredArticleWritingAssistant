"""Fit a staged source-only pair after the full unique-article validation."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from src.social_metrics_prediction import SocialMetricsPredictor
from src.social_metrics_prediction.training import cross_fitted_scores,train_text_head
from .compare_text_models import write_json


def main(args):
    validation=json.loads(args.validation.read_text(encoding='utf8'))
    if not validation['every_row_tested_once'] or not validation['one_row_per_article']:raise ValueError('Complete single-row validation required')
    data=joblib.load(args.corpus);x=data['features'];rows=data['rows'];labels=data['labels'];groups=data['groups']
    assert len(set(data['article_ids']))==len(x)==validation['rows']
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        scores,audit=cross_fitted_scores(x,labels,groups,folds=3,seed=47,on_progress=print,
            factory=lambda:AssesmentConformPPtextBased(strategy='additive_ensemble',num_threads=4))
        inputs=[dict(row,score=float(score)) for row,score in zip(rows,scores)]
        social=SocialMetricsPredictor().fit(inputs,data['counts'],document_embeddings=x[:,:x.shape[1]//2],groups=groups)
        social.set_resources(data['resources']);text=train_text_head(data['encoder'],x,labels,strategy='additive_ensemble')
    args.output.mkdir(parents=True,exist_ok=True)
    text.metadata.update(one_row_per_article=True,purpose='production candidate trained on all unique latest articles')
    text.save(args.output/'text.joblib')
    social.metadata.update(text_model_sha256=hashlib.sha256((args.output/'text.joblib').read_bytes()).hexdigest(),
        validation_report=str(args.validation),one_row_per_article=True)
    social.save(args.output/'social.joblib')
    write_json(args.output/'training.json',dict(rows=len(x),crossfit=audit,metadata=social.metadata))
    print('STAGED SINGLE-ROW PAIR '+json.dumps(social.metadata['selection']['configuration']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','validation','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())

"""Stage only a reaction head; preserve the current text, view and comment models."""
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
from src.social_metrics_prediction.training import cross_fitted_scores
from src.social_metrics_prediction.unified_models import UnifiedRateModel, UnifiedMetricHead
from .compare_text_models import write_json
from .refine_shared_reactions import select


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    data=joblib.load(args.corpus);x=data['features'];y=data['counts'];groups=data['groups']
    social=SocialMetricsPredictor.load(args.social)
    assert len(set(data['article_ids']))==len(y)==social.metadata['training_rows']
    assert social.metadata['text_model_sha256']==hashlib.sha256(args.text.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    cache=args.output/'training_scores.joblib'
    with threadpool_limits(limits=4):
        if cache.exists():scores,audit=joblib.load(cache)
        else:
            scores,audit=cross_fitted_scores(x,data['labels'],groups,folds=3,seed=47,
                on_progress=lambda line:print(line,flush=True),
                factory=lambda:AssesmentConformPPtextBased(strategy='additive_ensemble',num_threads=4))
            joblib.dump((scores,audit),cache,compress=3)
        if args.scores_only:
            print('Training-only OOF scores prepared',flush=True)
            return
        validation=json.loads(args.validation.read_text(encoding='utf8'))
        assert validation['every_row_tested_once'] and validation['one_row_per_article']
        assert validation['rows']==len(y) and validation['gains_percent']['reactions']>=1.
        rows=[dict(row,score=float(score)) for row,score in zip(data['rows'],scores)]
        vectors=x[:,:x.shape[1]//2]
        configuration,selection=select(rows,y,vectors,groups,
            on_progress=lambda line:print(line,flush=True))
        matrix=social.shared_encoder_.transform(rows,vectors)
        before=social.predict_rows(rows,document_embeddings=vectors)
        params=dict(configuration);strength=params.pop('strength')
        social.estimators_[1]=UnifiedMetricHead(UnifiedRateModel(**params).fit(matrix,rows,y),1,strength)
        after=social.predict_rows(rows,document_embeddings=vectors)
        np.testing.assert_array_equal(before[:,[0,2]],after[:,[0,2]])
        social.metadata['selection']['configuration']['reactions']=configuration
        social.metadata['selection']['reaction_refinement']=selection
        social.metadata['selection']['inner_mse']['reactions']=selection['inner_mse']
        social.metadata.update(validation_report=str(args.validation),
            reaction_validation_gain_percent=validation['gains_percent']['reactions'])
        social.save(args.output/'social.joblib')
        write_json(args.output/'training.json',dict(rows=len(y),configuration=configuration,
            audit=selection,unchanged_models=['text','views','comments'],
            text_sha256=social.metadata['text_model_sha256'],
            previous_social_sha256=hashlib.sha256(args.social.read_bytes()).hexdigest()))
        print('STAGED REACTIONS '+json.dumps(configuration),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('corpus','social','text','validation','output'):p.add_argument('--'+key,type=Path,required=key!='validation')
    p.add_argument('--scores-only',action='store_true')
    main(p.parse_args())

"""Latest row per article; identical social feature schema; whole-corpus nested check."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from src.social_metrics_prediction.training import cross_fitted_scores,train_text_head
from src.social_metrics_prediction.unified_models import fit_heads
from src.social_metrics_prediction.unified_features import INPUT_FIELDS
from .compare_social_metrics import evaluate
from .compare_text_models import write_json


def prepare(args):
    if args.corpus:
        data=joblib.load(args.corpus)
    else:
        import asyncio,os
        from dotenv import load_dotenv
        from src.social_metrics_prediction.dataset import load_metric_dataset
        from src.assesment_conform_of_PP_text_based import TextTopicEncoder
        load_dotenv('.env')
        data_set=asyncio.run(load_metric_dataset(os.environ['DATABASE_URL']),loop_factory=asyncio.SelectorEventLoop)
        encoder=TextTopicEncoder();torch.set_num_threads(4)
        with threadpool_limits(limits=4):features=encoder.transform(data_set.text.texts,data_set.text.topics,data_set.text.embeddings)
        rows=data_set.inputs(np.arange(len(features)),data_set.scores)
        for row in rows:row.pop('score',None)
        data=dict(rows=rows,features=features,labels=data_set.scores,counts=data_set.counts,groups=data_set.groups,
            article_ids=data_set.text.article_ids,texts=data_set.text.texts,resources=data_set.resources,encoder=encoder)
    assert len(set(data['article_ids']))==len(data['rows'])
    joblib.dump(data,args.output/'corpus.joblib',compress=3)
    write_json(args.output/'data_audit.json',dict(rows=len(data['rows']),one_row_per_article=True,
        selection='one current row per article',unique_groups=len(set(data['groups'])),resources=len(data['resources']),
        embedding_signature=data['encoder'].embedding_signature))


def run(args):
    data=joblib.load(args.output/'corpus.joblib');x=data['features'];y=data['counts'];labels=data['labels'];groups=data['groups'];rows=data['rows']
    prediction=np.full_like(y,np.nan);baseline=np.full_like(y,np.nan);old=np.full_like(y,np.nan);text=np.full(len(x),np.nan);hits=np.zeros(len(x),int)
    from src.social_metrics_prediction import SocialMetricsPredictor
    try:
        snapshot=SocialMetricsPredictor.load('artifacts/social_metrics/model.joblib')
        previous_configuration=snapshot.metadata.get('selection',{}).get('configuration') if snapshot.feature_mode=='unified' else None
    except (FileNotFoundError,ModuleNotFoundError,ValueError):previous_configuration=None
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for fold,(a,b) in enumerate(GroupKFold(5,shuffle=True,random_state=20261112).split(x,groups=groups),1):
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            if (path/'predictions.npz').exists():
                d=np.load(path/'predictions.npz');prediction[b]=d['predicted'];baseline[b]=d['baseline'];old[b]=d['old'];text[b]=d['text'];hits[b]+=1;continue
            scores,audit=cross_fitted_scores(x[a],labels[a],groups[a],folds=3,seed=47,on_progress=lambda s:print(f'fold {fold}: {s}',flush=True),
                factory=lambda:AssesmentConformPPtextBased(strategy='additive_ensemble',num_threads=4))
            nn=train_text_head(data['encoder'],x[a],labels[a],strategy='additive_ensemble');text[b]=nn.model.predict(x[b])
            fitting=[dict(rows[i],score=float(score)) for i,score in zip(a,scores)]
            validation=[dict(rows[i],score=float(score)) for i,score in zip(b,text[b])]
            encoder,heads,selection=fit_heads(fitting,y[a],x[a,:x.shape[1]//2],groups[a])
            matrix=encoder.transform(validation,x[b,:x.shape[1]//2])
            prediction[b]=np.column_stack([head.predict(matrix,validation) for head in heads]);hits[b]+=1
            means={};grouped={}
            for i,row in zip(a,fitting):grouped.setdefault((row['source'],row['resource_id']),[]).append(y[i])
            means={key:np.mean(values,axis=0) for key,values in grouped.items()};fallback=y[a].mean(axis=0)
            baseline[b]=[means.get((r['source'],r['resource_id']),fallback) for r in validation]
            if previous_configuration:
                old_encoder,old_heads,_=fit_heads(fitting,y[a],x[a,:x.shape[1]//2],groups[a],configuration=previous_configuration)
                old_matrix=old_encoder.transform(validation,x[b,:x.shape[1]//2])
                old[b]=np.column_stack([h.predict(old_matrix,validation) for h in old_heads])
            np.savez_compressed(path/'predictions.npz',predicted=prediction[b],baseline=baseline[b],old=old[b],text=text[b])
            joblib.dump(dict(train=a,test=b,fit_rows=fitting,test_rows=validation,encoder=encoder,heads=heads,selection=selection,audit=audit),path/'models.joblib',compress=3)
            nn.save(path/'text.joblib');print('UNIFIED FOLD '+str(fold)+' '+json.dumps(selection['configuration']),flush=True)
    assert np.all(hits==1) and np.isfinite(prediction).all()
    base=evaluate(y,baseline);candidate=evaluate(y,prediction);prior=evaluate(y,old) if np.isfinite(old).all() else None
    gains={m:100*(1-candidate[m]['mse']/base[m]['mse']) for m in base}
    report=dict(status='complete',rows=len(x),one_row_per_article=True,every_row_tested_once=True,
        shared_input_fields=list(INPUT_FIELDS),baseline=base,candidate=candidate,previous_models_same_rows=prior,
        gains_percent=gains,change_vs_previous_mse_percent=({m:100*(candidate[m]['mse']/prior[m]['mse']-1) for m in base} if prior else None),
        text_mse=float(np.mean((text-labels)**2)),production_weights_modified=False,
        protocol='5 outer article/text-group folds, 3 upstream OOF folds, 2 tuning folds; all heads receive one identical complete feature matrix; exactly one latest row per article.',
        limitations=['Historical corpus reused; new future-date confirmation is required.',
            '1527 older supplementary articles have a batch reference timestamp.'])
    np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=prediction,baseline=baseline,previous=old,text=text)
    write_json(args.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path);p.add_argument('--database-only',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not (a.output/'corpus.joblib').exists():prepare(a)
    if not (a.output/'report.json').exists():run(a)

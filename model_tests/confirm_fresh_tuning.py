"""One-shot confirmation on chronologically newer posts of unused channels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np

from .compare_social_metrics import evaluate,normalized_mse
from .compare_text_models import write_json
from .social_metric_candidates import ResourceMeanBaseline
from .tuning_candidates import ResidualRatePredictor,SOCIAL_CONFIGS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=Path('model_tests/results/expansion_20261111'))
    args=parser.parse_args()
    if (args.directory/'fresh_confirmation.json').exists():raise ValueError('Fresh test has already been evaluated')
    frozen=json.loads((args.directory/'fresh_choices_frozen.json').read_text(encoding='utf8'))
    snapshot=json.loads((args.directory/'fresh_snapshot.json').read_text(encoding='utf8'))
    screening=json.loads((args.directory/'screening.json').read_text(encoding='utf8'))
    original=json.loads((args.directory/'confirmation/nested_inputs.json').read_text(encoding='utf8'))
    splits=json.loads((args.previous/'social_metrics/splits.json').read_text(encoding='utf8'))
    with np.load(args.previous/'social_metrics/dataset.npz') as saved:old_counts=saved['counts'];old_groups=set(saved['groups'])
    articles=snapshot['articles']
    totals=Counter(r['resource_id'] for r in articles)
    articles=[r for r in articles if totals[r['resource_id']]>=10]
    assert articles and not old_groups.intersection(r['group'] for r in articles)
    assert len({r['group'] for r in articles})==len(articles)
    fitting=[];test=[]
    for resource in sorted({r['resource_id'] for r in articles}):
        indices=sorted([i for i,r in enumerate(articles) if r['resource_id']==resource],key=lambda i:articles[i]['published_at'])
        cut=int(len(indices)*.8)
        fitting.extend(indices[:cut]);test.extend(indices[cut:])
    assert not set(fitting)&set(test)
    assert {articles[i]['resource_id'] for i in test}.issubset({articles[i]['resource_id'] for i in fitting})
    old_rows=[None]*len(old_counts)
    for indices,rows in [(splits['train_indices'],original['heldout']['train']),(splits['test_indices'],original['heldout']['test'])]:
        for index,row in zip(indices,rows):old_rows[index]=row
    def inputs(indices):
        return [{k:articles[i][k] for k in ['source','resource_id','topic','audience','score']} for i in indices]
    train_rows=old_rows+inputs(fitting)
    train_counts=np.vstack([old_counts,np.asarray([articles[i]['counts'] for i in fitting])])
    validation=inputs(test);actual=np.asarray([articles[i]['counts'] for i in test])
    selected=frozen['selected']
    chosen=screening['social'][selected]
    configuration=dict(SOCIAL_CONFIGS[chosen['config']],blend=chosen['blend'])
    write_json(args.directory/'fresh_split_frozen.json',dict(configuration=configuration,
        train=fitting,test=test,protocol='oldest 80% per channel for training, newest 20% for test; channels with at least 10 distinct new texts'))
    model=ResidualRatePredictor(**configuration).fit(train_rows,train_counts)
    baseline=ResourceMeanBaseline().fit(train_rows,train_counts)
    prediction=model.predict_rows(validation);reference=baseline.predict_rows(validation)
    # Error normalization is fitted on the new-channel training subset so the
    # variance of the older reaction outlier cannot dominate this new study.
    scale_counts=np.asarray([articles[i]['counts'] for i in fitting])
    score=normalized_mse(actual,prediction,scale_counts)
    base_score=normalized_mse(actual,reference,scale_counts)
    variance=np.maximum(np.var(scale_counts,axis=0),1.)
    differences=((actual-reference)**2-(actual-prediction)**2)/variance
    resources=[articles[i]['resource_id'] for i in test]
    unique=sorted(set(resources))
    blocks=[np.asarray([i for i,key in enumerate(resources) if key==resource]) for resource in unique]
    rng=np.random.default_rng(20261111)
    boot=[]
    for _ in range(2000):
        indices=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),size=len(blocks))])
        boot.append(differences[indices].mean())
    report=dict(status='complete',selected=selected,configuration=configuration,
        fresh_articles=len(articles),fresh_training_rows=len(fitting),fresh_test_rows=len(test),
        fresh_channels=len(unique),total_training_rows=len(train_rows),old_training_rows=len(old_counts),
        protocol='configuration frozen before collection; within each unused channel, oldest 80% train / newest 20% test; exact duplicates and previously seen text groups excluded',
        metric_normalization='per-target variance of fresh training rows; same scale for model and baseline',
        text_model='frozen evaluation text head trained on 8143 old articles; text and topic only',
        model=dict(normalized_mse=score,metrics=evaluate(actual,prediction)),
        channel_mean=dict(normalized_mse=base_score,metrics=evaluate(actual,reference)),
        improvement_percent=(1-score/base_score)*100,improves=bool(score<base_score),
        paired_channel_bootstrap=dict(improvement_mean=float(differences.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),replicates=2000),
        limitations=['Same collection age window; not a test collected on a later date',
                     'Test channels have their own older training posts, so this is not zero-shot resource prediction'])
    joblib.dump(model,args.directory/'fresh_social_evaluation.joblib',compress=3)
    write_json(args.directory/'fresh_training_inputs.json',dict(rows=train_rows,counts=train_counts.tolist(),validation=validation))
    np.savez_compressed(args.directory/'fresh_test_predictions.npz',actual=actual,model=prediction,channel_mean=reference)
    write_json(args.directory/'fresh_confirmation.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

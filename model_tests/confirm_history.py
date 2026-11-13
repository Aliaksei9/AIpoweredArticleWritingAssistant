"""Compare the frozen candidate, channel mean and current profile on new posts."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .compare_social_metrics import evaluate,normalized_mse
from .compare_text_models import write_json
from .history_candidates import ConditionalHistoryPredictor
from .tuning_candidates import ResidualRatePredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    output=args.directory
    if (output/'confirmation.json').exists():raise ValueError('Control already evaluated')
    data=json.loads((output/'development.json').read_text(encoding='utf8'))
    snapshot=json.loads((output/'control_snapshot.json').read_text(encoding='utf8'))
    frozen=json.loads((output/'choices_frozen.json').read_text(encoding='utf8'))
    assert snapshot['status']=='complete'
    articles=snapshot['articles']
    assert articles and not set(data['groups']).intersection(r['group'] for r in articles)
    assert {r['resource_id'] for r in articles}.issubset({r['resource_id'] for r in data['rows']})
    counts=np.asarray(data['counts']);rows=data['rows']
    inputs=[{k:r[k] for k in ['source','resource_id','topic','audience','score']} for r in articles]
    actual=np.asarray([r['counts'] for r in articles])
    models=dict(channel_mean=ConditionalHistoryPredictor(),
        previous_profile=ResidualRatePredictor(family='catboost',epochs=150,alpha=20,blend=.25),
        candidate=joblib.load(output/'selected_evaluation.joblib'))
    predictions={};results={}
    for name,model in models.items():
        if name!='candidate':model.fit(rows,counts)
        prediction=model.predict_rows(inputs)
        predictions[name]=prediction
        results[name]=dict(normalized_mse=normalized_mse(actual,prediction,counts),metrics=evaluate(actual,prediction))
    variance=np.maximum(np.var(counts,axis=0),1.)
    difference=((actual-predictions['channel_mean'])**2-(actual-predictions['candidate'])**2)/variance
    resources=[r['resource_id'] for r in articles]
    unique=sorted(set(resources))
    blocks=[np.asarray([i for i,key in enumerate(resources) if key==resource]) for resource in unique]
    rng=np.random.default_rng(20261112)
    bootstrap=[]
    for _ in range(3000):
        indices=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),size=len(blocks))])
        bootstrap.append(difference[indices].mean())
    report=dict(status='complete',selected=frozen['selected'],configuration=frozen['configuration'],
        training_rows=len(rows),control_rows=len(articles),control_channels=len(unique),results=results,
        improvement_percent=(1-results['candidate']['normalized_mse']/results['channel_mean']['normalized_mse'])*100,
        improvement_vs_previous_profile_percent=(1-results['candidate']['normalized_mse']/results['previous_profile']['normalized_mse'])*100,
        channel_bootstrap_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
        protocol='candidate frozen before collection; new publication interval later than all development posts; exact normalized repeats excluded; known channels',
        normalization='per-target variance of all development training counts; shared by all compared models',
        limitations=['New articles are around two days old; development ages span 2-14 days',
                     'Metric counters are observational; no guarantee for individual future publications'])
    np.savez_compressed(output/'control_predictions.npz',actual=actual,**predictions)
    write_json(output/'confirmation.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

"""Confirm tuning with nested upstream selection and a frozen historical test.

Text and topic remain the only text-model inputs. Resource IDs are available
only to the social regressor and the diagnostic channel-mean baselines.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from .benchmark_utils import metrics
from .compare_social_metrics import evaluate,normalized_mse
from .compare_text_models import write_json
from .social_metric_candidates import ResourceMeanBaseline
from .tuning_candidates import TEXT_CONFIGS,SOCIAL_CONFIGS,ResidualRatePredictor


class NestedTextSelection:
    def __init__(self, features, labels, groups, allowed, output):
        self.features,self.labels,self.groups=features,labels,groups
        self.allowed=set(allowed)
        self.output=output
        identity=hashlib.sha256(features.tobytes()+labels.tobytes())
        identity.update(json.dumps(groups.tolist()).encode('utf8'))
        identity.update(json.dumps(TEXT_CONFIGS,sort_keys=True).encode('utf8'))
        self.cache_identity=identity.digest()
        output.mkdir(parents=True,exist_ok=True)

    def fit(self, indices):
        indices=np.asarray(indices,dtype=int)
        assert set(indices).issubset(self.allowed)
        digest=hashlib.sha256(self.cache_identity+indices.tobytes()).hexdigest()[:20]
        record_path=self.output/(digest+'.json')
        model_path=self.output/(digest+'.joblib')
        if record_path.exists() and model_path.exists():
            record=json.loads(record_path.read_text(encoding='utf8'))
            assert record['training_indices']==indices.tolist()
            return joblib.load(model_path),record
        splitter=GroupKFold(n_splits=2,shuffle=True,random_state=42)
        results={}
        for name,parameters in TEXT_CONFIGS.items():
            errors=[]
            for train,validation in splitter.split(indices,groups=self.groups[indices]):
                a,b=indices[train],indices[validation]
                assert not set(self.groups[a]) & set(self.groups[b])
                model=AssesmentConformPPtextBased(seed=42,num_threads=4,**parameters)
                model.train_features(self.features[a],self.labels[a])
                errors.append(metrics(self.labels[b],model.model.predict(self.features[b]))['mse'])
            results[name]=dict(mean=float(np.mean(errors)),folds=errors)
        chosen=min(results,key=lambda name:results[name]['mean'])
        model=AssesmentConformPPtextBased(seed=42,num_threads=4,**TEXT_CONFIGS[chosen])
        model.train_features(self.features[indices],self.labels[indices])
        record=dict(training_indices=indices.tolist(),chosen=chosen,cv=results)
        joblib.dump(model,model_path,compress=3)
        write_json(record_path,record)
        print(f'Nested text selection: rows={len(indices)}, chosen={chosen}, CV MSE={results[chosen]["mean"]:.4f}',flush=True)
        return model,record

    def inputs(self, fitting, validation, rows):
        scores=np.full(len(fitting),np.nan)
        audit=[]
        splitter=GroupKFold(n_splits=3,shuffle=True,random_state=42)
        for a,b in splitter.split(fitting,groups=self.groups[fitting]):
            train,holdout=fitting[a],fitting[b]
            model,record=self.fit(train)
            scores[b]=model.model.predict(self.features[holdout])
            audit.append(dict(train=train.tolist(),validation=holdout.tolist(),configuration=record['chosen']))
        assert np.isfinite(scores).all()
        head,record=self.fit(fitting)
        predicted=head.model.predict(self.features[validation])
        def with_scores(indices,values):
            return [dict(rows[int(i)],score=float(score)) for i,score in zip(indices,values)]
        return dict(train=with_scores(fitting,scores),validation=with_scores(validation,predicted),
                    fitting=fitting,validation_indices=validation,head=head,
                    head_configuration=record['chosen'],text_predictions=predicted,audit=audit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=Path('model_tests/results/expansion_20261111'))
    parser.add_argument('--screening',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'confirmation.json').exists():
        raise ValueError('Use a new output directory')
    screening=json.loads(args.screening.read_text(encoding='utf8'))
    assert screening['status']=='complete'
    # Freeze a small, declared shortlist before confirmation. Keep the best of
    # each family plus the original direct MLP for comparison.
    shortlisted=[]
    for family in ['ridge','mlp','boosting','catboost']:
        available=[(k,v) for k,v in screening['social'].items() if SOCIAL_CONFIGS[v['config']]['family']==family]
        shortlisted.append(min(available,key=lambda pair:pair[1]['cv_mse'])[0])
    with np.load(args.previous/'social_metrics/dataset.npz') as saved:
        features=saved['text_features']; labels=saved['metric_scores']; groups=saved['groups']; counts=saved['counts']
    splits=json.loads((args.previous/'social_metrics/splits.json').read_text(encoding='utf8'))
    original=json.loads((args.previous/'social_metrics/inputs.json').read_text(encoding='utf8'))
    train=np.asarray(splits['train_indices']); test=np.asarray(splits['test_indices'])
    rows=[None]*len(labels)
    for indices,values in [(train,original['heldout']['train']),(test,original['heldout']['test'])]:
        for index,row in zip(indices,values): rows[int(index)]={k:v for k,v in row.items() if k!='score'}
    selection=NestedTextSelection(features,labels,groups,train,args.output/'nested_text_models')
    torch.set_num_threads(4)
    report=dict(status='preparing_nested_scores',protocol='nested two-fold text configuration selection; three-fold group cross-fitting',
        text_inputs=['text','topic'],historical_test_reused=True,
        target_formula_version='log-activity-v3',shortlist=shortlisted,results={},text_folds=[])
    write_json(args.output/'progress.json',report)
    with threadpool_limits(limits=4):
        folds=[]
        for i,fold in enumerate(splits['cv'],1):
            print(f'Preparing nested social CV {i}/3',flush=True)
            a,b=np.asarray(fold['train_indices']),np.asarray(fold['validation_indices'])
            prepared=selection.inputs(a,b,rows)
            folds.append(prepared)
            report['text_folds'].append(dict(configuration=prepared['head_configuration'],
                metrics=metrics(labels[b],prepared['text_predictions'])))
            write_json(args.output/'progress.json',report)
        from .social_metric_candidates import CandidateSocialMetricsPredictor
        candidates=['resource_mean','original_direct_mlp',*shortlisted]
        for name in candidates:
            errors=[]; per_metric=[]
            for fold in folds:
                if name=='resource_mean':model=ResourceMeanBaseline()
                elif name=='original_direct_mlp':model=CandidateSocialMetricsPredictor('neural_network',seed=42,threads=4)
                else:
                    entry=screening['social'][name]
                    model=ResidualRatePredictor(**SOCIAL_CONFIGS[entry['config']],blend=entry['blend'])
                a,b=fold['fitting'],fold['validation_indices']
                model.fit(fold['train'],counts[a])
                predicted=model.predict_rows(fold['validation'])
                errors.append(normalized_mse(counts[b],predicted,counts[a]))
                per_metric.append(evaluate(counts[b],predicted))
            report['results'][name]=dict(cv_mse=float(np.mean(errors)),cv_folds=errors,per_metric=per_metric)
            write_json(args.output/'progress.json',report)
            print('CONFIRM CV',name,json.dumps(report['results'][name]['cv_folds']),flush=True)
        # Choices are written to disk before examining any held-out errors.
        chosen=min(candidates,key=lambda n:report['results'][n]['cv_mse'])
        final=selection.inputs(train,test,rows)
        report.update(status='choices_frozen',social_selected=chosen,text_configuration=final['head_configuration'])
        write_json(args.output/'choices_frozen.json',report)
        text_head=final['head']
        joblib.dump(text_head,args.output/'text_evaluation.joblib',compress=3)
        text_errors=metrics(labels[test],final['text_predictions'])
        channel_scores={}
        for key in {(r['source'],r['resource_id']) for r in final['train']}:
            channel_scores[key]=np.mean([labels[i] for i in train if (rows[i]['source'],rows[i]['resource_id'])==key])
        text_baseline=np.asarray([channel_scores.get((rows[i]['source'],rows[i]['resource_id']),labels[train].mean()) for i in test])
        report['text_test']=dict(model=text_errors,channel_mean=metrics(labels[test],text_baseline))
        predictions={}
        for name in candidates:
            if name=='resource_mean':model=ResourceMeanBaseline()
            elif name=='original_direct_mlp':model=CandidateSocialMetricsPredictor('neural_network',seed=42,threads=4)
            else:
                entry=screening['social'][name]
                model=ResidualRatePredictor(**SOCIAL_CONFIGS[entry['config']],blend=entry['blend'])
            model.fit(final['train'],counts[train])
            predicted=model.predict_rows(final['validation'])
            predictions[name]=predicted
            report['results'][name].update(test=evaluate(counts[test],predicted),
                test_normalized_mse=normalized_mse(counts[test],predicted,counts[train]))
            if name==chosen:joblib.dump(model,args.output/'social_evaluation.joblib',compress=3)
        np.savez_compressed(args.output/'test_predictions.npz',actual=counts[test],**predictions)
        write_json(args.output/'nested_inputs.json',dict(cv=[dict(train=f['train'],validation=f['validation'],audit=f['audit']) for f in folds],
            heldout=dict(train=final['train'],test=final['validation'],audit=final['audit'])))
        report['social_improves_cv']=report['results'][chosen]['cv_mse']<report['results']['resource_mean']['cv_mse']
        report['social_improves_test']=report['results'][chosen]['test_normalized_mse']<report['results']['resource_mean']['test_normalized_mse']
        report['text_improves_channel_mean']=text_errors['mse']<report['text_test']['channel_mean']['mse']
        report['status']='complete'
        write_json(args.output/'confirmation.json',report)
        print(json.dumps(dict(status='complete',social_selected=chosen,text_configuration=final['head_configuration'],
            social_improves_cv=report['social_improves_cv'],social_improves_test=report['social_improves_test'],text_test=report['text_test'])),flush=True)


if __name__=='__main__':
    main()

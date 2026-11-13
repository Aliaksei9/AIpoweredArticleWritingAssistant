"""Nested complete-corpus architecture search, with immutable production weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import torch

from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from src.database.embedding_config import EMBEDDING_SIGNATURE
from .advanced_text_networks import AdvancedTextRegressor,CONFIGURATIONS,TextPredictorBundle
from .compare_text_models import write_json
from .evaluate_text_ensembles import CONFIGURATIONS as PREVIOUS_CONFIGS


REFERENCE_CONFIG=PREVIOUS_CONFIGS['wide_regularized']
REFERENCE_SEEDS=(42,73)


def fingerprint(corpus_hash, indices, configuration, seed):
    digest=hashlib.sha256(corpus_hash.encode())
    digest.update(np.asarray(indices,dtype=np.int64).tobytes())
    digest.update(json.dumps(dict(configuration=configuration,seed=seed),sort_keys=True).encode())
    return digest.hexdigest()


def fitted_model(path, x, y, indices, configuration, corpus_hash, seed=42, reference=False):
    expected=fingerprint(corpus_hash,indices,configuration,seed)
    if path.exists():
        saved=joblib.load(path)
        if saved['fingerprint']!=expected:raise ValueError('Cached model provenance mismatch: '+str(path))
        return saved['model']
    if reference:
        model=AssesmentConformPPtextBased(seed=seed,num_threads=4,**configuration).train_features(x[indices],y[indices])
    else:model=AdvancedTextRegressor(configuration,seed=seed,threads=4).fit(x[indices],y[indices])
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(fingerprint=expected,model=model),path,compress=3)
    return model


def fit_selection(selection, x, y, indices, path, corpus_hash, reference_models=None):
    """Refit a frozen inner choice; validation labels are never accepted here."""
    models=[];weights=[]
    use_reference=selection=='reference' or selection.endswith(':blend')
    if use_reference:
        refs=reference_models
        if refs is None:
            refs=[fitted_model(path/f'reference_{seed}.joblib',x,y,indices,REFERENCE_CONFIG,
                                corpus_hash,seed=seed,reference=True) for seed in REFERENCE_SEEDS]
        models.extend(refs);weights.extend([.5 if selection=='reference' else .25]*2)
    if selection!='reference':
        name=selection.split(':')[0]
        model=fitted_model(path/f'{name}.joblib',x,y,indices,CONFIGURATIONS[name],corpus_hash)
        models.append(model);weights.append(.5 if use_reference else 1.)
    return TextPredictorBundle(models,weights)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Completed study already exists')
    audit=json.loads((args.corpus/'audit.json').read_text(encoding='utf8'))
    assert audit['fixed_embedding_model']==EMBEDDING_SIGNATURE
    corpus_hash=hashlib.sha256((args.corpus/'corpus.npz').read_bytes()).hexdigest()
    data=np.load(args.corpus/'corpus.npz');x=data['features'];y=data['labels'];groups=data['groups']
    old=np.load(args.previous/'predictions.npz');previous_report=json.loads((args.previous/'report.json').read_text(encoding='utf8'))
    np.testing.assert_array_equal(old['actual'],y)
    prediction=np.full(len(y),np.nan);reference_prediction=np.full(len(y),np.nan);hits=np.zeros(len(y),int)
    report=dict(rows=len(y),groups=len(set(groups)),configurations=CONFIGURATIONS,folds=[],
        corpus_sha256=corpus_hash,embedding_signature=EMBEDDING_SIGNATURE,
        configuration_count=1+2*len(CONFIGURATIONS),prediction_inputs=['text','topic'],
        protocol='Five frozen outer exact-text-group folds; two inner folds choose architecture or a blend with previous neural ensemble',
        production_weights_modified=False)
    started=time.perf_counter();torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for fold,(train,test) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=20261111).split(x,groups=groups),1):
            assert not set(groups[train])&set(groups[test])
            path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
            inner=list(GroupKFold(n_splits=2,shuffle=True,random_state=91).split(train,groups=groups[train]))
            reference=[];reference_error=0.;n=0
            assert previous_report['folds'][fold-1]['selected']=='wide_regularized:2'
            for j,(a,b) in enumerate(inner,1):
                refs=[joblib.load(args.previous/f'fold_{fold}'/f'inner_{j}_wide_regularized_{seed}.joblib') for seed in REFERENCE_SEEDS]
                p=TextPredictorBundle(refs).predict(x[train[b]])
                reference.append(p);reference_error+=float(np.sum((p-y[train[b]])**2));n+=len(b)
            np.testing.assert_allclose(reference_error/n,previous_report['folds'][fold-1]['inner_mse']['wide_regularized:2'],rtol=1e-8)
            scores={'reference':reference_error/n}
            for name,config in CONFIGURATIONS.items():
                sse=0.;blended_sse=0.
                for j,(a,b) in enumerate(inner,1):
                    model=fitted_model(path/f'inner_{j}_{name}.joblib',x,y,train[a],config,corpus_hash)
                    p=model.predict(x[train[b]])
                    sse+=float(np.sum((p-y[train[b]])**2))
                    blended_sse+=float(np.sum(((p+reference[j-1])/2-y[train[b]])**2))
                scores[name]=sse/n;scores[name+':blend']=blended_sse/n
                print(f'fold {fold}/5 {name}: MSE {scores[name]:.5f}, blend {scores[name+":blend"]:.5f}',flush=True)
            selected=min(scores,key=scores.get)
            refs=[joblib.load(args.previous/f'fold_{fold}'/f'selected_wide_regularized_{seed}.joblib') for seed in REFERENCE_SEEDS]
            reference_prediction[test]=TextPredictorBundle(refs).predict(x[test])
            np.testing.assert_allclose(reference_prediction[test],old['predicted'][test],rtol=1e-7,atol=1e-7)
            final=fit_selection(selected,x,y,train,path/'outer_models',corpus_hash,reference_models=refs)
            prediction[test]=final.predict(x[test]);hits[test]+=1
            joblib.dump(final,path/'predictor.joblib',compress=3)
            times=[];final.predict(x[test[:1]])
            for _ in range(30):
                t=time.perf_counter();final.predict(x[test[:1]]);times.append(time.perf_counter()-t)
            entry=dict(fold=fold,selected=selected,inner_mse=scores,train_rows=len(train),test_rows=len(test),
                outer_mse=float(np.mean((prediction[test]-y[test])**2)),
                reference_mse=float(np.mean((reference_prediction[test]-y[test])**2)),
                warm_head_max_seconds=max(times),warm_head_mean_seconds=float(np.mean(times)))
            write_json(path/'selection.json',entry);report['folds'].append(entry)
            write_json(args.output/'progress.json',report)
            print(f'fold {fold}/5 selected {selected}, outer MSE {entry["outer_mse"]:.5f}',flush=True)
    assert np.all(hits==1) and np.isfinite(prediction).all()
    mse=float(np.mean((prediction-y)**2));reference_mse=float(np.mean((reference_prediction-y)**2))
    baseline=json.loads((args.corpus/'text_baseline.json').read_text(encoding='utf8'))['mse']
    report.update(status='complete',mse=mse,previous_mse=reference_mse,baseline_mse=baseline,
        reduction_vs_previous_percent=100*(1-mse/reference_mse),reduction_vs_baseline_percent=100*(1-mse/baseline),
        every_row_tested_once=True,elapsed_seconds=time.perf_counter()-started,
        limitations=['The historical corpus has already been used for research; this is not a new independent future-time holdout',
                     'Text-only MSE does not prove the social models reach their separate five-percent requirement'])
    np.savez_compressed(args.output/'predictions.npz',actual=y,predicted=prediction,previous=reference_prediction)
    write_json(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ['mse','reduction_vs_previous_percent','reduction_vs_baseline_percent']}),flush=True)


if __name__=='__main__':main()

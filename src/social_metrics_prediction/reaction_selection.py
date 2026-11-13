"""Training-only selection for the independent reaction head, with shared inputs."""
import itertools

import numpy as np
from sklearn.model_selection import GroupKFold

from .unified_features import UnifiedFeatureEncoder
from .unified_models import UnifiedRateModel

CONTEXT_WEIGHTS=(.003,.01,.03,.1)
QUANTILES=(.75,.85,.9,.95,.975,1.)
NEIGHBOUR_COUNTS=(3,5,10,20,40)
TEMPERATURES=(.03,.06,.25,1.,4.)
STRENGTHS=(.25,.5,.75,1.,1.25)


def reaction_configurations():
    """All dimensions retain positive weights; only reaction hyperparameters vary."""
    return [dict(family='neighbours', same_resource=True, context_weight=context,
                 quantile=quantile, neighbours=k, temperature=temp, strength=strength)
            for context, quantile, k, temp, strength in itertools.product(
                CONTEXT_WEIGHTS, QUANTILES, NEIGHBOUR_COUNTS, TEMPERATURES, STRENGTHS)]


def select_reaction_configuration(rows, counts, vectors, groups, *, folds=3, on_progress=None):
    candidates=reaction_configurations(); errors=np.zeros(len(candidates)); hits=np.zeros(len(rows),int)
    # Sorting groups makes all tuning subsets reproducible.
    for fold,(a,b) in enumerate(GroupKFold(folds,shuffle=True,random_state=47).split(counts,groups=groups),1):
        train=[rows[i] for i in a];valid=[rows[i] for i in b]
        encoder=UnifiedFeatureEncoder().fit(train,vectors[a])
        xa=encoder.transform(train,vectors[a]);xb=encoder.transform(valid,vectors[b])
        position=0
        for context in CONTEXT_WEIGHTS:
            nearest=None
            for quantile in QUANTILES:
                model=UnifiedRateModel(family='neighbours',same_resource=True,
                    context_weight=context,quantile=quantile).fit(xa,train,counts[a])
                if nearest is None:nearest=model.nearest(xb,max(NEIGHBOUR_COUNTS),valid)
                prior=model.prior.predict(valid)[:,1]
                for k in NEIGHBOUR_COUNTS:
                    for temp in TEMPERATURES:
                        model.neighbours=k;model.temperature=temp
                        raw=model.raw(xb,valid,nearest)[:,1]
                        for strength in STRENGTHS:
                            prediction=np.rint(np.maximum(0,prior+strength*(raw-prior)))
                            errors[position]+=np.sum((prediction-counts[b,1])**2)
                            position+=1
        assert position==len(candidates)
        hits[b]+=1
        if on_progress:on_progress(f'Inner fold {fold}/{folds} complete')
    assert np.all(hits==1)
    best=int(np.argmin(errors))
    return candidates[best],dict(inner_mse=float(errors[best]/len(rows)),
        tuning_folds=folds,candidates=len(candidates),
        top=[dict(configuration=candidates[int(i)],mse=float(errors[i]/len(rows)))
             for i in np.argsort(errors)[:10]])

"""Experimental positive-link heads with training-only channel calibration."""
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.social_metrics_prediction.features import MetricFeatureEncoder, validate_counts
from .history_candidates import ConditionalHistoryPredictor


class CalibratedCountLink:
    """Predict a multiplicative departure from a channel's arithmetic mean.

    Neither channel priors nor smearing constants use validation labels.
    ``weight_power`` controls the training loss only; evaluation always uses
    the original, uncapped counts. Each instance fits exactly one target.
    """
    def __init__(self, metric, alpha=1000., weight_power=0., strength=1.):
        self.metric=metric; self.alpha=alpha
        self.weight_power=weight_power; self.strength=strength

    @staticmethod
    def _key(row):
        return row['source'],row['resource_id']

    def _features(self, rows):
        x=self.encoder.transform(rows)
        categories=x[:,:-2]
        score=np.asarray([r['score'] for r in rows])/10.
        z=np.arctanh(np.clip(score,-.98,.98))
        center=np.asarray([self.centers.get(self._key(r),self.global_center) for r in rows])
        delta=z-center
        basis=np.column_stack([delta,delta**2,np.tanh(delta),z])
        return np.column_stack([x,basis,categories*delta[:,None],categories*(delta**2)[:,None]])

    def _priors(self, rows):
        rates=np.asarray([self.base.channels.get(self._key(r),dict(mean=self.base.global_mean))['mean'][self.metric]
                          for r in rows])
        return rates*np.asarray([r['audience'] for r in rows])/1000.

    def fit(self, rows, counts):
        values=validate_counts(counts,len(rows))[:,self.metric]
        self.encoder=MetricFeatureEncoder().fit(rows)
        self.base=ConditionalHistoryPredictor().fit(rows,counts)
        groups=defaultdict(list)
        for r in rows: groups[self._key(r)].append(float(np.arctanh(np.clip(r['score']/10.,-.98,.98))))
        self.centers={k:float(np.mean(v)) for k,v in groups.items()}
        self.global_center=float(np.mean([v for vs in groups.values() for v in vs]))
        prior=self._priors(rows)
        target=np.log((values+.5)/(prior+.5))
        weight=np.maximum(prior,1.)**self.weight_power
        weight/=np.mean(weight)
        self.scaler=StandardScaler().fit(self._features(rows))
        x=self.scaler.transform(self._features(rows))
        self.model=Ridge(alpha=self.alpha).fit(x,target,sample_weight=weight)
        factors=np.exp(np.clip(self.model.predict(x),-6.,6.))
        grouped=defaultdict(list)
        for r,f in zip(rows,factors): grouped[self._key(r)].append(float(f))
        self.calibration={k:float(np.mean(v)) for k,v in grouped.items()}
        self.global_calibration=float(np.mean(factors))
        return self

    def predict(self, rows):
        x=self.scaler.transform(self._features(rows))
        factors=np.exp(np.clip(self.model.predict(x),-6.,6.))
        factors/=np.asarray([self.calibration.get(self._key(r),self.global_calibration) for r in rows])
        prediction=self._priors(rows)*(1.+self.strength*(factors-1.))
        return np.rint(np.maximum(0.,prediction))


def configurations():
    return {f'link_a{alpha:g}_w{weight:g}_s{strength:g}':
            dict(alpha=alpha,weight_power=weight,strength=strength)
            for alpha in (10.,1000.,100000.)
            for weight in (0.,1.,2.)
            for strength in (.25,.5,1.)}

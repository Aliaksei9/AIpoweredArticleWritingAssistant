"""Experimental calendar/horizon inputs; no inference labels are required."""
from datetime import datetime,timedelta

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.social_metrics_prediction.features import MetricFeatureEncoder
from .history_candidates import ConditionalHistoryPredictor


class CalendarCountRegressor:
    def __init__(self, alpha):self.alpha=alpha

    def _prior(self, rows):
        return np.asarray([self.base.channels.get(self.base._key(r),dict(mean=self.base.global_mean))['mean'] for r in rows])*np.asarray([r['audience'] for r in rows])[:,None]/1000.

    def _features(self, rows):
        x=self.encoder.transform(rows);calendar=[]
        for r in rows:
            date=datetime.fromisoformat(r['published_at'])+timedelta(hours=3)
            hour=date.hour+date.minute/60.;day=date.weekday();age=float(r['horizon_days'])/14.
            calendar.append([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                np.sin(2*np.pi*day/7),np.cos(2*np.pi*day/7),age,age**2,age**3])
        c=np.asarray(calendar)
        return np.column_stack([x,c,(x[:,:-2,None]*c[:,None,:]).reshape(len(rows),-1)])

    def fit(self, rows, counts):
        self.base=ConditionalHistoryPredictor().fit(rows,counts);self.encoder=MetricFeatureEncoder().fit(rows)
        self.scaler=StandardScaler().fit(self._features(rows));x=self.scaler.transform(self._features(rows))
        audience=np.asarray([r['audience'] for r in rows],dtype=float);target=(counts-self._prior(rows))/audience[:,None]*1000.
        self.scale=np.maximum(target.std(axis=0),1.);weight=audience**2;weight/=weight.mean()
        self.model=Ridge(alpha=self.alpha).fit(x,target/self.scale,sample_weight=weight)
        return self

    def predict(self, rows):
        residual=self.model.predict(self.scaler.transform(self._features(rows)))*self.scale
        value=self._prior(rows)+residual*np.asarray([r['audience'] for r in rows])[:,None]/1000.
        return np.rint(np.maximum(0,value))


class CalendarMetricHead:
    def __init__(self, model, metric, strength):self.model=model;self.metric=metric;self.strength=strength

    def predict(self, rows, vectors=None, texts=None):
        prior=self.model._prior(rows);value=self.model.predict(rows)
        return np.rint(np.maximum(0,prior+self.strength*(value-prior)))[:,self.metric]

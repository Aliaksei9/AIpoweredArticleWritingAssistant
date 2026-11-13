"""Research metric heads: smooth within-channel score/topic neighbourhoods."""
from collections import defaultdict

import numpy as np

from src.social_metrics_prediction.features import input_columns,validate_counts
from .history_candidates import ConditionalHistoryPredictor
from .independent_metric_candidates import RateRegressor


class KernelChannelPredictor:
    """All neighbourhoods, priors and robust caps are fitted on training rows."""
    def __init__(self, bandwidth=1.5, topic_boost=1., shrink=20., upper_quantile=1.):
        self.bandwidth=bandwidth;self.topic_boost=topic_boost
        self.shrink=shrink;self.upper_quantile=upper_quantile

    @staticmethod
    def _key(row):return row['source'],row['resource_id']

    def fit(self, rows, counts):
        categories,_=input_columns(rows);counts=validate_counts(counts,len(rows))
        if self.bandwidth<=0 or self.shrink<0 or self.topic_boost<=0:raise ValueError('Invalid kernel configuration')
        values=counts/np.asarray([r['audience'] for r in rows])[:,None]*1000.
        self.sources=set(categories[:,2]);self.global_mean=values.mean(axis=0)
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[self._key(r)].append(i)
        self.channels={}
        for key,indices in groups.items():
            rates=values[indices].copy()
            if self.upper_quantile<1:rates=np.minimum(rates,np.quantile(rates,self.upper_quantile,axis=0))
            self.channels[key]=dict(values=rates,mean=rates.mean(axis=0),
                score=np.asarray([rows[i]['score'] for i in indices]),
                topics=np.asarray([rows[i]['topic'] for i in indices]))
        return self

    def predict_rows(self, rows):
        categories,_=input_columns(rows)
        if not set(categories[:,2]).issubset(self.sources):raise ValueError('Unknown source')
        groups=defaultdict(list);result=np.empty((len(rows),3))
        for i,r in enumerate(rows):groups[self._key(r)].append(i)
        for key,indices in groups.items():
            channel=self.channels.get(key)
            if channel is None:result[indices]=self.global_mean;continue
            score=np.asarray([rows[i]['score'] for i in indices])
            topics=np.asarray([rows[i]['topic'] for i in indices])
            weights=np.exp(-.5*((score[:,None]-channel['score'][None,:])/self.bandwidth)**2)
            weights*=np.where(topics[:,None]==channel['topics'][None,:],self.topic_boost,1.)
            total=weights.sum(axis=1)+self.shrink
            result[indices]=(weights@channel['values']+self.shrink*channel['mean'])/np.maximum(total[:,None],1e-12)
        audience=np.asarray([r['audience'] for r in rows])
        return np.rint(np.maximum(0,result)*audience[:,None]/1000.)


def configurations():
    return {f'kernel_{bandwidth:g}_{boost:g}_{shrink:g}_{quantile:g}':
            dict(bandwidth=bandwidth,topic_boost=boost,shrink=shrink,upper_quantile=quantile)
            for bandwidth in (.5,1.5,3.) for boost in (1.,3.)
            for shrink in (5.,20.) for quantile in (.95,1.)}


class SingleMetricHead:
    """Stable import path allows loading a head saved by a ``python -m`` runner."""
    def __init__(self, metric, configuration):self.metric=metric;self.configuration=configuration

    def fit(self, rows, counts):
        kind=self.configuration['kind'];p=self.configuration['parameters']
        cls=ConditionalHistoryPredictor if kind=='history' else KernelChannelPredictor
        self.model=(RateRegressor(self.metric,**p) if kind=='ridge' else cls(**p)).fit(rows,counts)
        return self

    def predict(self, rows):
        if self.configuration['kind']=='ridge':return self.model.predict(rows)
        return self.model.predict_rows(rows)[:,self.metric]

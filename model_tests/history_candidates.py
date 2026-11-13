"""Conditional, regularized channel histories; experimental models only."""
from collections import defaultdict
from datetime import datetime

import numpy as np

from src.social_metrics_prediction.features import input_columns,validate_counts


class ConditionalHistoryPredictor:
    """Learn channel/topic priors and local score slopes from past posts only.

    Publication times are training metadata, never required for prediction.
    Counts may be clipped for robust fitting; validation targets are unchanged.
    """
    def __init__(self,half_life=None,history_size=None,upper_quantile=1.,
                 topic_strength=0.,topic_shrink=20.,score_strength=0.,slope_ridge=20.):
        self.half_life=half_life;self.history_size=history_size
        self.upper_quantile=upper_quantile;self.topic_strength=topic_strength
        self.topic_shrink=topic_shrink;self.score_strength=score_strength;self.slope_ridge=slope_ridge

    @staticmethod
    def _key(row):
        return row['source'],row['resource_id']

    def fit(self,rows,counts):
        categories,_=input_columns(rows)
        counts=validate_counts(counts,len(rows))
        rates=counts/np.asarray([r['audience'] for r in rows])[:,None]*1000.
        self.sources=set(categories[:,2]);self.global_mean=rates.mean(axis=0)
        grouped=defaultdict(list)
        for i,row in enumerate(rows):grouped[self._key(row)].append(i)
        self.channels={}
        for key,indices in grouped.items():
            indices=sorted(indices,key=lambda i:rows[i].get('published_at') or '')
            if self.history_size:indices=indices[-self.history_size:]
            values=rates[indices].copy()
            if self.upper_quantile<1:
                values=np.minimum(values,np.quantile(values,self.upper_quantile,axis=0))
            weights=(np.ones(len(indices)) if self.half_life is None else
                     np.exp2(-np.arange(len(indices)-1,-1,-1)/self.half_life))
            mean=np.average(values,axis=0,weights=weights)
            topics={}
            if self.topic_strength:
                for topic in {rows[i]['topic'] for i in indices}:
                    mask=np.asarray([rows[i]['topic']==topic for i in indices])
                    topics[topic]=np.sum(weights[mask,None]*(values[mask]-mean),axis=0)/(weights[mask].sum()+self.topic_shrink)
            scores=np.asarray([rows[i]['score'] for i in indices],dtype=float)
            score_center=float(np.average(scores,weights=weights))
            centered=scores-score_center
            target=values-mean
            if self.topic_strength:
                target-=self.topic_strength*np.asarray([topics[rows[i]['topic']] for i in indices])
            slope=(np.sum(weights[:,None]*centered[:,None]*target,axis=0)/
                   (np.sum(weights*centered**2)+self.slope_ridge))
            self.channels[key]=dict(mean=mean,topics=topics,score_center=score_center,slope=slope,
                score_low=float(scores.min()),score_high=float(scores.max()))
        return self

    def predict_rows(self,rows):
        categories,_=input_columns(rows)
        if not set(categories[:,2]).issubset(self.sources):raise ValueError('Unknown source')
        predictions=[]
        for row in rows:
            history=self.channels.get(self._key(row))
            if history is None:rate=self.global_mean
            else:
                rate=history['mean'].copy()
                rate+=self.topic_strength*history['topics'].get(row['topic'],np.zeros(3))
                score=np.clip(row['score'],history['score_low'],history['score_high'])
                rate+=self.score_strength*history['slope']*(score-history['score_center'])
            predictions.append(np.rint(np.maximum(0,rate)*row['audience']/1000.))
        return np.asarray(predictions)


def configurations():
    configs={'mean':{}}
    for half in [3,7,14,30,60]:
        configs[f'ema_{half}']=dict(half_life=half)
    for size in [5,10,20,40,80]:
        configs[f'last_{size}']=dict(history_size=size)
    for quantile in [.8,.9,.95,.98]:
        configs[f'robust_{quantile}']=dict(upper_quantile=quantile)
    for shrink in [2.,5.,20.,50.,100.]:
        for strength in [.25,.5,1.]:
            configs[f'topic_{shrink:g}_{strength:g}']=dict(topic_shrink=shrink,topic_strength=strength)
    for ridge in [5.,20.,100.,500.]:
        for strength in [.25,.5,1.]:
            configs[f'slope_{ridge:g}_{strength:g}']=dict(slope_ridge=ridge,score_strength=strength)
    for half in [7,14,30]:
        for quantile in [.9,.95,1.]:
            for topic in [0.,.5,1.]:
                for score in [0.,.5]:
                    name=f'joint_{half}_{quantile}_{topic}_{score}'
                    configs[name]=dict(half_life=half,upper_quantile=quantile,
                        topic_strength=topic,topic_shrink=20.,score_strength=score,slope_ridge=100.)
    return configs

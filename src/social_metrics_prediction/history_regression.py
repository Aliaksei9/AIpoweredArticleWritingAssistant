"""Robust per-channel histories with topic shrinkage and score calibration."""
from datetime import datetime

import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.utils.validation import check_is_fitted,validate_data


class ChannelHistoryRegressor(RegressorMixin,BaseEstimator):
    """Predict a scalar count rate from a known channel's past publications.

    Publication dates are used for fitting only. Prediction still requires
    only the topic, channel/source identity, popularity score and audience.
    Without complete dates in a channel, recency weights are uniform.
    """
    def __init__(self,resource_columns=(),half_life=30,upper_quantile=.95,
                 topic_strength=1.,topic_shrink=20.,score_strength=.5,slope_ridge=100.):
        self.resource_columns=resource_columns
        self.half_life=half_life;self.upper_quantile=upper_quantile
        self.topic_strength=topic_strength;self.topic_shrink=topic_shrink
        self.score_strength=score_strength;self.slope_ridge=slope_ridge

    def _categories(self,features):
        resources=features[:,self.resource_columns]
        topics=features[:,:self.resource_columns[0]]
        for values in [resources,topics]:
            if not np.isin(values,[0.,1.]).all() or np.any(values.sum(axis=1)>1):
                raise ValueError('Channel and topic identities must be one-hot columns')
        return (np.where(resources.sum(axis=1)>0,np.argmax(resources,axis=1),-1),
                np.where(topics.sum(axis=1)>0,np.argmax(topics,axis=1),-1))

    def fit(self,X,y):
        return self.fit_with_dates(X,y)

    def fit_with_dates(self,X,y,published_at=None):
        features,targets=validate_data(self,X,y,dtype=np.float64,y_numeric=True)
        if not self.resource_columns or self.resource_columns[0]<1:
            raise ValueError('Channel columns must follow at least one topic column')
        if max(self.resource_columns)>=features.shape[1]-2:
            raise ValueError('Channel columns must precede score and audience')
        if not 0<self.upper_quantile<=1 or not np.isfinite(self.upper_quantile):
            raise ValueError('Robust upper quantile must be in (0, 1]')
        if self.half_life<=0 or self.topic_shrink<=0 or self.slope_ridge<=0:
            raise ValueError('History and regularization parameters must be positive')
        if not all(np.isfinite(value) for value in [self.half_life,self.topic_shrink,self.slope_ridge,
                                                    self.topic_strength,self.score_strength]):
            raise ValueError('History parameters must be finite')
        if published_at is None:published_at=[None]*len(features)
        if len(published_at)!=len(features):raise ValueError('Date count does not match training rows')
        channels,topics=self._categories(features)
        scores=features[:,-2]*10.
        self.global_mean_=float(targets.mean())
        self.channels_={}
        self.channels_without_dates_=0
        for channel in np.unique(channels):
            indices=np.flatnonzero(channels==channel)
            complete=all(published_at[i] is not None for i in indices)
            if complete:
                indices=np.asarray(sorted(indices,key=lambda i:(published_at[i].timestamp()
                    if isinstance(published_at[i],datetime) else datetime.fromisoformat(published_at[i]).timestamp())))
                weights=np.exp2(-np.arange(len(indices)-1,-1,-1)/self.half_life)
            else:
                self.channels_without_dates_+=1
                weights=np.ones(len(indices))
            values=targets[indices].copy()
            values=np.minimum(values,np.quantile(values,self.upper_quantile))
            mean=float(np.average(values,weights=weights))
            adjustments={}
            for topic in np.unique(topics[indices]):
                mask=topics[indices]==topic
                adjustments[int(topic)]=float(np.sum(weights[mask]*(values[mask]-mean))/
                                               (weights[mask].sum()+self.topic_shrink))
            score_center=float(np.average(scores[indices],weights=weights))
            centered=scores[indices]-score_center
            residual=values-mean-self.topic_strength*np.asarray([adjustments[int(t)] for t in topics[indices]])
            slope=float(np.sum(weights*centered*residual)/(np.sum(weights*centered**2)+self.slope_ridge))
            self.channels_[int(channel)]=dict(mean=mean,topics=adjustments,score_center=score_center,
                slope=slope,score_low=float(scores[indices].min()),score_high=float(scores[indices].max()))
        return self

    def predict(self,X):
        check_is_fitted(self,'channels_')
        features=validate_data(self,X,dtype=np.float64,reset=False)
        channels,topics=self._categories(features)
        result=[]
        for row,channel,topic in zip(features,channels,topics):
            history=self.channels_.get(int(channel))
            if history is None:result.append(self.global_mean_);continue
            score=np.clip(row[-2]*10.,history['score_low'],history['score_high'])
            value=(history['mean']+self.topic_strength*history['topics'].get(int(topic),0.)+
                   self.score_strength*history['slope']*(score-history['score_center']))
            result.append(float(value))
        return np.asarray(result)

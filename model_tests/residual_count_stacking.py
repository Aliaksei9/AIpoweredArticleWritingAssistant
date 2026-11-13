"""Research correction trained only on out-of-fold social predictions."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.social_metrics_prediction.features import MetricFeatureEncoder
from .alternative_metric_models import predict_heads
from .history_candidates import ConditionalHistoryPredictor


class CountCorrection:
    def __init__(self, alpha):self.alpha=alpha

    def _prior(self, rows):
        return np.asarray([self.base.channels.get(self.base._key(r),dict(mean=self.base.global_mean))['mean'] for r in rows])*np.asarray([r['audience'] for r in rows])[:,None]/1000.

    def _features(self, rows, predictions, priors):
        encoded=self.encoder.transform(rows)
        ratio=np.clip((predictions+1)/(priors+1),.001,50.)
        return np.column_stack([encoded,ratio-1,np.log(ratio),
            (encoded[:,:-2,None]*(ratio[:,None,:]-1)).reshape(len(rows),-1)])

    def fit(self, rows, counts, predictions, oof_priors):
        self.base=ConditionalHistoryPredictor().fit(rows,counts);self.encoder=MetricFeatureEncoder().fit(rows)
        self.scaler=StandardScaler().fit(self._features(rows,predictions,oof_priors))
        x=self.scaler.transform(self._features(rows,predictions,oof_priors));self.models=[]
        for metric in range(3):
            target=(counts[:,metric]-predictions[:,metric])/(oof_priors[:,metric]+1)
            weights=(oof_priors[:,metric]+1)**2;weights/=weights.mean()
            self.models.append(Ridge(alpha=self.alpha).fit(x,target,sample_weight=weights))
        return self

    def raw(self, rows, predictions):
        prior=self._prior(rows);x=self.scaler.transform(self._features(rows,predictions,prior))
        return predictions+(prior+1)*np.column_stack([m.predict(x) for m in self.models])


class StackedCountPipeline:
    def __init__(self, base_heads, corrections):self.base_heads=base_heads;self.corrections=corrections

    def predict(self, rows, vectors, texts):
        base=predict_heads(self.base_heads,rows,vectors,texts);cache={};result=base.copy()
        for metric,(model,strength) in self.corrections.items():
            key=id(model)
            if key not in cache:cache[key]=model.raw(rows,base)
            result[:,metric]=np.rint(np.maximum(0,base[:,metric]+strength*(cache[key][:,metric]-base[:,metric])))
        return result


class StackedMetricHead:
    def __init__(self, pipeline, metric):self.pipeline=pipeline;self.metric=metric

    def predict(self, rows, vectors, texts):return self.pipeline.predict(rows,vectors,texts)[:,self.metric]

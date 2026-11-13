"""Separate estimators and parameter selection for each social metric."""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.social_metrics_prediction.features import MetricFeatureEncoder
from .history_candidates import ConditionalHistoryPredictor,configurations


class RateRegressor:
    def __init__(self,metric,family='ridge',alpha=100.,depth=4,iterations=150,relative=False):
        self.metric=metric;self.family=family;self.alpha=alpha;self.depth=depth
        self.iterations=iterations;self.relative=relative

    def fit(self,rows,counts):
        self.encoder=MetricFeatureEncoder().fit(rows)
        self.base=ConditionalHistoryPredictor().fit(rows,counts)
        X=self.encoder.transform(rows)
        prior=np.asarray([self.base.channels[self.base._key(r)]['mean'][self.metric] for r in rows])
        audience=np.asarray([r['audience'] for r in rows])
        y=counts[:,self.metric]/audience*1000.
        self.scale=max(float(np.std(y)),1e-8)
        if self.relative:
            y=y/np.maximum(prior,1.)-1.;weights=(audience*np.maximum(prior,1.))**2
        else:y=(y-prior)/self.scale;weights=audience**2
        weights=weights/np.mean(weights)
        onehot=X[:,:-2];score=X[:,-2: -1]
        X=np.column_stack([X,onehot*score,score**2])
        self.scaler=StandardScaler().fit(X);X=self.scaler.transform(X)
        if self.family=='ridge':model=Ridge(alpha=self.alpha)
        elif self.family=='extra':model=ExtraTreesRegressor(n_estimators=self.iterations,max_depth=self.depth,
            min_samples_leaf=10,n_jobs=4,random_state=42)
        elif self.family=='hist':model=HistGradientBoostingRegressor(max_iter=self.iterations,max_leaf_nodes=7,
            min_samples_leaf=40,l2_regularization=self.alpha,early_stopping=False,random_state=42)
        elif self.family=='cat':
            from catboost import CatBoostRegressor
            model=CatBoostRegressor(iterations=self.iterations,depth=self.depth,l2_leaf_reg=self.alpha,
                learning_rate=.04,loss_function='RMSE',verbose=False,thread_count=4,random_seed=42,allow_writing_files=False)
        else:raise ValueError(self.family)
        self.model=model.fit(X,y,sample_weight=weights)
        return self

    def predict(self,rows):
        X=self.encoder.transform(rows);onehot=X[:,:-2];score=X[:,-2:-1]
        X=self.scaler.transform(np.column_stack([X,onehot*score,score**2]))
        prior=np.asarray([self.base.channels.get(self.base._key(r),dict(mean=self.base.global_mean))['mean'][self.metric] for r in rows])
        residual=self.model.predict(X)
        value=prior+residual*(np.maximum(prior,1.) if self.relative else self.scale)
        return np.rint(np.maximum(0,value)*np.asarray([r['audience'] for r in rows])/1000.)


@dataclass
class MetricModel:
    """One independently selected estimator, not a shared three-output model."""
    metric: int
    configuration: dict

    def fit(self,rows,counts):
        if self.configuration.get('family'):
            self.estimator=RateRegressor(self.metric,**self.configuration).fit(rows,counts)
        else:self.estimator=ConditionalHistoryPredictor(**self.configuration).fit(rows,counts)
        return self

    def predict(self,rows):
        if isinstance(self.estimator,RateRegressor):return self.estimator.predict(rows)
        return self.estimator.predict_rows(rows)[:,self.metric]


class ViewsModel(MetricModel):
    def __init__(self,configuration):super().__init__(0,configuration)


class ReactionsModel(MetricModel):
    def __init__(self,configuration):super().__init__(1,configuration)


class CommentsModel(MetricModel):
    def __init__(self,configuration):super().__init__(2,configuration)


def candidate_grid():
    result={name:dict(value) for name,value in configurations().items()}
    for alpha in [1.,100.,10000.]:
        for relative in [False,True]:result[f'ridge_{alpha}_{relative}']=dict(family='ridge',alpha=alpha,relative=relative)
    for family in ['cat','extra','hist']:
        for depth in [3,6]:
            for relative in [False,True]:
                result[f'{family}_{depth}_{relative}']=dict(family=family,alpha=20.,depth=depth,iterations=120,relative=relative)
    return result

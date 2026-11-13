"""Regularized baseline corrections for experiments, outside application code."""
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.social_metrics_prediction.features import MetricFeatureEncoder, validate_counts
from .social_metric_candidates import ResourceMeanBaseline


TEXT_CONFIGS = {
    'original': dict(hidden_layers=(256,64), dropout=.2, epochs=100, weight_decay=.01, learning_rate=.001),
    'compact_25': dict(hidden_layers=(128,), dropout=.5, epochs=25, weight_decay=.1, learning_rate=.0005),
    'compact_50': dict(hidden_layers=(128,), dropout=.5, epochs=50, weight_decay=.1, learning_rate=.0005),
    'compact_deep_40': dict(hidden_layers=(128,64), dropout=.35, epochs=40, weight_decay=.1, learning_rate=.001),
    'wide_30': dict(hidden_layers=(256,64), dropout=.5, epochs=30, weight_decay=.1, learning_rate=.001),
    'wide_60': dict(hidden_layers=(256,128), dropout=.5, epochs=60, weight_decay=.1, learning_rate=.0005),
    'strong_l2_20': dict(hidden_layers=(128,), dropout=.2, epochs=20, weight_decay=1., learning_rate=.001),
    'slow_80': dict(hidden_layers=(128,64), dropout=.5, epochs=80, weight_decay=.1, learning_rate=.0003),
}


class ResidualRatePredictor:
    """Fit rate residuals around train-only resource means, then shrink them.

    The baseline, categorical vocabularies, centering statistics and scalers
    are fitted exclusively on training rows. No true popularity labels or
    social counts are required at prediction time.
    """
    def __init__(self, family='ridge', alpha=100., hidden=(32,), epochs=80,
                 blend=1., seed=42, threads=4):
        self.family, self.alpha, self.hidden = family, alpha, hidden
        self.epochs, self.blend, self.seed, self.threads = epochs, blend, seed, threads

    @staticmethod
    def _keys(rows):
        return [(r['source'],r['resource_id']) for r in rows]

    def _baseline_rates(self, rows):
        return np.asarray([self.baseline.means.get(key,self.baseline.global_mean)*1000.
                           for key in self._keys(rows)])

    def _features(self, rows):
        base = self.encoder.transform(rows)
        onehot = base[:,:-2]
        # Relative score separates within-channel text effects from fixed means.
        score = np.asarray([(r['score']-self.score_means.get(k,self.global_score))/10.
                            for r,k in zip(rows,self._keys(rows))])[:,None]
        return np.column_stack([onehot,base[:,-1:],score,score**2,score**3,onehot*score])

    def fit(self, rows, counts):
        counts = validate_counts(counts,len(rows))
        self.baseline = ResourceMeanBaseline().fit(rows,counts)
        self.encoder = MetricFeatureEncoder().fit(rows)
        keys = self._keys(rows)
        self.global_score = np.mean([r['score'] for r in rows])
        self.score_means = {k:np.mean([r['score'] for r,key in zip(rows,keys) if key==k]) for k in set(keys)}
        audience = np.asarray([r['audience'] for r in rows],dtype=float)
        residual = counts/audience[:,None]*1000.-self._baseline_rates(rows)
        weights=(audience/audience.max())**2
        weights/=weights.mean()
        self.x_scaler = StandardScaler().fit(self._features(rows))
        self.y_scaler = StandardScaler().fit(residual)
        features=self.x_scaler.transform(self._features(rows))
        targets=self.y_scaler.transform(residual)
        self.estimators=[]
        with threadpool_limits(limits=self.threads), warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=ConvergenceWarning)
            for metric in range(3):
                if self.family=='ridge':
                    estimator=Ridge(alpha=self.alpha)
                elif self.family=='mlp':
                    estimator=MLPRegressor(hidden_layer_sizes=self.hidden,alpha=self.alpha,
                        batch_size=128,max_iter=self.epochs,n_iter_no_change=self.epochs,
                        early_stopping=False,random_state=self.seed+metric,learning_rate_init=.001)
                elif self.family=='boosting':
                    estimator=HistGradientBoostingRegressor(max_iter=self.epochs,max_leaf_nodes=5,
                        min_samples_leaf=80,l2_regularization=self.alpha,learning_rate=.03,
                        early_stopping=False,random_state=self.seed+metric)
                elif self.family=='catboost':
                    from catboost import CatBoostRegressor
                    estimator=CatBoostRegressor(iterations=self.epochs,depth=3,learning_rate=.03,
                        l2_leaf_reg=self.alpha,loss_function='RMSE',random_seed=self.seed+metric,
                        thread_count=self.threads,verbose=False,allow_writing_files=False)
                else:
                    raise ValueError('Unknown residual family')
                estimator.fit(features,targets[:,metric],sample_weight=weights)
                self.estimators.append(estimator)
        return self

    def predict_rates(self, rows):
        features=self.x_scaler.transform(self._features(rows))
        with threadpool_limits(limits=self.threads):
            residual=self.y_scaler.inverse_transform(np.column_stack([m.predict(features) for m in self.estimators]))
        return np.maximum(0,self._baseline_rates(rows)+self.blend*residual)

    def predict_rows(self, rows):
        rates=self.predict_rates(rows)
        result=np.rint(rates*np.asarray([r['audience'] for r in rows])[:,None]/1000.)
        if not np.isfinite(result).all():
            raise RuntimeError('Nonfinite residual predictions')
        return result


SOCIAL_CONFIGS = {
    **{f'ridge_{alpha:g}':dict(family='ridge',alpha=alpha) for alpha in [1.,10.,100.,1000.,10000.]},
    **{f'mlp_{width}_{epochs}_{alpha:g}':dict(family='mlp',hidden=(width,),epochs=epochs,alpha=alpha)
       for width,epochs,alpha in [(16,40,1.),(16,100,10.),(32,60,1.),(32,120,10.),(64,60,10.),(32,60,100.)]},
    'boosting_60':dict(family='boosting',epochs=60,alpha=10.),
    'boosting_150':dict(family='boosting',epochs=150,alpha=20.),
    'catboost_150':dict(family='catboost',epochs=150,alpha=20.),
    'catboost_300':dict(family='catboost',epochs=300,alpha=50.),
}


class MetricWiseResidualPredictor:
    """One frozen CV-selected configuration per social count."""
    def __init__(self, configurations):
        self.configurations=configurations

    def fit(self, rows, counts):
        self.models=[]
        for configuration in self.configurations:
            model=(ResourceMeanBaseline() if configuration is None
                   else ResidualRatePredictor(**configuration))
            model.fit(rows,counts)
            self.models.append(model)
        return self

    def predict_rows(self, rows):
        return np.column_stack([m.predict_rows(rows)[:,i] for i,m in enumerate(self.models)])

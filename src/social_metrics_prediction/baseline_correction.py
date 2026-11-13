"""Train-only channel means with a regularized correction from input features."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data
from threadpoolctl import threadpool_limits


class ChannelMeanCorrection(RegressorMixin, BaseEstimator):
    """A scalar rate regressor; channel identity is supplied by one-hot columns.

    Fit all priors, score centers and scalers on training data only. The
    underlying estimator learns residual rates. Shrinkage controls how far
    predictions depart from a channel's empirical mean. Predictions for an
    unknown channel use the global training mean as their starting point.
    """
    def __init__(self, estimator, resource_columns, blend=.5, threads=4):
        self.estimator = estimator
        self.resource_columns = resource_columns
        self.blend = blend
        self.threads = threads

    def _channels(self, features):
        columns = features[:, self.resource_columns]
        if not np.isin(columns, [0., 1.]).all() or np.any(columns.sum(axis=1)>1):
            raise ValueError('Resource columns must contain one-hot channel identities')
        return np.where(columns.sum(axis=1)>0, np.argmax(columns,axis=1), -1)

    def _features(self, features):
        channels = self._channels(features)
        score = np.asarray([value-self.score_means_.get(int(channel),self.global_score_)
                            for value,channel in zip(features[:,-2],channels)])[:,None]
        categories = features[:,:-2]
        return np.column_stack([categories,features[:,-1:],score,score**2,score**3,categories*score])

    def _means(self, features):
        return np.asarray([self.channel_means_.get(int(channel),self.global_mean_)
                           for channel in self._channels(features)])

    def fit(self, X, y, sample_weight=None):
        features,targets = validate_data(self,X,y,dtype=np.float64,y_numeric=True)
        if not 0 <= self.blend <= 1 or not np.isfinite(self.blend):
            raise ValueError('Correction strength must be finite and in [0, 1]')
        if (not self.resource_columns or any(i<0 or i>=features.shape[1]-2 for i in self.resource_columns)):
            raise ValueError('Resource columns must precede the score and audience features')
        channels = self._channels(features)
        self.global_mean_ = float(np.mean(targets))
        self.global_score_ = float(np.mean(features[:,-2]))
        self.channel_means_ = {int(key):float(np.mean(targets[channels==key])) for key in np.unique(channels)}
        self.score_means_ = {int(key):float(np.mean(features[channels==key,-2])) for key in np.unique(channels)}
        residual = targets-self._means(features)
        self.x_scaler_ = StandardScaler().fit(self._features(features))
        self.y_scaler_ = StandardScaler().fit(residual[:,None])
        self.estimator_ = clone(self.estimator)
        with threadpool_limits(limits=self.threads):
            self.estimator_.fit(self.x_scaler_.transform(self._features(features)),
                self.y_scaler_.transform(residual[:,None]).ravel(),sample_weight=sample_weight)
        return self

    def predict(self, X):
        check_is_fitted(self,'estimator_')
        features = validate_data(self,X,dtype=np.float64,reset=False)
        with threadpool_limits(limits=self.threads):
            residual = self.estimator_.predict(self.x_scaler_.transform(self._features(features)))
        correction = self.y_scaler_.inverse_transform(residual[:,None]).ravel()
        return self._means(features)+self.blend*correction

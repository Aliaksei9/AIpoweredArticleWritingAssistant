"""The whole-corpus validated additive/topic neural architecture."""
from copy import deepcopy
import math

import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
import torch
from torch import nn

from .features import FEATURE_DIMENSIONS
from .neural_network import TorchMLPRegressor


class AdditiveFeatureScaler:
    def fit(self,x):
        self.split=FEATURE_DIMENSIONS//2
        self.document_scaler=StandardScaler().fit(x[:,:self.split])
        self.topic_scaler=StandardScaler().fit(x[:,self.split:]);self.topic_scale=.25
        return self

    def transform(self,x):
        return np.column_stack([self.document_scaler.transform(x[:,:self.split]),
            self.topic_scaler.transform(x[:,self.split:])*self.topic_scale]).astype(np.float32)


class AdditivePopularityNetwork(nn.Module):
    def __init__(self):
        super().__init__();self.document_dimensions=FEATURE_DIMENSIONS//2
        self.document=nn.Sequential(nn.Linear(self.document_dimensions,256),nn.GELU(),nn.Dropout(.5),
            nn.Linear(256,128),nn.GELU(),nn.Dropout(.5))
        self.document_output=nn.Linear(128,1)
        self.topic_output=nn.Linear(self.document_dimensions,1,bias=False)

    def forward(self,x):
        value=self.document_output(self.document(x[:,:self.document_dimensions]))+self.topic_output(x[:,self.document_dimensions:])
        return 10.*torch.tanh(value.squeeze(-1))


class AdditiveTextRegressor:
    def __init__(self,seed=42,num_threads=4):self.seed=seed;self.num_threads=num_threads

    def fit(self,x,y):
        self.preprocessing=AdditiveFeatureScaler().fit(x)
        inputs=torch.from_numpy(self.preprocessing.transform(x));expected=torch.from_numpy(np.asarray(y,dtype=np.float32))
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(self.num_threads)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed);network=AdditivePopularityNetwork()
                optimizer=torch.optim.AdamW(network.parameters(),lr=.0005,weight_decay=.2)
                generator=torch.Generator().manual_seed(self.seed);average=None;averaged=0;self.history_=[]
                for epoch in range(35):
                    rate=.0005*(.05+.95*.5*(1+math.cos(math.pi*epoch/34)))
                    for group in optimizer.param_groups:group['lr']=rate
                    network.train();order=torch.randperm(len(inputs),generator=generator);total=0.
                    for start in range(0,len(inputs),128):
                        ids=order[start:start+128];optimizer.zero_grad(set_to_none=True)
                        loss=((network(inputs[ids])-expected[ids])**2).mean()
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5.);optimizer.step()
                        total+=float(loss.detach())*len(ids)
                    if epoch>=35*3//4:
                        if average is None:average=deepcopy(network)
                        else:
                            with torch.no_grad():
                                for a,b in zip(average.parameters(),network.parameters()):a.add_((b-a)/(averaged+1))
                        averaged+=1
                    self.history_.append(dict(epoch=epoch+1,train_loss=total/len(inputs)))
                self.network_=average.eval()
        finally:torch.set_num_threads(previous)
        return self

    def predict(self,x):
        self.network_.eval()
        with torch.inference_mode():return self.network_(torch.from_numpy(self.preprocessing.transform(x))).numpy().astype(float)


class TextPopularityEnsemble(RegressorMixin,BaseEstimator):
    def __init__(self,seed=42,num_threads=4):self.seed=seed;self.num_threads=num_threads

    def __sklearn_is_fitted__(self):return hasattr(self,'components_')

    def fit(self,x,y):
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=FEATURE_DIMENSIONS or y.shape!=(len(x),) or not len(x):raise ValueError('Invalid text training shapes')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.abs(y)>10):raise ValueError('Invalid text training values')
        self.n_features_in_=x.shape[1]
        self.components_=[AdditiveTextRegressor(self.seed,self.num_threads).fit(x,y)]
        for seed in (self.seed,self.seed+31):
            self.components_.append(TorchMLPRegressor(hidden_layers=(512,128),dropout=.65,epochs=25,
                learning_rate=.0005,weight_decay=.2,seed=seed,num_threads=self.num_threads).fit(x,y))
        self.history_=self.components_[0].history_;return self

    def predict(self,x):
        check_is_fitted(self);x=np.asarray(x,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=FEATURE_DIMENSIONS or not np.isfinite(x).all():raise ValueError('Invalid text features')
        if not len(x):return np.empty(0,dtype=float)
        result=np.average([m.predict(x) for m in self.components_],axis=0,weights=[.5,.25,.25])
        if not np.isfinite(result).all():raise RuntimeError('Invalid popularity prediction')
        return np.clip(result,-10.,10.)

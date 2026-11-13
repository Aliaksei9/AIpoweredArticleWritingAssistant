"""Research neural networks; never imported by the production modules."""
from copy import deepcopy
import math

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from src.assesment_conform_of_PP_text_based.features import FEATURE_DIMENSIONS


CONFIGURATIONS = {
    'gelu_swa': dict(widths=(512,128), dropout=.65, epochs=40, lr=.001, decay=.2),
    'mixup_swa': dict(widths=(512,128), dropout=.6, epochs=40, lr=.0008, decay=.2, mixup=.15),
    'residual_norm': dict(kind='residual', widths=(256,256), dropout=.4, epochs=35, lr=.0005, decay=.2),
    'pca128': dict(pca=128, widths=(256,64), dropout=.5, epochs=40, lr=.001, decay=.2),
    'pca64': dict(pca=64, widths=(128,64), dropout=.4, epochs=40, lr=.001, decay=.2),
    'additive_topic': dict(kind='additive', widths=(256,128), topic_scale=.25, dropout=.5, epochs=35, lr=.0005, decay=.2),
    'ordinal': dict(kind='ordinal', widths=(256,128), dropout=.5, epochs=40, lr=.001, decay=.2),
    'balanced_features': dict(scaling='centered', widths=(256,128), dropout=.5, epochs=40, lr=.001, decay=.2),
}


class NeuralFeatures:
    """Fit scaling and optional document PCA on training features only."""
    def __init__(self, pca=None, scaling='standard', topic_scale=1.):
        self.components=pca; self.scaling=scaling; self.topic_scale=topic_scale

    def fit(self, features):
        x=self._validate(features)
        self.split=FEATURE_DIMENSIONS//2
        standardized=self.scaling=='standard'
        self.document_scaler=StandardScaler(with_std=standardized).fit(x[:,:self.split])
        self.topic_scaler=StandardScaler(with_std=standardized).fit(x[:,self.split:])
        doc=self.document_scaler.transform(x[:,:self.split])
        self.pca=None
        if self.components:
            n=min(self.components,len(x)-1,self.split)
            self.pca=PCA(n_components=n,whiten=True,svd_solver='randomized',random_state=42).fit(doc)
        self.document_dimensions=self.pca.n_components_ if self.pca is not None else self.split
        return self

    @staticmethod
    def _validate(features):
        x=np.asarray(features,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=FEATURE_DIMENSIONS or not np.isfinite(x).all():
            raise ValueError(f'Expected finite (n, {FEATURE_DIMENSIONS}) features')
        return x

    def transform(self, features):
        x=self._validate(features)
        doc=self.document_scaler.transform(x[:,:self.split])
        if self.pca is not None:doc=self.pca.transform(doc)
        topic=self.topic_scaler.transform(x[:,self.split:])*self.topic_scale
        if self.scaling=='centered':
            doc=doc*math.sqrt(self.split);topic=topic*math.sqrt(self.split)
        return np.column_stack([doc,topic]).astype(np.float32)


def dense_layers(start, widths, dropout):
    layers=[]
    for width in widths:
        layers.extend([nn.Linear(start,width),nn.GELU(),nn.Dropout(dropout)])
        start=width
    return nn.Sequential(*layers)


class ResearchNetwork(nn.Module):
    def __init__(self, dimensions, document_dimensions, kind, widths, dropout):
        super().__init__();self.kind=kind;self.document_dimensions=document_dimensions
        if kind=='residual':
            width=widths[0]
            self.input=nn.Sequential(nn.Linear(dimensions,width),nn.GELU())
            self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),
                nn.GELU(),nn.Dropout(dropout),nn.Linear(width,width),nn.Dropout(dropout)) for _ in widths])
            self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        elif kind=='additive':
            self.document=dense_layers(document_dimensions,widths,dropout)
            self.document_output=nn.Linear(widths[-1],1)
            self.topic_output=nn.Linear(dimensions-document_dimensions,1,bias=False)
        else:
            self.hidden=dense_layers(dimensions,widths,dropout)
            self.output=nn.Linear(widths[-1],21 if kind=='ordinal' else 1)
        self.register_buffer('bins',torch.linspace(-10.,10.,21))

    def logits(self, x):
        if self.kind=='residual':
            h=self.input(x)
            for block in self.blocks:h=h+.3*block(h)
            return self.output(h)
        if self.kind=='additive':
            return self.document_output(self.document(x[:,:self.document_dimensions]))+self.topic_output(x[:,self.document_dimensions:])
        return self.output(self.hidden(x))

    def forward(self, x):
        value=self.logits(x)
        if self.kind=='ordinal':return (torch.softmax(value,dim=-1)*self.bins).sum(dim=-1)
        return 10.*torch.tanh(value.squeeze(-1))


class AdvancedTextRegressor:
    """Cosine learning rate, optional mixup, and final-epoch weight averaging."""
    def __init__(self, configuration, seed=42, threads=4):
        self.configuration=dict(configuration);self.seed=seed;self.threads=threads

    def fit(self, features, labels):
        c=self.configuration;x=NeuralFeatures._validate(features);y=np.asarray(labels,dtype=np.float32)
        if y.shape!=(len(x),) or not np.isfinite(y).all() or np.any(np.abs(y)>10):
            raise ValueError('Expected aligned finite scores in [-10,10]')
        if len(x)<2:raise ValueError('At least two training rows required')
        self.preprocessing=NeuralFeatures(c.get('pca'),c.get('scaling','standard'),c.get('topic_scale',1.)).fit(x)
        inputs=torch.from_numpy(self.preprocessing.transform(x));expected=torch.from_numpy(y)
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(self.threads)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed);rng=np.random.default_rng(self.seed)
                network=ResearchNetwork(inputs.shape[1],self.preprocessing.document_dimensions,
                    c.get('kind','dense'),tuple(c['widths']),c['dropout'])
                optimizer=torch.optim.AdamW(network.parameters(),lr=c['lr'],weight_decay=c['decay'])
                generator=torch.Generator().manual_seed(self.seed)
                epochs=c['epochs'];average=None;averaged=0;self.history=[]
                for epoch in range(epochs):
                    rate=c['lr']*(.05+.95*.5*(1+math.cos(math.pi*epoch/max(1,epochs-1))))
                    for group in optimizer.param_groups:group['lr']=rate
                    network.train();order=torch.randperm(len(inputs),generator=generator)
                    total=0.
                    for start in range(0,len(inputs),128):
                        index=order[start:start+128];batch=inputs[index];target=expected[index]
                        if c.get('mixup'):
                            lam=float(rng.beta(c['mixup'],c['mixup']))
                            other=torch.randperm(len(index),generator=generator)
                            batch=lam*batch+(1-lam)*batch[other];target=lam*target+(1-lam)*target[other]
                        optimizer.zero_grad(set_to_none=True)
                        if network.kind=='ordinal':
                            logits=network.logits(batch);z=(target+10.).clamp(0,20)
                            low=torch.floor(z).long();high=torch.clamp(low+1,max=20);weight=z-low
                            logp=torch.log_softmax(logits,dim=-1);indices=torch.arange(len(z))
                            cross_entropy=-((1-weight)*logp[indices,low]+weight*logp[indices,high]).mean()
                            predicted=(torch.softmax(logits,dim=-1)*network.bins).sum(dim=-1)
                            loss=5.*cross_entropy+.25*((predicted-target)**2).mean()
                        else:loss=((network(batch)-target)**2).mean()
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural training loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
                        optimizer.step();total+=float(loss.detach())*len(index)
                    if epoch>=epochs*3//4:
                        if average is None:average=deepcopy(network)
                        else:
                            with torch.no_grad():
                                for avg,current in zip(average.parameters(),network.parameters()):
                                    avg.add_((current-avg)/(averaged+1))
                        averaged+=1
                    self.history.append(dict(epoch=epoch+1,loss=total/len(inputs),lr=rate))
                self.network=average.eval();self.averaged_epochs=averaged
        finally:torch.set_num_threads(previous)
        return self

    def predict(self, features):
        x=self.preprocessing.transform(features)
        if not len(x):return np.empty(0,dtype=float)
        self.network.eval()
        with torch.inference_mode():result=self.network(torch.from_numpy(x)).numpy().astype(float)
        if not np.isfinite(result).all() or np.any(np.abs(result)>10.00001):
            raise RuntimeError('Invalid neural prediction')
        return np.clip(result,-10.,10.)


class TextPredictorBundle:
    def __init__(self, models, weights=None):
        self.models=models
        self.weights=np.asarray(weights if weights is not None else np.ones(len(models)),dtype=float)
        self.weights/=self.weights.sum()

    def predict(self, features):
        values=[m.model.predict(features) if hasattr(m,'model') else m.predict(features) for m in self.models]
        return np.average(values,axis=0,weights=self.weights)

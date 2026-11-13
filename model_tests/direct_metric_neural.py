"""Independent research neural count heads; no channel enters module 3."""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn

from src.social_metrics_prediction.features import validate_counts
from .alternative_metric_models import AlternativeFeatures
from .history_candidates import ConditionalHistoryPredictor


CONFIGURATIONS={
    'dense_counts':dict(features='dense',target='counts'),
    'dense_rates':dict(features='dense',target='rates'),
    'pca_counts':dict(features='pca',target='counts'),
    'positive_poisson':dict(features='dense',target='poisson'),
}


class DirectMetricNeural:
    def __init__(self, configuration, epochs=25, threads=4):
        self.configuration=dict(configuration);self.epochs=epochs;self.threads=threads

    def _prior(self, rows):
        return np.asarray([self.base.channels.get(self.base._key(r),dict(mean=self.base.global_mean))['mean']
                           for r in rows])*np.asarray([r['audience'] for r in rows])[:,None]/1000.

    def fit(self, rows, counts, vectors, texts):
        counts=validate_counts(counts,len(rows));c=self.configuration
        self.poisson_output_offset=0.
        self.base=ConditionalHistoryPredictor().fit(rows,counts)
        self.features=AlternativeFeatures(c['features']).fit(rows,vectors,texts)
        x=torch.from_numpy(np.asarray(self.features.transform(rows,vectors,texts),dtype=np.float32))
        prior=self._prior(rows);audience=np.asarray([r['audience'] for r in rows],dtype=float)
        target=counts-prior;weight=np.ones(len(rows))
        if c['target']=='rates':target=target/audience[:,None]*1000.;weight=audience**2;weight/=weight.mean()
        self.scale=np.maximum(np.std(target,axis=0),1.)
        expected=torch.from_numpy((target/self.scale).astype(np.float32))
        actual=torch.from_numpy(counts.astype(np.float32));offset=torch.from_numpy(np.log1p(prior).astype(np.float32))
        mean=torch.from_numpy(np.maximum(counts.mean(axis=0),1.).astype(np.float32))
        weights=torch.from_numpy(weight.astype(np.float32));previous=torch.get_num_threads()
        try:
            torch.set_num_threads(self.threads)
            with torch.random.fork_rng(devices=[]):
                networks=[]
                for metric in range(3):
                    torch.manual_seed(42+metric)
                    network=nn.Sequential(nn.Linear(x.shape[1],128),nn.GELU(),nn.Dropout(.5),
                        nn.Linear(128,64),nn.GELU(),nn.Dropout(.5),nn.Linear(64,1))
                    nn.init.zeros_(network[-1].weight);nn.init.zeros_(network[-1].bias);networks.append(network)
                optimizer=torch.optim.AdamW([p for model in networks for p in model.parameters()],lr=.0005,weight_decay=.2)
                generator=torch.Generator().manual_seed(42);average=None;averaged=0
                for epoch in range(self.epochs):
                    for group in optimizer.param_groups:group['lr']=.0005*(.05+.95*.5*(1+math.cos(math.pi*epoch/max(self.epochs-1,1))))
                    for model in networks:model.train()
                    order=torch.randperm(len(x),generator=generator)
                    for start in range(0,len(x),128):
                        ids=order[start:start+128];optimizer.zero_grad(set_to_none=True)
                        predicted=torch.cat([model(x[ids]) for model in networks],dim=1)
                        if c['target']=='poisson':
                            logmu=offset[ids]+predicted.clamp(-6,6)
                            loss=((torch.exp(logmu)-actual[ids]*logmu)/mean).mean()
                        else:loss=(((predicted-expected[ids])**2)*weights[ids,None]).mean()
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite direct neural training loss')
                        loss.backward();optimizer.step()
                    if epoch>=self.epochs*3//4:
                        if average is None:average=deepcopy(networks)
                        else:
                            with torch.no_grad():
                                for old,new in zip(average,networks):
                                    for a,b in zip(old.parameters(),new.parameters()):a.add_((b-a)/(averaged+1))
                        averaged+=1
                self.networks=[n.eval() for n in average]
        finally:torch.set_num_threads(previous)
        return self

    def raw(self, rows, vectors, texts):
        x=torch.from_numpy(np.asarray(self.features.transform(rows,vectors,texts),dtype=np.float32))
        with torch.inference_mode():value=torch.cat([n(x) for n in self.networks],dim=1).numpy().astype(float)
        prior=self._prior(rows)
        if self.configuration['target']=='poisson':
            # The likelihood fits counts directly; offset=1 preserves only the
            # explicitly historical v1 artifacts, which used an extra subtraction.
            return np.maximum(0,(prior+1)*np.exp(np.clip(value,-6,6))-getattr(self,'poisson_output_offset',1.))
        value*=self.scale
        if self.configuration['target']=='rates':value*=np.asarray([r['audience'] for r in rows])[:,None]/1000.
        return prior+value

    def predict(self, rows, vectors, texts, strength=1.):
        prior=self._prior(rows);value=prior+strength*(self.raw(rows,vectors,texts)-prior)
        if not np.isfinite(value).all():raise RuntimeError('Nonfinite count forecast')
        return np.rint(np.maximum(0,value))

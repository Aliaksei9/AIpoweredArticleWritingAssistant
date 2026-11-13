"""Semantic transfer of relative engagement, with group-excluded training priors."""
from collections import defaultdict

import numpy as np

from src.assesment_conform_of_PP_text_based.dataset import text_group
from src.social_metrics_prediction.features import validate_counts
from .history_candidates import ConditionalHistoryPredictor


class CrossResourceTransfer:
    def __init__(self, configuration):self.configuration=dict(configuration)

    def _prior(self, rows):
        return np.asarray([self.base.channels.get(self.base._key(r),dict(mean=self.base.global_mean))['mean'] for r in rows])*np.asarray([r['audience'] for r in rows])[:,None]/1000.

    def fit(self, rows, counts, vectors, texts):
        counts=validate_counts(counts,len(rows));self.base=ConditionalHistoryPredictor().fit(rows,counts)
        self.split=vectors.shape[1]//2;self.vectors=np.asarray(vectors[:,:self.split],dtype=np.float32)
        self.resources=np.asarray([r['resource_id'] for r in rows]);keys=[self.base._key(r) for r in rows]
        rates=counts/np.asarray([r['audience'] for r in rows])[:,None]*1000.
        channels=defaultdict(list);groups=defaultdict(list);hashes=[text_group(t) for t in texts]
        for i,(key,group) in enumerate(zip(keys,hashes)):channels[key].append(i);groups[key,group].append(i)
        group_sum={key:rates[index].sum(axis=0) for key,index in groups.items()}
        channel_sum={key:rates[index].sum(axis=0) for key,index in channels.items()}
        prior=[]
        for i,row in enumerate(rows):
            key=keys[i];group=key,hashes[i];n=len(channels[key])-len(groups[group])
            rate=(channel_sum[key]-group_sum[group])/n if n else self.base.global_mean
            prior.append(np.maximum(0,rate)*row['audience']/1000.)
        self.relative=(counts+1)/(np.asarray(prior)+1)
        return self

    def retrieval(self, rows, vectors, exclude=None):
        exclude=self.configuration['exclude_resource'] if exclude is None else exclude
        similarity=np.asarray(vectors[:,:self.split],dtype=np.float32)@self.vectors.T
        if exclude:
            mask=np.asarray([r['resource_id'] for r in rows])[:,None]==self.resources[None,:]
            mask[np.all(mask,axis=1)]=False
            similarity[mask]=-np.inf
        k=min(50,len(self.vectors));index=np.argpartition(-similarity,k-1,axis=1)[:,:k]
        value=np.take_along_axis(similarity,index,axis=1);order=np.argsort(-value,axis=1)
        return np.take_along_axis(index,order,axis=1),np.take_along_axis(value,order,axis=1)

    def raw(self, rows, vectors, texts, retrieved=None):
        index,value=self.retrieval(rows,vectors) if retrieved is None else retrieved
        c=self.configuration;k=min(c['neighbours'],index.shape[1]);index=index[:,:k];value=value[:,:k]
        weights=np.exp((value-value[:,:1])/c['temperature'])
        relative=np.minimum(self.relative,c['ratio_cap'])
        ratios=np.sum(weights[:,:,None]*relative[index],axis=1)/weights.sum(axis=1)[:,None]
        return np.maximum(0,(self._prior(rows)+1)*ratios-1)

    def predict(self, rows, vectors, texts, strength=1.):
        prior=self._prior(rows)
        return np.rint(np.maximum(0,prior+strength*(self.raw(rows,vectors,texts)-prior)))


def configurations():
    return {f'transfer_{n}_{temp:g}_{cap:g}_{exclude}':dict(neighbours=n,temperature=temp,ratio_cap=cap,exclude_resource=exclude)
            for n in (5,20,50) for temp in (.02,.1) for cap in (10.,50.) for exclude in (False,True)}

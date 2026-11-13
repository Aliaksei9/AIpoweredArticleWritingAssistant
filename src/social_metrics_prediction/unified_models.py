"""Independent targets, identical complete input features, one current row per article."""
from collections import defaultdict
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .features import METRICS


class ChannelPrior:
    def fit(self,rows,counts):
        self.mean=(counts/np.asarray([r['audience'] for r in rows])[:,None]*1000.).mean(axis=0)
        grouped=defaultdict(list)
        for i,row in enumerate(rows):grouped[row['source'],row['resource_id']].append(i)
        self.channels={key:(counts[ids]/np.asarray([rows[i]['audience'] for i in ids])[:,None]*1000.).mean(axis=0)
                       for key,ids in grouped.items()}
        return self

    def predict(self,rows):
        return np.asarray([self.channels.get((r['source'],r['resource_id']),self.mean)*r['audience']/1000. for r in rows])


class UnifiedRateModel:
    def __init__(self,family='ridge',alpha=1000.,neighbours=20,temperature=1.,quantile=.95,context_weight=1.,same_resource=False):
        if not np.isfinite(context_weight) or context_weight<=0:raise ValueError('Every input group must retain a positive weight')
        self.family=family;self.alpha=alpha;self.neighbours=neighbours;self.temperature=temperature;self.quantile=quantile
        self.context_weight=context_weight;self.same_resource=same_resource

    def fit(self,features,rows,counts):
        self.prior=ChannelPrior().fit(rows,counts)
        audience=np.asarray([r['audience'] for r in rows],dtype=float);rates=counts/audience[:,None]*1000.
        if self.family=='ridge':
            residual=(counts-self.prior.predict(rows))/audience[:,None]*1000.
            self.scale=np.maximum(residual.std(axis=0),1.);weight=audience**2;weight/=weight.mean()
            self.model=Ridge(alpha=self.alpha).fit(features,residual/self.scale,sample_weight=weight)
        else:
            from src.database.embedding_config import EMBEDDING_DIMENSIONS
            self.input_scale=np.ones(features.shape[1],np.float32);self.input_scale[:-EMBEDDING_DIMENSIONS]=self.context_weight
            self.features=np.asarray(features,dtype=np.float32)*self.input_scale
            self.norm=np.sum(self.features*self.features,axis=1)
            grouped=defaultdict(list)
            for i,row in enumerate(rows):grouped[row['source'],row['resource_id']].append(i)
            self.channel_indices={key:np.asarray(ids) for key,ids in grouped.items()}
            self.rates=rates.copy()
            for ids in grouped.values():self.rates[ids]=np.minimum(rates[ids],np.quantile(rates[ids],self.quantile,axis=0))
        return self

    def nearest(self,features,k,rows=None):
        neighbours=[];distances=[]
        for start in range(0,len(features),128):
            query=features[start:start+128]*getattr(self,'input_scale',1.)
            distance=np.maximum(0,np.sum(query*query,axis=1)[:,None]+self.norm[None,:]-2*query@self.features.T)
            if getattr(self,'same_resource',False):
                for j,row in enumerate(rows[start:start+128]):
                    ids=self.channel_indices.get((row['source'],row['resource_id']))
                    if ids is not None:
                        mask=np.ones(len(self.features),bool);mask[ids]=False;distance[j,mask]=np.inf
            n=min(k,len(self.features));indices=np.argpartition(distance,n-1,axis=1)[:,:n]
            values=np.take_along_axis(distance,indices,axis=1);order=np.argsort(values,axis=1)
            neighbours.append(np.take_along_axis(indices,order,axis=1));distances.append(np.take_along_axis(values,order,axis=1))
        return np.vstack(neighbours),np.vstack(distances)

    def raw(self,features,rows,cache=None):
        audience=np.asarray([r['audience'] for r in rows])
        if self.family=='ridge':
            return self.prior.predict(rows)+self.model.predict(features)*self.scale*audience[:,None]/1000.
        ids,distance=self.nearest(features,self.neighbours,rows) if cache is None else cache
        ids=ids[:,:self.neighbours];distance=distance[:,:self.neighbours]
        weights=np.exp(-(distance-distance[:,:1])/self.temperature)
        rates=np.sum(weights[:,:,None]*self.rates[ids],axis=1)/weights.sum(axis=1)[:,None]
        return rates*audience[:,None]/1000.


class UnifiedMetricHead:
    def __init__(self,model,metric,strength=1.):self.model=model;self.metric=metric;self.strength=strength

    def predict(self,features,rows,*,cache=None):
        if cache is None:
            cache={}
        key=id(self.model)
        if key not in cache:
            cache[key]=self.model.prior.predict(rows),self.model.raw(features,rows)
        base,raw=cache[key]
        return np.rint(np.maximum(0,base+self.strength*(raw-base)))[:,self.metric]


def fit_heads(rows,counts,vectors,groups,*,configuration=None):
    from .unified_features import UnifiedFeatureEncoder
    encoder=UnifiedFeatureEncoder().fit(rows,vectors);features=encoder.transform(rows,vectors)
    if configuration is None:
        # Fit the encoder afresh inside each tuning fold as well.
        predictions={}
        for a,b in GroupKFold(2,shuffle=True,random_state=47).split(counts,groups=groups):
            train=[rows[i] for i in a];valid=[rows[i] for i in b]
            inner=UnifiedFeatureEncoder().fit(train,vectors[a]);xa=inner.transform(train,vectors[a]);xb=inner.transform(valid,vectors[b])
            for alpha in (10.,1000.,100000.):
                params=dict(family='ridge',alpha=alpha);model=UnifiedRateModel(**params).fit(xa,train,counts[a])
                prior=model.prior.predict(valid);raw=model.raw(xb,valid)
                for strength in (.1,.25,.5,1.):
                    key=json.dumps(dict(params,strength=strength),sort_keys=True)
                    predictions.setdefault(key,np.zeros_like(counts))[b]=np.rint(np.maximum(0,prior+strength*(raw-prior)))
            for context_weight,same_resource in [(1.,False),(.01,True),(.03,True),(.1,True),(.3,True),(1.,True)]:
                nearest=None
                for quantile in (.85,.95,1.):
                    model=UnifiedRateModel(family='neighbours',quantile=quantile,context_weight=context_weight,same_resource=same_resource).fit(xa,train,counts[a])
                    if nearest is None:nearest=model.nearest(xb,40,valid)
                    prior=model.prior.predict(valid)
                    for neighbours in (5,20,40):
                        for temperature in (.01,.03,.06,.25,1.,4.):
                            model.neighbours=neighbours;model.temperature=temperature;raw=model.raw(xb,valid,nearest)
                            for strength in (.5,1.,1.5):
                                params=dict(family='neighbours',quantile=quantile,neighbours=neighbours,temperature=temperature,strength=strength,context_weight=context_weight,same_resource=same_resource)
                                key=json.dumps(params,sort_keys=True)
                                predictions.setdefault(key,np.zeros_like(counts))[b]=np.rint(np.maximum(0,prior+strength*(raw-prior)))
        errors={key:np.mean((value-counts)**2,axis=0) for key,value in predictions.items()}
        configuration={name:json.loads(min(errors,key=lambda key:errors[key][i])) for i,name in enumerate(METRICS)}
        selection=dict(configuration=configuration,inner_mse={name:float(min(v[i] for v in errors.values())) for i,name in enumerate(METRICS)})
        # Reactions have their own finer search and independent target loss.
        from .reaction_selection import select_reaction_configuration
        configuration['reactions'],audit=select_reaction_configuration(rows,counts,vectors,groups)
        selection['reaction_refinement']=audit
        selection['inner_mse']['reactions']=audit['inner_mse']
    else:selection=dict(configuration=configuration)
    models={};heads=[]
    for i,name in enumerate(METRICS):
        params=dict(configuration[name]);strength=params.pop('strength',1.);key=json.dumps(params,sort_keys=True)
        if key not in models:models[key]=UnifiedRateModel(**params).fit(features,rows,counts)
        heads.append(UnifiedMetricHead(models[key],i,strength))
    return encoder,heads,selection

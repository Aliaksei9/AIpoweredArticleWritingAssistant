"""Research alternatives: native categorical boosting and direct text features."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.social_metrics_prediction.features import MetricFeatureEncoder,input_columns,validate_counts
from .history_candidates import ConditionalHistoryPredictor


class AlternativeFeatures:
    def __init__(self, mode):self.mode=mode

    def fit(self, rows, vectors, texts):
        self.encoder=MetricFeatureEncoder().fit(rows)
        base=self.encoder.transform(rows);self.numeric=StandardScaler().fit(base[:,-2:])
        self.split=vectors.shape[1]//2
        if self.mode in ('word','char'):
            self.tfidf=TfidfVectorizer(analyzer='word' if self.mode=='word' else 'char_wb',
                ngram_range=(1,2) if self.mode=='word' else (3,5),max_features=18000,
                min_df=3,sublinear_tf=True,dtype=np.float32).fit(texts)
        if self.mode in ('dense','pca','native_pca'):
            self.document=StandardScaler().fit(vectors[:,:self.split])
            if self.mode!='dense':
                self.pca=PCA(n_components=min(32,len(rows)-1,self.split),whiten=True,
                             svd_solver='randomized',random_state=42).fit(self.document.transform(vectors[:,:self.split]))
        return self

    def transform(self, rows, vectors, texts):
        base=self.encoder.transform(rows);numeric=self.numeric.transform(base[:,-2:])
        matrix=np.column_stack([base[:,:-2],numeric])
        if self.mode in ('native','native_pca'):
            categories,_=input_columns(rows)
            matrix=np.column_stack([categories,numeric])
        if self.mode in ('word','char'):
            return sparse.hstack([sparse.csr_matrix(matrix),self.tfidf.transform(texts)],format='csr',dtype=np.float64)
        if self.mode in ('dense','pca','native_pca'):
            doc=self.document.transform(vectors[:,:self.split])
            if self.mode!='dense':doc=self.pca.transform(doc)
            matrix=np.column_stack([matrix,doc])
        return matrix


class AlternativeRegressor:
    """Predict residuals in counts/rates, or positive Poisson count means.

    All priors, vocabularies, scaling and components use training rows only.
    A single fit holds three mathematically independent target regressions.
    """
    def __init__(self, configuration):self.configuration=dict(configuration)

    def _prior(self, rows):
        return np.asarray([self.base.channels.get(self.base._key(r),dict(mean=self.base.global_mean))['mean']
                           for r in rows])*np.asarray([r['audience'] for r in rows])[:,None]/1000.

    def fit(self, rows, counts, vectors, texts, feature_builder=None, feature_matrix=None):
        c=self.configuration;counts=validate_counts(counts,len(rows))
        self.base=ConditionalHistoryPredictor().fit(rows,counts)
        self.features=feature_builder or AlternativeFeatures(c['features']).fit(rows,vectors,texts)
        x=self.features.transform(rows,vectors,texts) if feature_matrix is None else feature_matrix
        prior=self._prior(rows)
        audience=np.asarray([r['audience'] for r in rows],dtype=float)
        if c.get('target','counts')=='rates':
            target=(counts-prior)/audience[:,None]*1000.;weight=audience**2
        else:target=counts-prior;weight=np.ones(len(rows))
        self.positive=c.get('loss')=='Poisson'
        if self.positive:target=counts.copy()
        self.scale=np.maximum(np.std(target,axis=0),1.)
        target/=self.scale;weight/=np.mean(weight)
        if c['family']=='ridge':
            self.model=Ridge(alpha=c['alpha'],solver='lsqr' if sparse.issparse(x) else 'auto',
                             tol=1e-5,max_iter=300).fit(x,target,sample_weight=weight)
        else:
            from catboost import CatBoostRegressor
            self.model=[]
            for metric in range(3):
                model=CatBoostRegressor(iterations=300,depth=c['depth'],learning_rate=.04,
                    l2_leaf_reg=30.,loss_function=c.get('loss','RMSE'),verbose=False,
                    random_seed=42+metric,thread_count=4,allow_writing_files=False,
                    cat_features=[0,1,2],boosting_type='Ordered')
                model.fit(x,target[:,metric],sample_weight=weight);self.model.append(model)
        return self

    def raw(self, rows, vectors, texts):
        x=self.features.transform(rows,vectors,texts)
        if self.configuration['family']=='ridge':value=self.model.predict(x)
        else:
            value=np.column_stack([m.predict(x,prediction_type='Exponent' if self.positive else 'RawFormulaVal') for m in self.model])
        value=value*self.scale
        if not self.positive:
            if self.configuration.get('target','counts')=='rates':
                value*=np.asarray([r['audience'] for r in rows])[:,None]/1000.
            value+=self._prior(rows)
        return value

    def predict(self, rows, vectors, texts, strength=1.):
        prior=self._prior(rows);raw=self.raw(rows,vectors,texts)
        value=prior+strength*(raw-prior)
        if not np.isfinite(value).all():raise RuntimeError('Nonfinite alternative count forecast')
        return np.rint(np.maximum(0,value))


class SemanticNeighbour:
    """Transfer training counts from semantically similar posts in the resource."""
    def __init__(self, configuration):self.configuration=dict(configuration)

    def fit(self, rows, counts, vectors, texts, feature_builder=None):
        counts=validate_counts(counts,len(rows));input_columns(rows)
        self.base=ConditionalHistoryPredictor(upper_quantile=.95).fit(rows,counts)
        self.split=vectors.shape[1]//2;self.channels={}
        for key in {self.base._key(r) for r in rows}:
            ids=np.asarray([i for i,r in enumerate(rows) if self.base._key(r)==key])
            rates=counts[ids]/np.asarray([rows[i]['audience'] for i in ids])[:,None]*1000.
            rates=np.minimum(rates,np.quantile(rates,.95,axis=0))
            self.channels[key]=dict(vectors=vectors[ids,:self.split],rates=rates)
        return self

    def _prior(self, rows):
        return np.asarray([self.base.channels.get(self.base._key(r),dict(mean=self.base.global_mean))['mean']
                           for r in rows])*np.asarray([r['audience'] for r in rows])[:,None]/1000.

    def raw(self, rows, vectors, texts):
        result=[];c=self.configuration
        for row,vector in zip(rows,vectors):
            channel=self.channels.get(self.base._key(row))
            if channel is None:rate=self.base.global_mean
            else:
                cosine=channel['vectors']@vector[:self.split]
                k=min(c['neighbours'],len(cosine));ids=np.argsort(cosine)[-k:]
                weights=np.exp((cosine[ids]-cosine[ids].max())/c['temperature'])
                rate=np.average(channel['rates'][ids],axis=0,weights=weights)
            result.append(rate*row['audience']/1000.)
        return np.asarray(result)

    def predict(self, rows, vectors, texts, strength=1.):
        prior=self._prior(rows)
        return np.rint(np.maximum(0,prior+strength*(self.raw(rows,vectors,texts)-prior)))


class AlternativeMetricHead:
    """Serializable independent selection of one metric and one blend strength."""
    def __init__(self, model, metric, strength):
        self.model=model;self.metric=metric;self.strength=strength

    def predict(self, rows, vectors, texts):
        return self.model.predict(rows,vectors,texts,strength=self.strength)[:,self.metric]


class ReferenceMetricHead:
    def __init__(self, head):self.head=head

    def predict(self, rows, vectors=None, texts=None):return self.head.predict(rows)


def predict_heads(heads, rows, vectors, texts):
    """Reuse preprocessing for independent targets sharing the same fitted core."""
    cache={};values=[]
    for name in ('views','reactions','comments'):
        head=heads[name]
        if isinstance(head,AlternativeMetricHead):
            key=id(head.model)
            if key not in cache:cache[key]=(head.model._prior(rows),head.model.raw(rows,vectors,texts))
            prior,raw=cache[key]
            values.append(np.rint(np.maximum(0,prior+head.strength*(raw-prior)))[:,head.metric])
        elif hasattr(head,'pipeline'):
            key=id(head.pipeline)
            if key not in cache:cache[key]=head.pipeline.predict(rows,vectors,texts)
            values.append(cache[key][:,head.metric])
        else:values.append(head.predict(rows,vectors,texts))
    return np.column_stack(values)


def configurations():
    result={}
    for mode in ('dense','pca','word','char'):
        for target in ('counts','rates'):
            for alpha in ((100.,10000.,1000000.) if mode in ('dense','pca') else (1.,30.,1000.)):
                result[f'{mode}_{target}_{alpha:g}']=dict(family='ridge',features=mode,target=target,alpha=alpha)
    for mode in ('native','native_pca'):
        for loss in ('RMSE','Poisson'):
            for depth in (4,6):
                result[f'{mode}_{loss}_{depth}']=dict(family='cat',features=mode,target='counts',loss=loss,depth=depth)
    for neighbours in (10,40):
        for temperature in (.01,.05,.2):
            result[f'semantic_{neighbours}_{temperature:g}']=dict(family='neighbour',neighbours=neighbours,temperature=temperature)
    return result

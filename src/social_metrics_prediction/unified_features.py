"""One feature matrix shared by all three social metric models."""
import numpy as np
from sklearn.preprocessing import StandardScaler

from .features import MetricFeatureEncoder
from .forecast_context import utc_datetime,validate_horizon,PUBLICATION_TIMEZONE
from src.database.embedding_config import EMBEDDING_DIMENSIONS,EMBEDDING_SIGNATURE

FEATURE_VERSION='shared-text-topic-resource-score-publication-age-v1'
INPUT_FIELDS=('document_embedding','topic','score','resource_id','source','audience','published_at','horizon_days')


def document_vectors(values,length):
    vectors=np.asarray(values,dtype=np.float32) if values is not None else np.empty((0,0))
    if vectors.shape!=(length,EMBEDDING_DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError(f'Document embeddings must have shape (n, {EMBEDDING_DIMENSIONS})')
    norm=np.linalg.norm(vectors.astype(float),axis=1)
    if np.any(norm<=0):raise ValueError('Document embeddings must be nonzero')
    return (vectors/norm[:,None]).astype(np.float32)


class UnifiedFeatureEncoder:
    input_fields=INPUT_FIELDS

    def _raw(self,rows,vectors):
        base=self.categories.transform(rows);calendar=[]
        for row in rows:
            date=utc_datetime(row['published_at']).astimezone(PUBLICATION_TIMEZONE)
            hour=date.hour+date.minute/60.;age=validate_horizon(row['horizon_days'])/14.
            calendar.append([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                np.sin(2*np.pi*date.weekday()/7),np.cos(2*np.pi*date.weekday()/7),age,age**2,age**3])
        calendar=np.asarray(calendar)
        interaction=(base[:,:-2,None]*calendar[:,None,:]).reshape(len(rows),-1)
        return np.column_stack([base,calendar,interaction,vectors])

    def fit(self,rows,vectors):
        self.categories=MetricFeatureEncoder().fit(rows)
        raw=self._raw(rows,document_vectors(vectors,len(rows)))
        self.scaler=StandardScaler().fit(raw);self.n_features_in_=raw.shape[1]
        self.document_representation='unit'
        self.embedding_signature=EMBEDDING_SIGNATURE
        return self

    def transform(self,rows,vectors):
        raw=self._raw(rows,document_vectors(vectors,len(rows)))
        result=self.scaler.transform(raw)
        # Keep the whole document block present without making 384 coordinates
        # outweigh every scalar solely because of dimensionality.
        if getattr(self,'document_representation','standardized')=='unit':
            result[:,-EMBEDDING_DIMENSIONS:]=raw[:,-EMBEDDING_DIMENSIONS:]
        else:result[:,-EMBEDDING_DIMENSIONS:]/=np.sqrt(EMBEDDING_DIMENSIONS)
        return result.astype(np.float32)

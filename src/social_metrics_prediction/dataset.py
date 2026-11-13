"""Training data from one current metric row per article."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

from src.database.connection import connect_database
from src.database.table_classes import ArticleTextTable,ResourceTable
from src.assesment_conform_of_PP_text_based.dataset import dataset_from_rows,TrainingDataset
from .features import METRICS,validate_counts
from .resources import ResourceInfo,normalize_source
from .forecast_context import observation_horizon,validate_horizon


@dataclass
class MetricDataset:
    text: TrainingDataset
    rows: list[dict]
    counts: np.ndarray
    resources: list[ResourceInfo]
    publication_times: list[str | None] | None = None
    collection_times: list[str | None] | None = None
    horizons: list[float] | None = None
    feature_mode: str = 'unified'

    @property
    def groups(self):return self.text.groups

    @property
    def scores(self):return self.text.scores

    @property
    def fingerprint(self):
        payload=[self.text.fingerprint,self.rows,self.counts.tolist(),self.publication_times,self.collection_times,self.horizons]
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

    def inputs(self,indices,scores):
        if len(indices)!=len(scores):raise ValueError('Score count does not match requested rows')
        result=[dict(self.rows[int(i)],score=float(score)) for i,score in zip(indices,scores)]
        if self.publication_times is not None:
            for row,i in zip(result,indices):
                row['published_at']=self.publication_times[int(i)]
                if self.horizons is not None:row['horizon_days']=self.horizons[int(i)]
        return result


def metric_dataset_from_rows(rows,resources,*,feature_mode='unified'):
    if feature_mode not in ('unified','legacy'):raise ValueError('Unknown social feature mode')
    ids=[str(r['article_id']) for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Expected one current metric row per article ID')
    original=len(rows);excluded=0
    if feature_mode=='unified':
        accepted=[]
        for row in rows:
            try:validate_horizon(observation_horizon(row.get('published_at'),row.get('collected_at')))
            except (TypeError,ValueError):excluded+=1;continue
            accepted.append(row)
        rows=accepted
    text=dataset_from_rows(rows)
    rows_by_id={str(row['article_id']):row for row in rows}
    accepted=[rows_by_id[article_id] for article_id in text.article_ids]
    inputs=[dict(topic=r['topic'].strip(),resource_id=str(r['resource_id']),source=normalize_source(r['source']),audience=int(r['audience'])) for r in accepted]
    if any(r['resource_id']=='None' for r in inputs):raise ValueError('Training article has no resource')
    counts=validate_counts([[r[m] for m in METRICS] for r in accepted],len(accepted))
    def stamp(value):return value.isoformat() if hasattr(value,'isoformat') else value
    dates=[stamp(r.get('published_at')) for r in accepted];collected=[stamp(r.get('collected_at')) for r in accepted]
    ages=[observation_horizon(p,c) for p,c in zip(dates,collected)] if feature_mode=='unified' else None
    text.report.update(source_rows=original,excluded_time_context=excluded,one_row_per_article=True)
    return MetricDataset(text,inputs,counts,resources,dates,collected,ages,feature_mode)


async def load_metric_dataset(dsn,*,feature_mode='unified'):
    async with await connect_database(dsn) as conn:
        rows=await ArticleTextTable(conn).list_training_data(include_resource_metadata=True)
        catalog=await ResourceTable(conn).list_prediction_resources()
    resources=[ResourceInfo.from_database(r) for r in catalog if r['reader_count'] and r['reader_count']>0]
    return metric_dataset_from_rows(rows,resources,feature_mode=feature_mode)

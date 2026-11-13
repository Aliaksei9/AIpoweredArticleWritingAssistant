"""Check the complete research pipeline, including SBERT, through a private API."""
import argparse
import asyncio
from datetime import datetime,timezone
import json
from pathlib import Path
from types import SimpleNamespace
import time

import httpx
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based import TextTopicEncoder
from src.controller.editor_controller import EditorController,EditorSettings
from src.database.model_embedding import TextEmbedder
from src.database.topic_detector import ArticleTopicDetector
from src.social_metrics_prediction import ResourceInfo
from src.ui.app import create_app
from .alternative_metric_models import predict_heads
from .benchmark_utils import representative_cases,summarize
from .compare_text_models import write_json


class AlternativeEngine:
    def __init__(self, embedder, detector, encoder, catalog, text, heads):
        self.embedder=embedder;self.detector=detector;self.encoder=encoder
        self.social=SimpleNamespace(resources=catalog);self.text=text;self.heads=heads

    def evaluate(self, text, topic, resource):
        vector=self.embedder.embed_document(text);topic=topic or self.detector.detect_from_embedding(vector)
        features=self.encoder.transform([text],[topic],[vector]);score=float(self.text.predict(features)[0])
        row=dict(topic=topic,score=score,resource_id=resource.id,source=resource.source,audience=resource.audience)
        row.update(published_at=datetime.now(timezone.utc).isoformat(),horizon_days=7.)
        value=predict_heads(self.heads,[row],features,[text])[0]
        return dict(topic=topic,score=score,metrics=dict(zip(('views','reactions','comments'),map(int,value))))


async def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--text-study',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--texts',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    data=np.load(args.corpus/'corpus.npz');records=json.loads(Path('model_tests/results/history_20261111/control_snapshot.json').read_text(encoding='utf8'))['articles']
    rows=json.loads((args.corpus/'rows.json').read_text(encoding='utf8'));texts=json.loads(args.texts.read_text(encoding='utf8'))
    expected=np.load(args.study/'predictions.npz')['predicted'];hits=np.zeros(len(rows),int)
    old_asof=json.loads(Path('model_tests/results/expansion_20261111/collection.json').read_text(encoding='utf8'))['as_of']
    control_asof=json.loads(Path('model_tests/results/history_20261111/control_protocol.json').read_text(encoding='utf8'))['created_at']
    asofs=[old_asof]*11493+[control_asof]*216
    for fold in range(1,6):
        prepared=joblib.load(args.previous/f'fold_{fold}'/'prepared.joblib');test=prepared['test']
        heads=joblib.load(args.study/f'fold_{fold}'/'heads.joblib')
        validation=[dict(r,horizon_days=(datetime.fromisoformat(asofs[i])-datetime.fromisoformat(r['published_at'])).total_seconds()/86400.)
                    for r,i in zip(prepared['test_rows'],test)]
        actual=predict_heads(heads,validation,data['features'][test],[texts[i] for i in test])
        np.testing.assert_array_equal(actual,expected[test]);hits[test]+=1
    assert np.all(hits==1)
    catalog=list({r['resource_id']:ResourceInfo(r['resource_id'],r['resource_id'],r['audience'],r['source']) for r in rows}.values())
    cases=representative_cases([r['text'] for r in records],range(len(records)),maximum=12)
    torch.set_num_threads(4);t=time.perf_counter();embedder=TextEmbedder();encoder=TextTopicEncoder(embedder=embedder);detector=ArticleTopicDetector(embedder)
    for topic in detector.topics:encoder.topic_embedding(topic.name)
    detector.detect_from_embedding(embedder.embed_document(records[cases[0]]['text']));loading=time.perf_counter()-t
    durations=[]
    for fold in range(1,6):
        heads=joblib.load(args.study/f'fold_{fold}'/'heads.joblib');text=joblib.load(args.text_study/f'fold_{fold}'/'predictor.joblib')
        async def resources():return catalog
        engine=AlternativeEngine(embedder,detector,encoder,catalog,text,heads)
        app=create_app(EditorController(EditorSettings(root=root),engine=engine,resource_loader=resources))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
                async def predict(value,record):
                    response=await client.post('/api/predictions',json=dict(text=value,topic=None,resource_id=record['resource_id'],source=record['source']))
                    response.raise_for_status();result=response.json()
                    assert -10<=result['score']<=10 and all(isinstance(v,int) and v>=0 for v in result['metrics'].values())
                await predict(records[cases[0]]['text'],records[cases[0]])
                for index in cases:
                    r=records[index];t=time.perf_counter();await predict(r['text'],r)
                    durations.append(dict(fold=fold,characters=len(r['text']),stress=False,seconds=time.perf_counter()-t))
                base=records[cases[-1]]['text'];stress=((base+'\n')*(50000//(len(base)+1)+1))[:50000]
                t=time.perf_counter();await predict(stress,records[cases[-1]])
                durations.append(dict(fold=fold,characters=50000,stress=True,seconds=time.perf_counter()-t))
        print('Latency fold '+str(fold)+' complete',flush=True)
    values=[r['seconds'] for r in durations]
    report=dict(status='passed' if max(values)<=10 else 'too_slow',loading_seconds=loading,cases=durations,
        response_time=dict(summarize(values),max=max(values)),all_11709_saved_predictions_exactly_reproduced=True,
        default_publication='request time',default_horizon_days=7,
        scope='Private ASGI controller, full SBERT, auto topic, neural assessment, alternative metric heads and JSON; warm; no LLM/network/DB')
    write_json(args.study/'latency.json',report);print(json.dumps(dict(status=report['status'],maximum_seconds=max(values))))


if __name__=='__main__':
    with threadpool_limits(limits=4):asyncio.run(main())

"""Time the experimental pair through a private local ASGI app, without deployment."""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import time

from dotenv import load_dotenv
import httpx
import joblib
import numpy as np
import torch

from src.assesment_conform_of_PP_text_based import TextTopicEncoder
from src.controller.editor_controller import EditorController,EditorSettings
from src.database.model_embedding import TextEmbedder
from src.database.topic_detector import ArticleTopicDetector
from src.social_metrics_prediction import ResourceInfo
from src.ui.app import create_app
from .benchmark_utils import representative_cases,summarize
from .compare_text_models import write_json


class ResearchEngine:
    def __init__(self, embedder, detector, encoder, catalog, text, heads):
        self.embedder=embedder;self.detector=detector;self.encoder=encoder
        self.social=SimpleNamespace(resources=catalog);self.text=text;self.heads=heads

    def evaluate(self, text, topic, resource):
        vector=self.embedder.embed_document(text)
        topic=topic or self.detector.detect_from_embedding(vector)
        features=self.encoder.transform([text],[topic],[vector])
        score=float(self.text.predict(features)[0])
        row=dict(topic=topic,score=score,resource_id=resource.id,source=resource.source,audience=resource.audience)
        metrics={name:int(head.predict([row])[0]) for name,head in self.heads.items()}
        return dict(topic=topic,score=score,metrics=metrics)


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--text-study',type=Path,required=True)
    parser.add_argument('--social-study',type=Path,required=True)
    parser.add_argument('--records',type=Path,default=Path('model_tests/results/history_20261111/control_snapshot.json'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];load_dotenv(root/'.env')
    rows=json.loads((args.corpus/'rows.json').read_text(encoding='utf8'))
    records=json.loads(args.records.read_text(encoding='utf8'))['articles']
    unique={r['resource_id']:ResourceInfo(r['resource_id'],r['resource_id'],r['audience'],r['source']) for r in rows}
    catalog=list(unique.values());texts=[r['text'] for r in records]
    cases=representative_cases(texts,np.arange(len(texts)),maximum=12)
    torch.set_num_threads(4);started=time.perf_counter()
    embedder=TextEmbedder();encoder=TextTopicEncoder(embedder=embedder);detector=ArticleTopicDetector(embedder)
    for topic in detector.topics:encoder.topic_embedding(topic.name)
    detector.detect_from_embedding(embedder.embed_document(texts[cases[0]]))
    loading=time.perf_counter()-started;all_results=[]
    for fold in range(1,6):
        text=joblib.load(args.text_study/f'fold_{fold}'/'predictor.joblib')
        heads=joblib.load(args.social_study/f'fold_{fold}'/'social_heads.joblib')
        engine=ResearchEngine(embedder,detector,encoder,catalog,text,heads)
        async def resources():return catalog
        controller=EditorController(EditorSettings(root=root),engine=engine,resource_loader=resources)
        app=create_app(controller)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
                async def predict(value,record):
                    r=await client.post('/api/predictions',json=dict(text=value,topic=None,resource_id=record['resource_id'],source=record['source']))
                    r.raise_for_status();result=r.json()
                    assert -10<=result['score']<=10 and all(isinstance(v,int) and v>=0 for v in result['metrics'].values())
                await predict(texts[cases[0]],records[cases[0]])
                for index in cases:
                    t=time.perf_counter();await predict(texts[index],records[index])
                    all_results.append(dict(fold=fold,characters=len(texts[index]),seconds=time.perf_counter()-t,stress=False))
                base=texts[cases[-1]];stress=((base+'\n')*(50000//(len(base)+1)+1))[:50000]
                t=time.perf_counter();await predict(stress,records[cases[-1]])
                all_results.append(dict(fold=fold,characters=50000,seconds=time.perf_counter()-t,stress=True))
        print(f'Latency fold {fold}/5 max {max(r["seconds"] for r in all_results if r["fold"]==fold):.3f}s',flush=True)
    durations=[r['seconds'] for r in all_results]
    report=dict(status='passed' if max(durations)<=10 else 'too_slow',cases=all_results,
        response_time=dict(summarize(durations),max=max(durations)),loading_seconds=loading,
        scope='Private ASGI editor/controller, all five research pairs, full document SBERT, automatic topic, neural score, separate social heads and JSON; warm, no network/DB/LLM',
        maximum_seconds=10,production_models_unchanged=True)
    write_json(args.social_study/'latency.json',report)
    print(json.dumps(dict(status=report['status'],maximum_seconds=max(durations))),flush=True)
    if report['status']!='passed':raise SystemExit(2)


if __name__=='__main__':asyncio.run(main())

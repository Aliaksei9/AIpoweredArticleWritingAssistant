"""Measure the complete local prediction path, including E5 and topic detection."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
import numpy as np
import httpx

from src.controller.editor_controller import EditorController, EditorSettings, ModelEngine
from src.ui.app import create_app
from src.social_metrics_prediction.dataset import load_metric_dataset
from .benchmark_utils import representative_cases, summarize


async def main():
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-response-seconds',type=float,default=10)
    args=parser.parse_args()
    dataset=await load_metric_dataset(os.environ['DATABASE_URL'])
    started=time.perf_counter()
    engine=ModelEngine(EditorSettings(root=root))
    loading_seconds=time.perf_counter()-started
    resources={r.id:r for r in dataset.resources}
    indices=representative_cases(dataset.text.texts,np.arange(len(dataset.scores)),maximum=12)
    cases=[dict(text=dataset.text.texts[i],resource=resources[str(dataset.rows[i]['resource_id'])],topic=None) for i in indices]
    base=cases[-1]['text']
    cases.append(dict(text=((base+'\n')*(50000//(len(base)+1)+1))[:50000],resource=cases[-1]['resource'],topic=None))
    records=[]
    controller=EditorController(EditorSettings(root=root),engine=engine)
    app=create_app(controller)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
            async def predict(case):
                response=await client.post('/api/predictions',json=dict(text=case['text'],topic=None,
                    resource_id=case['resource'].id,source=case['resource'].source))
                response.raise_for_status()
                return response.json()
            first_started=time.perf_counter()
            await predict(cases[0])
            first_seconds=time.perf_counter()-first_started
            for index,case in enumerate(cases):
                started=time.perf_counter()
                result=await predict(case)
                seconds=time.perf_counter()-started
                assert np.isfinite(result['score']) and -10<=result['score']<=10
                assert all(isinstance(x,int) and x>=0 for x in result['metrics'].values())
                records.append(dict(characters=len(case['text']),seconds=seconds,topic=result['topic'],
                                    synthetic_maximum_length=index==len(cases)-1))
    durations=[r['seconds'] for r in records]
    report=dict(checked_at=datetime.now(timezone.utc).isoformat(),status='passed' if max(durations)<=args.max_response_seconds else 'too_slow',
        model_loading_seconds=loading_seconds,first_prediction_seconds=first_seconds,
        response_time=dict(summarize(durations),max=max(durations)),cases=records,
        max_response_seconds=args.max_response_seconds,scope='warm UI predictions API via local ASGI; E5, automatic topic, module 3/4, JSON; no LLM/external network',
        database_rows=len(dataset.scores),distinct_texts=len(set(dataset.groups)),database_resources=len(dataset.resources),
        text_model_training_rows=engine.text.metadata.get('training_rows'),
        social_model_training_rows=engine.social.metadata.get('training_rows'),
        model_resources=len(engine.social.resources),
        text_model_sha256=hashlib.sha256((root/'artifacts/text_models/neural_network.joblib').read_bytes()).hexdigest(),
        social_model_sha256=hashlib.sha256((root/'artifacts/social_metrics/model.joblib').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if report['status']!='passed': raise SystemExit(2)


if __name__=='__main__':
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform=='win32' else None) as runner:
        runner.run(main())

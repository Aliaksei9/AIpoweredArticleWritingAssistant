"""Measure the candidate through the editor API with a frozen resource catalog."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx
import numpy as np

from src.controller.editor_controller import EditorController,EditorSettings,ModelEngine
from src.ui.app import create_app
from .benchmark_utils import representative_cases,summarize
from .compare_text_models import write_json


async def main():
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--model-file',type=Path,required=True)
    args=parser.parse_args()
    snapshot=json.loads((args.directory/'control_snapshot.json').read_text(encoding='utf8'))
    records=snapshot['articles']
    texts=[r['text'] for r in records]
    cases=representative_cases(texts,np.arange(len(texts)),maximum=12)
    settings=EditorSettings(root=root,social_model=args.model_file)
    started=time.perf_counter();engine=ModelEngine(settings);loading=time.perf_counter()-started
    catalog={r.id:r for r in engine.social.resources}
    async def resource_loader():return list(catalog.values())
    controller=EditorController(settings,engine=engine,resource_loader=resource_loader)
    app=create_app(controller)
    results=[]
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
            async def predict(text,record):
                response=await client.post('/api/predictions',json=dict(text=text,topic=None,
                    resource_id=record['resource_id'],source='TG'))
                response.raise_for_status()
                data=response.json()
                assert np.isfinite(data['score']) and all(isinstance(x,int) and x>=0 for x in data['metrics'].values())
            await predict(texts[cases[0]],records[cases[0]])
            for index in cases:
                started=time.perf_counter();await predict(texts[index],records[index])
                results.append(dict(characters=len(texts[index]),seconds=time.perf_counter()-started,stress=False))
            base=texts[cases[-1]]
            stress=((base+'\n')*(50000//(len(base)+1)+1))[:50000]
            started=time.perf_counter();await predict(stress,records[cases[-1]])
            results.append(dict(characters=50000,seconds=time.perf_counter()-started,stress=True))
    durations=[r['seconds'] for r in results]
    report=dict(status='passed' if max(durations)<=10 else 'too_slow',loading_seconds=loading,
        response_time=dict(summarize(durations),max=max(durations)),cases=results,max_response_seconds=10,
        scope='warm local ASGI editor API, E5, automatic topic, module 3/4 and JSON; frozen training resource catalog; no LLM/network/DB',
        text_model_sha256=hashlib.sha256(settings.path(settings.text_model).read_bytes()).hexdigest(),
        social_model_sha256=hashlib.sha256(settings.path(settings.social_model).read_bytes()).hexdigest())
    write_json(args.directory/'api_latency.json',report)
    print(json.dumps(report),flush=True)
    if report['status']!='passed':raise SystemExit(2)


if __name__=='__main__':asyncio.run(main())

"""Whole local inference through an owned ASGI fixture, including full-text SBERT."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import httpx
import joblib
from threadpoolctl import threadpool_limits
from src.controller.editor_controller import EditorController,EditorSettings,ModelEngine
from src.ui.app import create_app
from .benchmark_utils import representative_cases,summarize
from .compare_text_models import write_json


async def main(args):
    data=joblib.load(args.corpus);settings=EditorSettings(text_model=args.text.resolve(),social_model=args.social.resolve())
    begin=time.perf_counter();engine=ModelEngine(settings)
    async def resources():return engine.social.resources
    controller=EditorController(settings,engine=engine,resource_loader=resources);app=create_app(controller)
    cases=representative_cases(data['texts'],range(len(data['rows'])),maximum=12);measurements=[]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',timeout=60) as client:
        async def request(i,text=None,auto=False):
            row=data['rows'][i]
            return await client.post('/api/predictions',json=dict(text=text or data['texts'][i],topic=None if auto else row['topic'],
                resource_id=row['resource_id'],source=row['source'],published_at='2026-11-12T18:00:00+03:00'))
        response=await request(cases[0]);assert response.status_code==200,response.text;startup=time.perf_counter()-begin
        for j,i in enumerate(cases):
            start=time.perf_counter();response=await request(i,auto=j%2==0);duration=time.perf_counter()-start
            assert response.status_code==200,response.text
            assert response.json()['forecast']['horizon_sensitive_metrics']==['views','reactions','comments']
            assert response.json()['forecast']['horizon_days']==7.
            measurements.append(dict(kind='corpus',characters=len(data['texts'][i]),seconds=duration))
        body=('Разработчики обновили игру, добавили новые карты и улучшили сетевой режим. '*1000)[:50000]
        for _ in range(3):
            start=time.perf_counter();response=await request(cases[0],text=body,auto=True);duration=time.perf_counter()-start
            assert response.status_code==200,response.text
            measurements.append(dict(kind='50000_chars',characters=len(body),seconds=duration))
    await app.state.jobs.aclose();await app.state.language_jobs.aclose();await controller.aclose()
    result=dict(status='passed',measurements=measurements,warm=summarize([r['seconds'] for r in measurements]),
        warm_max_seconds=max(r['seconds'] for r in measurements),within_ten_seconds=all(r['seconds']<=10 for r in measurements),
        startup_and_first_request_seconds=startup,shared_input_fields=engine.social.shared_input_fields,
        text_sha256=hashlib.sha256(args.text.read_bytes()).hexdigest(),social_sha256=hashlib.sha256(args.social.read_bytes()).hexdigest())
    write_json(args.output,result);print(json.dumps(result),flush=True)
    if not result['within_ten_seconds']:raise RuntimeError('Warm response limit exceeded')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','text','social','output'):p.add_argument('--'+name,type=Path,required=True)
    with threadpool_limits(limits=4):asyncio.run(main(p.parse_args()),loop_factory=asyncio.SelectorEventLoop)

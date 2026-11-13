"""Read unused public channels for one additional, frozen-model confirmation.

This script writes a local experimental snapshot only. It does not change
the application's database, join channels, or post messages.
"""
import argparse
import asyncio
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import sys
from uuid import NAMESPACE_URL,uuid5

from dotenv import load_dotenv
import joblib
import numpy as np
import torch
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.functions.channels import GetFullChannelRequest
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based.dataset import text_group
from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.database.resources.telegram.collector import reaction_count
from src.database.topic_detector import ArticleTopicDetector
from .discover_channels import CachedEntityClient,write_json


async def main():
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--target',type=int,default=1500)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    previous=root/'model_tests/results/expansion_20261111'
    screening=json.loads((previous/'channel_screening.json').read_text(encoding='utf8'))
    collection=json.loads((previous/'collection.json').read_text(encoding='utf8'))
    frozen=json.loads((args.directory/'fresh_choices_frozen.json').read_text(encoding='utf8'))
    assert frozen['cv_max_ratios'][frozen['selected']]<1
    used={r['username'].lower() for r in collection['resources'] if r['status']=='complete'}
    candidates=sorted([r for r in screening['channels'] if r.get('qualifies') and r['username'].lower() not in used],
                      key=lambda r:-r.get('estimated_window_posts',0))
    with np.load(previous/'social_metrics/dataset.npz') as data:old_groups=set(data['groups'])
    path=args.directory/'fresh_snapshot.json'
    if path.exists() and not args.resume:raise ValueError('Use --resume or another output')
    report=json.loads(path.read_text(encoding='utf8')) if path.exists() else dict(status='collecting',channels=[],articles=[],
        as_of=collection['as_of'],excluded_old_text_groups=len(old_groups),frozen_configuration=frozen['selected'])
    done={r['username'].lower() for r in report['channels']}
    seen=old_groups|{r['group'] for r in report['articles']}
    as_of=datetime.fromisoformat(report['as_of'])
    oldest,newest=as_of-timedelta(days=14),as_of-timedelta(days=2)
    torch.set_num_threads(4)
    head=joblib.load(args.directory/'confirmation/text_evaluation.joblib')
    print('Loading local E5 for independent public texts',flush=True)
    embedder=head.encoder.embedder
    detector=ArticleTopicDetector(embedder)
    formula=AssesmentConformPPmetricsBased()
    client=TelegramClient(os.getenv('TG_SESSION','.telegram_session'),int(os.environ['TG_API_ID']),os.environ['TG_API_HASH'],flood_sleep_threshold=60)
    await client.connect()
    try:
        if not await client.is_user_authorized():raise RuntimeError('Authorize Telegram locally first')
        cached=CachedEntityClient(client)
        for candidate in candidates:
            username=candidate['username']
            if username.lower() in done:continue
            print('Reading public channel @'+username,flush=True)
            start=len(report['articles'])
            try:
                entity=await cached.get_entity('@'+username)
                full=await client(GetFullChannelRequest(entity))
                audience=int(full.full_chat.participants_count or candidate['audience'])
                resource=str(uuid5(NAMESPACE_URL,'https://t.me/'+username.lower()))
                if audience<=0:raise ValueError('Unknown audience')
                async for message in client.iter_messages(entity,offset_date=newest):
                    date=message.date
                    if date is None:continue
                    if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
                    if date<=oldest:break
                    if date>=newest:continue
                    text=message.message or ''
                    if not text.strip() or not getattr(getattr(message,'replies',None),'comments',False):continue
                    reactions=reaction_count(message)
                    comments=getattr(getattr(message,'replies',None),'replies',None)
                    views=getattr(message,'views',None)
                    if views is None or views<=0 or reactions is None or comments is None:continue
                    group=text_group(text)
                    if group in seen:continue
                    with threadpool_limits(limits=4):
                        vector=embedder.embed_document(text)
                        topic=detector.detect_from_embedding(vector)
                        features=head.encoder.transform([text],[topic],[vector])
                        predicted=float(head.model.predict(features)[0])
                    report['articles'].append(dict(article_id=f'TG:{entity.id}:{message.id}',group=group,text=text,
                        topic=topic,resource_id=resource,resource_name=getattr(entity,'title',username),username=username,
                        source='TG',audience=audience,score=predicted,metric_score=formula.assess(views,reactions,comments,audience),
                        counts=[views,reactions,comments],published_at=date.isoformat()))
                    seen.add(group)
                    if len(report['articles'])-start>=1000:break
                record=dict(username=username,status='complete',articles=len(report['articles'])-start,audience=audience)
            except FloodWaitError as error:
                report.update(status='rate_limited',retry_after_seconds=error.seconds)
                write_json(path,report)
                raise
            except Exception as error:
                record=dict(username=username,status='failed',error_type=type(error).__name__,articles=len(report['articles'])-start)
            report['channels'].append(record)
            write_json(path,report)
            print(json.dumps(dict(**record,total=len(report['articles'])),ensure_ascii=False),flush=True)
            if len(report['articles'])>=args.target:break
            await asyncio.sleep(.4)
        report.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
        write_json(path,report)
    finally:await client.disconnect()
    print(json.dumps(dict(status=report['status'],articles=len(report['articles']),channels=len(report['channels']))),flush=True)


if __name__=='__main__':
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform=='win32' else None) as runner:
        runner.run(main())

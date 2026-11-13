"""Collect the previously reserved, unseen publication interval read-only."""
import argparse
import asyncio
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import torch
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.functions.channels import GetFullChannelRequest
from threadpoolctl import threadpool_limits

from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from src.assesment_conform_of_PP_text_based.dataset import text_group
from src.database.resources.telegram.collector import reaction_count
from src.database.topic_detector import ArticleTopicDetector
from .discover_channels import CachedEntityClient,write_json


async def main():
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    frozen=json.loads((args.directory/'choices_frozen.json').read_text(encoding='utf8'))
    protocol=json.loads((args.directory/'control_protocol.json').read_text(encoding='utf8'))
    data=json.loads((args.directory/'development.json').read_text(encoding='utf8'))
    model_path=root/'artifacts/text_models/neural_network.joblib'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==protocol['text_model_sha256']
    old_groups=set(data['groups'])
    start=datetime.fromisoformat(protocol['holdout_publication_start_exclusive'])
    end=datetime.fromisoformat(protocol['holdout_publication_end_exclusive'])
    path=args.directory/'control_snapshot.json'
    if path.exists() and not args.resume:raise ValueError('Use --resume to continue')
    report=json.loads(path.read_text(encoding='utf8')) if path.exists() else dict(
        status='collecting',protocol=protocol,selected=frozen['selected'],articles=[],channels=[])
    done={r['username'].lower() for r in report['channels'] if r['status']=='complete'}
    seen=old_groups|{r['group'] for r in report['articles']}
    torch.set_num_threads(4)
    head=AssesmentConformPPtextBased.load(model_path)
    embedder=head.encoder.embedder
    detector=ArticleTopicDetector(embedder)
    client=TelegramClient(os.getenv('TG_SESSION','.telegram_session'),int(os.environ['TG_API_ID']),
        os.environ['TG_API_HASH'],flood_sleep_threshold=60)
    await client.connect()
    try:
        if not await client.is_user_authorized():raise RuntimeError('Authorize Telegram locally')
        cached=CachedEntityClient(client)
        for channel in protocol['channels']:
            username=channel['username']
            if username.lower() in done:continue
            count=0
            try:
                entity=await cached.get_entity('@'+username)
                full=await client(GetFullChannelRequest(entity))
                audience=int(full.full_chat.participants_count or channel['audience'])
                if audience<=0:raise ValueError('Unknown audience')
                async for message in client.iter_messages(entity,offset_date=end):
                    date=message.date
                    if date is None:continue
                    if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
                    if date<=start:break
                    if date>=end:continue
                    text=message.message or ''
                    if not text.strip() or not getattr(getattr(message,'replies',None),'comments',False):continue
                    views=getattr(message,'views',None)
                    reactions=reaction_count(message)
                    comments=getattr(getattr(message,'replies',None),'replies',None)
                    if views is None or views<=0 or reactions is None or comments is None:continue
                    group=text_group(text)
                    if group in seen:continue
                    with threadpool_limits(limits=4):
                        vector=embedder.embed_document(text)
                        topic=detector.detect_from_embedding(vector)
                        features=head.encoder.transform([text],[topic],[vector])
                        score=float(head.model.predict(features)[0])
                    report['articles'].append(dict(article_id=f'TG:{entity.id}:{message.id}',
                        group=group,text=text,topic=topic,source='TG',resource_id=channel['resource_id'],
                        resource_name=getattr(entity,'title',username),username=username,audience=audience,
                        score=score,counts=[views,reactions,comments],published_at=date.isoformat()))
                    seen.add(group);count+=1
                    if count>=1000:break
                record=dict(username=username,status='complete',articles=count,audience=audience)
            except FloodWaitError as error:
                report.update(status='rate_limited',retry_after_seconds=error.seconds)
                write_json(path,report)
                raise
            except Exception as error:
                record=dict(username=username,status='failed',articles=count,error_type=type(error).__name__)
            report['channels'].append(record)
            write_json(path,report)
            print(json.dumps(dict(**record,total=len(report['articles'])),ensure_ascii=False),flush=True)
            await asyncio.sleep(.4)
        report.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
        write_json(path,report)
    finally:await client.disconnect()
    print(json.dumps(dict(status=report['status'],articles=len(report['articles']),channels=len(report['channels']))),flush=True)


if __name__=='__main__':
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform=='win32' else None) as runner:
        runner.run(main())

"""Recover whole frozen texts read-only; verify every normalized text fingerprint."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np

from src.assesment_conform_of_PP_text_based.dataset import text_group
from src.database.connection import connect_database


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);load_dotenv('.env')
    if (args.output/'raw_texts.json').exists():raise ValueError('Text snapshot already exists')
    old=np.load('model_tests/results/expansion_20261111/social_metrics/dataset.npz')
    frozen=np.load(args.corpus/'corpus.npz')
    async with await connect_database(os.environ['DATABASE_URL']) as conn:
        cursor=await conn.execute('SELECT id,text FROM article_text')
        texts={str(r['id']):r['text'] for r in await cursor.fetchall()}
    result=[texts[str(article_id)] for article_id in old['article_ids']]
    for name in ['tuning_20261111/fresh_snapshot.json','history_20261111/control_snapshot.json']:
        result.extend(r['text'] for r in json.loads((Path('model_tests/results')/name).read_text(encoding='utf8'))['articles'])
    assert len(result)==len(frozen['groups'])==11709
    assert all(text_group(t)==g for t,g in zip(result,frozen['groups']))
    (args.output/'raw_texts.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf8')
    report=dict(rows=len(result),all_text_fingerprints_match=True,database_modified=False,
        corpus_sha256=hashlib.sha256((args.corpus/'corpus.npz').read_bytes()).hexdigest())
    (args.output/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':asyncio.run(main(),loop_factory=asyncio.SelectorEventLoop)

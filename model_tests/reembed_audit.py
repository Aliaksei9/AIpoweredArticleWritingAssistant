"""Read-only table fingerprints around the explicitly requested reembedding."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from src.database.connection import connect_database
from src.database.embedding_config import SETTINGS,EMBEDDING_SIGNATURE


async def main():
    root=Path(__file__).resolve().parents[1];load_dotenv(root/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--before',type=Path)
    args=parser.parse_args();result=dict(profile=EMBEDDING_SIGNATURE,dimensions=SETTINGS.dimensions,tables={})
    async with await connect_database(os.environ['DATABASE_URL']) as conn:
        for name in ['resource','article_text','article_sources','article_metrics']:
            cursor=await conn.execute('SELECT * FROM '+name+' ORDER BY 1')
            rows=await cursor.fetchall()
            digest=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str,ensure_ascii=False).encode('utf8')).hexdigest()
            result['tables'][name]=dict(rows=len(rows),sha256=digest)
        cursor=await conn.execute('SELECT count(*) AS n FROM article_vectors')
        result['vectors']=(await cursor.fetchone())['n']
        cursor=await conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='article_vectors' AND column_name='embedding_model'")
        exists=bool(await cursor.fetchone());result['model_metadata_column']=exists
        if exists:
            cursor=await conn.execute('SELECT count(*) AS n FROM article_vectors WHERE embedding_model IS DISTINCT FROM %s OR vector_dims(embedding)<>%s',
                                     (EMBEDDING_SIGNATURE,SETTINGS.dimensions))
            result['incompatible_vectors']=(await cursor.fetchone())['n']
            cursor=await conn.execute('SELECT max(abs(vector_norm(embedding)-1.)) AS error FROM article_vectors')
            result['maximum_norm_error']=(await cursor.fetchone())['error']
    if args.before:
        previous=json.loads(args.before.read_text(encoding='utf8'))
        result['nonvector_tables_unchanged']=previous['tables']==result['tables']
        assert result['nonvector_tables_unchanged'],'Articles or metrics changed during reembedding; inspect concurrent writes'
        assert result['vectors']==result['tables']['article_text']['rows'] and result.get('incompatible_vectors')==0
        assert result['maximum_norm_error']<1e-5
        result['status']='passed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform=='win32' else None) as runner:
        runner.run(main())

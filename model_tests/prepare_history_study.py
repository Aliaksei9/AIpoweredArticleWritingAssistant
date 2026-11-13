"""Prepare temporal development folds and reserve an unseen publication window."""
import argparse
import asyncio
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import numpy as np

from src.database.connection import connect_database
from src.database.ids_converter import resource_id_for
from .compare_text_models import write_json


async def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--as-of',required=True)
    args = parser.parse_args()
    if (args.output/'development.json').exists():
        raise ValueError('Use a new study directory')
    previous = root/'model_tests/results/expansion_20261111'
    tuning = root/'model_tests/results/tuning_20261111'
    collection = json.loads((previous/'collection.json').read_text(encoding='utf8'))
    screening = json.loads((previous/'channel_screening.json').read_text(encoding='utf8'))
    fresh = json.loads((tuning/'fresh_snapshot.json').read_text(encoding='utf8'))
    splits = json.loads((previous/'social_metrics/splits.json').read_text(encoding='utf8'))
    inputs = json.loads((tuning/'confirmation/nested_inputs.json').read_text(encoding='utf8'))
    with np.load(previous/'social_metrics/dataset.npz') as saved:
        counts = saved['counts']; ids = saved['article_ids']; groups = saved['groups']
    async with await connect_database(os.environ['DATABASE_URL']) as conn:
        cursor = await conn.execute('SELECT id, published_at FROM article_text')
        dates = {str(row['id']):row['published_at'].isoformat() for row in await cursor.fetchall()}
    rows = [None]*len(counts)
    for indices, values in [(splits['train_indices'],inputs['heldout']['train']),
                             (splits['test_indices'],inputs['heldout']['test'])]:
        for index,row in zip(indices,values):
            rows[index] = dict(row,published_at=dates[str(ids[index])])
    channels = {r['username'].lower():r for r in screening['channels'] if r.get('qualifies')}
    id_map = {name:str(resource_id_for('TG',record['channel_id'])) for name,record in channels.items()}
    for result in collection['resources']:
        if result['status']=='complete' and result.get('resource'):
            assert result['resource']['id']==id_map[result['username'].lower()]
    records = []
    for article in fresh['articles']:
        records.append(dict(article,resource_id=id_map[article['username'].lower()]))
    fresh_rows = [{key:r[key] for key in ['source','resource_id','topic','audience','score','published_at']} for r in records]
    all_rows = rows+fresh_rows
    all_counts = np.vstack([counts,np.asarray([r['counts'] for r in records])])
    all_groups = groups.tolist()+[r['group'] for r in records]
    assert not set(groups).intersection(r['group'] for r in records)
    folds = []
    eligible = {r['resource_id'] for r in fresh_rows if sum(x['resource_id']==r['resource_id'] for x in fresh_rows)>=10}
    for fraction in [.4,.6,.8]:
        train = list(range(len(rows))); validation = []
        for resource in sorted(eligible):
            indices = sorted([len(rows)+i for i,r in enumerate(fresh_rows) if r['resource_id']==resource],
                             key=lambda i:all_rows[i]['published_at'])
            start = int(len(indices)*fraction)
            end = len(indices) if fraction==.8 else int(len(indices)*(fraction+.2))
            train.extend(indices[:start]); validation.extend(indices[start:end])
            assert max(all_rows[i]['published_at'] for i in indices[:start])<=min(all_rows[i]['published_at'] for i in indices[start:end])
        assert not set(all_groups[i] for i in train)&set(all_groups[i] for i in validation)
        folds.append(dict(train=train,validation=validation,history_fraction=fraction))
    as_of = datetime.fromisoformat(args.as_of)
    old_as_of = datetime.fromisoformat(collection['as_of'])
    protocol = dict(created_at=args.as_of,
        holdout_publication_start_exclusive=(old_as_of-timedelta(days=2)).isoformat(),
        holdout_publication_end_exclusive=(as_of-timedelta(days=2)).isoformat(),
        age_days=dict(min_exclusive=2,max_exclusive=14),
        selection='minimum mean chronological development MSE ratio to channel mean; no use of new holdout labels; warm latency <=10 s',
        text_model_sha256='25271135e787a642a1507889925cd3f312f4ebffffb54a9d45bb669fea3fb7c9',
        channels=[dict(username=name,resource_id=id_map[name],audience=r['audience']) for name,r in channels.items()])
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'control_protocol.json',protocol)
    write_json(args.output/'development.json',dict(rows=all_rows,counts=all_counts.tolist(),groups=all_groups,
        old_rows=len(rows),fresh_rows=len(fresh_rows),folds=folds))
    print(json.dumps(dict(rows=len(all_rows),fresh_rows=len(records),channels=len(channels),
        validation_rows=[len(f['validation']) for f in folds],reserved_window=[protocol['holdout_publication_start_exclusive'],protocol['holdout_publication_end_exclusive']])) ,flush=True)


if __name__=='__main__':
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform=='win32' else None) as runner:
        runner.run(main())

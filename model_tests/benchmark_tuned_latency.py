"""Warm latency of the frozen evaluation pair before production replacement."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.social_metrics_prediction.dataset import load_metric_dataset
from .benchmark_utils import representative_cases,response_timing
from .compare_text_models import write_json


async def main():
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args()
    confirmation=json.loads((args.directory/'confirmation/confirmation.json').read_text(encoding='utf8'))
    dataset=await load_metric_dataset(os.environ['DATABASE_URL'])
    previous=Path('model_tests/results/expansion_20261111/social_metrics')
    split=json.loads((previous/'splits.json').read_text(encoding='utf8'))
    report=json.loads((previous/'report.json').read_text(encoding='utf8'))
    assert report['dataset_fingerprint']==dataset.fingerprint
    text=joblib.load(args.directory/'confirmation/text_evaluation.joblib')
    social=joblib.load(args.directory/('fresh_social_evaluation.joblib' if args.fresh else 'confirmation/social_evaluation.joblib'))
    torch.set_num_threads(4)
    cases=representative_cases(dataset.text.texts,split['train_indices'])
    def predict(index):
        score=text.assess(dataset.text.texts[index],dataset.text.topics[index])
        result=social.predict_rows([dict(dataset.rows[index],score=float(score))])
        assert result.shape==(1,3) and np.isfinite(result).all()
        return result
    with threadpool_limits(limits=4):
        timing=response_timing(predict,cases)
        longest=dataset.text.texts[cases[-1]]
        stress=((longest+'\n')*(50000//(len(longest)+1)+1))[:50000]
        started=time.perf_counter()
        score=text.assess(stress,dataset.text.topics[cases[-1]])
        social.predict_rows([dict(dataset.rows[cases[-1]],score=float(score))])
        stress_seconds=time.perf_counter()-started
    selected=(json.loads((args.directory/'fresh_confirmation.json').read_text(encoding='utf8'))['selected']
              if args.fresh else confirmation['social_selected'])
    result=dict(status='passed' if max(timing['max'],stress_seconds)<=10 else 'too_slow',
        response_time=timing,stress_50000_seconds=stress_seconds,max_response_seconds=10,
        case_lengths=[len(dataset.text.texts[i]) for i in cases],social_selected=selected,
        scope='E5 and frozen text/social evaluation pair; warmed; no LLM/network/DB in timed section')
    write_json(args.directory/('fresh_evaluation_latency.json' if args.fresh else 'evaluation_latency.json'),result)
    print(json.dumps(result),flush=True)
    if result['status']!='passed':raise SystemExit(2)


if __name__=='__main__':
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform=='win32' else None) as runner:
        runner.run(main())

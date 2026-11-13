"""Audit optimized contracts and prepare the complete frozen experiment corpus."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import torch

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.assesment_conform_of_PP_text_based import TextTopicEncoder
from src.database.model_embedding import TextEmbedder,MODEL_NAME
from src.database.vector_utils import normalize_vector
from src.database.embedding_config import EMBEDDING_SIGNATURE
from .compare_text_models import write_json


def main():
    root=Path(__file__).resolve().parents[1];load_dotenv(root/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    history=root/'model_tests/results/history_20261111'
    previous=root/'model_tests/results/expansion_20261111'
    data=json.loads((history/'development.json').read_text(encoding='utf8'))
    control=json.loads((history/'control_snapshot.json').read_text(encoding='utf8'))
    fresh=json.loads((root/'model_tests/results/tuning_20261111/fresh_snapshot.json').read_text(encoding='utf8'))['articles']
    raw=fresh+control['articles']
    rows=data['rows']+[{k:r[k] for k in ['source','resource_id','topic','audience','score','published_at']} for r in control['articles']]
    counts=np.vstack([data['counts'],[r['counts'] for r in control['articles']]])
    groups=np.asarray(data['groups']+[r['group'] for r in control['articles']])
    with np.load(previous/'social_metrics/dataset.npz') as cache:
        old_features=cache['text_features'];old_labels=cache['metric_scores']
    metadata=json.loads((previous/'text_models/report.json').read_text(encoding='utf8'))
    assert metadata['feature_model']==MODEL_NAME
    assert EMBEDDING_SIGNATURE==metadata['feature_model'], 'Historical cached features require their original embedding profile; rebuild the corpus for a different model/prefix/dimension'
    assert np.max(np.abs(np.linalg.norm(old_features[:,:384],axis=1)-1))<1e-5
    formula=AssesmentConformPPmetricsBased()
    labels=np.asarray([formula.assess(*values,row['audience']) for row,values in zip(rows,counts)])
    assert np.max(np.abs(labels[:len(old_labels)]-old_labels))<=1e-6
    rng=np.random.default_rng(42);differences=[]
    for _ in range(10000):
        v,r,c,a=np.exp(rng.uniform(-3,25,4))
        activity=.2*math.log1p(v/(.1*a))+.4*math.log1p(r/(.0008*a))+.4*math.log1p(c/(.00005*a))
        expected=round(10*math.tanh(math.log(activity/math.log(2))/1.5),6)
        differences.append(abs(expected-formula.assess(v,r,c,a)))
    assert max(differences)<=1e-6
    torch.set_num_threads(4)
    embedder=TextEmbedder();encoder=TextTopicEncoder(embedder=embedder)
    chunks=[];bounds=[]
    for record in raw:
        start=len(chunks);chunks.extend(embedder._chunks('passage',record['text']));bounds.append((start,len(chunks)))
    print(f'Encoding {len(raw)} complete texts in {len(chunks)} chunks',flush=True)
    embeddings=embedder.model.encode(chunks,normalize_embeddings=True,convert_to_numpy=True,batch_size=32,show_progress_bar=False)
    new_features=[]
    for record,(start,end) in zip(raw,bounds):
        vector=normalize_vector([math.fsum(float(v[j]) for v in embeddings[start:end])/(end-start) for j in range(384)])
        new_features.append(np.concatenate([encoder._vector(vector),encoder.topic_embedding(record['topic'])]))
    features=np.vstack([old_features,np.asarray(new_features,dtype=np.float32)])
    assert len(features)==len(rows)==len(groups)==11709
    np.savez_compressed(args.output/'corpus.npz',features=features,labels=labels,counts=counts,groups=groups)
    write_json(args.output/'rows.json',rows)
    write_json(args.output/'audit.json',dict(status='passed',rows=len(rows),distinct_texts=len(set(groups)),
        optimized_formula_cases=10000,max_formula_difference=max(differences),stored_labels_match=True,
        fixed_embedding_model=MODEL_NAME,old_cached_features=10182,newly_encoded_texts=len(raw),
        database_modified=False,production_weights_modified=False,
        old_feature_provenance='historical local snapshot explicitly records fixed E5; current DB vectors not used',
        social_sha256=hashlib.sha256((root/'artifacts/social_metrics/model.joblib').read_bytes()).hexdigest(),
        text_sha256=hashlib.sha256((root/'artifacts/text_models/neural_network.joblib').read_bytes()).hexdigest()))
    print('Corpus and optimized-module numerical audit ready',flush=True)


if __name__=='__main__':main()

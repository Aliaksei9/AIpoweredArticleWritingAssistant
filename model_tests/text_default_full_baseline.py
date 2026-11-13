"""Fixed current text configuration on the same five full-corpus outer folds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased
from .compare_text_models import write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    saved=np.load(args.directory/'corpus.npz');X=saved['features'];y=saved['labels'];groups=saved['groups']
    predictions=np.full(len(y),np.nan);folds=[];torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for number,(train,test) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=20261111).split(X,groups=groups),1):
            model=AssesmentConformPPtextBased(seed=42,num_threads=4).train_features(X[train],y[train])
            predictions[test]=model.model.predict(X[test]);folds.append(float(np.mean((y[test]-predictions[test])**2)))
            print(f'Text baseline {number}/5 MSE={folds[-1]:.4f}',flush=True)
    write_json(args.directory/'text_baseline.json',dict(rows=len(y),mse=float(np.mean((predictions-y)**2)),folds=folds,
        configuration=model.model.estimator.get_params(),production_weights_modified=False))

if __name__=='__main__':main()

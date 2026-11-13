"""Choose alternative families by inner errors, never by outer test performance."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .compare_social_metrics import evaluate
from .compare_text_models import write_json


METRICS=('views','reactions','comments')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--classical',type=Path,required=True)
    p.add_argument('--neural',type=Path,required=True);p.add_argument('--transfer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Completed merged study exists')
    directories=dict(classical=args.classical,neural=args.neural,transfer=args.transfer)
    old=np.load(args.previous/'predictions.npz');actual=old['actual'];baseline=old['baseline'];previous=old['predicted']
    sources={name:np.load(path/'predictions.npz') for name,path in directories.items()}
    for data in sources.values():np.testing.assert_array_equal(actual,data['actual'])
    values={name:data['richer_inputs'] if name=='classical' else data['predicted'] for name,data in sources.items()}
    prediction=np.full_like(actual,np.nan);hits=np.zeros(len(actual),int)
    report=dict(rows=len(actual),folds=[],target_reduction_percent=2.,production_weights_modified=False,
                protocol='Five frozen outer exact-text-group folds; family and metric chosen by their common training-only inner validation errors')
    for fold in range(1,6):
        path=args.output/f'fold_{fold}';path.mkdir(exist_ok=True)
        prepared=joblib.load(args.previous/f'fold_{fold}'/'prepared.joblib');test=prepared['test']
        selection={name:json.loads((directory/f'fold_{fold}'/'selection.json').read_text(encoding='utf8')) for name,directory in directories.items()}
        error={name:np.asarray([item['inner_mse'][(item['selected']['richer_inputs'] if name=='classical' else item['selected'])[m]][i]
                              for i,m in enumerate(METRICS)]) for name,item in selection.items()}
        chosen={m:min(error,key=lambda name:error[name][i]) for i,m in enumerate(METRICS)}
        heads={};loaded={}
        for i,m in enumerate(METRICS):
            name=chosen[m];filename='richer_inputs.joblib' if name=='classical' else 'heads.joblib'
            if name not in loaded:loaded[name]=joblib.load(directories[name]/f'fold_{fold}'/filename)
            heads[m]=loaded[name][m];prediction[test,i]=values[name][test,i]
        hits[test]+=1;joblib.dump(heads,path/'heads.joblib',compress=3)
        selected_model={m:(selection[name]['selected']['richer_inputs'] if name=='classical' else selection[name]['selected'])[m]
                        for m,name in chosen.items()}
        report['folds'].append(dict(fold=fold,selected_family=chosen,selected_model=selected_model,
                                    inner_mse={n:v.tolist() for n,v in error.items()}))
    assert np.all(hits==1) and np.isfinite(prediction).all()
    base=evaluate(actual,baseline);prior=evaluate(actual,previous);candidate=evaluate(actual,prediction)
    gains={m:100*(1-candidate[m]['mse']/base[m]['mse']) for m in METRICS}
    report.update(status='complete',every_row_tested_once=True,baseline=base,previous=prior,candidate=candidate,
        gains_percent=gains,gains_vs_previous_percent={m:100*(1-candidate[m]['mse']/prior[m]['mse']) for m in METRICS},
        passes_accuracy_gate=bool(all(v>=2 for v in gains.values())),
        family_results={name:evaluate(actual,value) for name,value in values.items()})
    np.savez_compressed(args.output/'predictions.npz',actual=actual,baseline=baseline,previous=previous,predicted=prediction)
    write_json(args.output/'report.json',report);print(json.dumps(dict(gains_percent=gains,passes_accuracy_gate=report['passes_accuracy_gate'])),flush=True)


if __name__=='__main__':main()

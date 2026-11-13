"""Freeze a conservative profile using CV errors only, before new collection."""
import argparse
import json
from pathlib import Path

from .compare_text_models import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.directory/'confirmation/confirmation.json').read_text(encoding='utf8'))
    baseline=report['results']['resource_mean']['cv_folds']
    ratios={name:max(a/b for a,b in zip(values['cv_folds'],baseline))
            for name,values in report['results'].items() if name!='resource_mean'}
    selected=min(ratios,key=ratios.get)
    if ratios[selected]>=1:raise ValueError('No candidate improves every CV fold')
    output=args.directory/'fresh_choices_frozen.json'
    if output.exists():
        previous=json.loads(output.read_text(encoding='utf8'))
        assert previous['selected']==selected and previous['cv_max_ratios']==ratios
    else:
        assert not (args.directory/'fresh_snapshot.json').exists()
        write_json(output,dict(protocol='additional confirmation: minimize worst-fold CV normalized MSE ratio vs channel mean; require every fold below 1; no fresh labels observed',
            selected=selected,cv_max_ratios=ratios,requires_ratio_below_one=True))
    print(json.dumps(dict(selected=selected,worst_cv_ratio=ratios[selected])),flush=True)


if __name__=='__main__':main()

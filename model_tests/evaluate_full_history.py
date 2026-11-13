"""Whole-corpus grouped OOF evaluation of the frozen history configuration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold

from .compare_social_metrics import evaluate
from .compare_text_models import write_json
from .history_candidates import ConditionalHistoryPredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('model_tests/results/history_20261111'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if (args.output/'report.json').exists():raise ValueError('Use a new output directory')
    data=json.loads((args.study/'development.json').read_text(encoding='utf8'))
    control=json.loads((args.study/'control_snapshot.json').read_text(encoding='utf8'))
    frozen=json.loads((args.study/'choices_frozen.json').read_text(encoding='utf8'))
    rows=data['rows']+[{k:r[k] for k in ['source','resource_id','topic','audience','score','published_at']} for r in control['articles']]
    counts=np.vstack([data['counts'],[r['counts'] for r in control['articles']]])
    groups=np.asarray(data['groups']+[r['group'] for r in control['articles']])
    predictions={name:np.zeros_like(counts) for name in ['channel_mean','candidate']}
    normalized={name:np.zeros_like(counts) for name in predictions}
    hits=np.zeros(len(rows),dtype=int);folds=[]
    for number,(train,test) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=20261111).split(counts,groups=groups),1):
        assert not set(groups[train])&set(groups[test])
        fitting=[rows[i] for i in train];validation=[rows[i] for i in test]
        variance=np.maximum(np.var(counts[train],axis=0),1.)
        results={}
        for name,parameters in [('channel_mean',{}),('candidate',frozen['configuration'])]:
            model=ConditionalHistoryPredictor(**parameters).fit(fitting,counts[train])
            prediction=model.predict_rows(validation)
            predictions[name][test]=prediction
            normalized[name][test]=(prediction-counts[test])**2/variance
            results[name]=evaluate(counts[test],prediction)
        hits[test]+=1
        folds.append(dict(fold=number,train_rows=len(train),test_rows=len(test),metrics=results))
        print(f'Fold {number}/5: evaluated {len(test)} articles',flush=True)
    assert np.all(hits==1)
    summary={name:dict(normalized_mse=float(normalized[name].mean()),metrics=evaluate(counts,prediction)) for name,prediction in predictions.items()}
    # Every text group is resampled as a block; descriptive uncertainty only.
    _,inverse=np.unique(groups,return_inverse=True)
    delta=(normalized['channel_mean']-normalized['candidate']).mean(axis=1)
    sums=np.bincount(inverse,weights=delta);sizes=np.bincount(inverse)
    rng=np.random.default_rng(42);bootstrap=[]
    for _ in range(2000):
        indices=rng.integers(0,len(sums),size=len(sums))
        bootstrap.append(float(sums[indices].sum()/sizes[indices].sum()))
    report=dict(status='complete',rows=len(rows),distinct_texts=len(set(groups)),resources=len({r['resource_id'] for r in rows}),
        folds=folds,results=summary,configuration=frozen['configuration'],
        improvement_percent=(1-summary['candidate']['normalized_mse']/summary['channel_mean']['normalized_mse'])*100,
        bootstrap_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
        every_row_predicted_once=True,exact_text_groups_disjoint=True,
        limitations=['Frozen configuration was previously selected using some of these data; this is not a fresh independent test',
                     'Random grouped folds measure same-channel interpolation, not future-time prediction',
                     'Upstream text scores are frozen historical cross-fitted inputs; upstream networks were not retrained for these new folds. This is conditional module-4 evaluation, not fully nested end-to-end validation'])
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'report.json',report)
    np.savez_compressed(args.output/'oof_predictions.npz',actual=counts,**predictions)
    with (args.output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','metric','mse','rmse','mae','r2'])
        for name,result in summary.items():
            for metric,values in result['metrics'].items():writer.writerow([name,metric,*[values[k] for k in ['mse','rmse','mae','r2']]])
    lines=['# Проверка модуля социальных метрик на всей выборке','',
        f"Оценены **{len(rows)} статей**, **{len(set(groups))} разных нормализованных текстов**, **{report['resources']} каналов**. "
        'Каждая строка проверена ровно один раз в пяти групповых разбиениях. Совпадающие тексты не пересекаются между обучением и проверкой. '
        'Среднее канала рассчитывается только по обучающей части каждого разбиения. Рабочие веса и БД не изменялись.','',
        f"Изменение общей нормированной MSE относительно среднего канала: **{report['improvement_percent']:.2f}% снижения**. "
        'Отрицательное снижение означает ухудшение. MSE каждой строки нормируется на дисперсию соответствующей обучающей части.','',
        '| Метрика | MSE среднего канала | MSE модели | RMSE среднего | RMSE модели | Снижение MSE |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for metric,label in [('views','Просмотры'),('reactions','Реакции'),('comments','Комментарии')]:
        a=summary['channel_mean']['metrics'][metric];b=summary['candidate']['metrics'][metric]
        lines.append(f"| {label} | {a['mse']:.2f} | {b['mse']:.2f} | {a['rmse']:.2f} | {b['rmse']:.2f} | {(1-b['mse']/a['mse'])*100:.2f}% |")
    lines+=['','## Границы проверки','',
        'Это OOF-проверка модуля 4 при фиксированных входных прогнозах текстовой модели. Для новых пяти разбиений '
        'текстовые сети заново не обучались: их исторические OOF-разбиения отличаются. Поэтому эта проверка '
        'не является полностью вложенной проверкой всей цепочки модулей 3–4. Конфигурация также ранее выбиралась '
        'на части этого корпуса. Числа описывают всю имеющуюся выборку, но не заменяют независимую проверку будущих публикаций.','',
        'Случайные группы могут использовать более новые публикации при прогнозе старых. '
        'Отдельный хронологический контроль на 216 новых статьях сохранён '
        'в [предыдущем отчёте](../2026-11-11-history/README.md). Его результат: снижение общей ошибки на 4,62%.','',
        '![MSE на всей выборке](mse.png)','',
        'Код: `model_tests/evaluate_full_history.py`. Полные прогнозы по строкам хранятся только локально; '
        'публикуются агрегированные показатели.','']
    (args.output/'README.md').write_text('\n'.join(lines),encoding='utf8')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,metric,label in zip(axes,['views','reactions','comments'],['Просмотры','Реакции','Комментарии']):
        values=[summary[n]['metrics'][metric]['mse'] for n in ['channel_mean','candidate']]
        ax.bar([0,1],values,color=['#9aa4b3','#4d9879']);ax.set(xticks=[0,1],xticklabels=['Среднее канала','Модель'],title=label,ylabel='MSE, количество²')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'Групповая OOF-проверка · {len(rows)} статей')
    for ext in ['png','svg']:fig.savefig(args.output/('mse.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k:report[k] for k in ['rows','distinct_texts','resources','improvement_percent','results']}),flush=True)


if __name__=='__main__':main()

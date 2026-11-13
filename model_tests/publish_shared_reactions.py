"""Publish reaction-only aggregate results, excluding the private training corpus."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .compare_text_models import write_json


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    report=json.loads(args.report.read_text(encoding='utf8'))
    latency=json.loads(args.latency.read_text(encoding='utf8'))
    training=json.loads(args.training.read_text(encoding='utf8'))
    write_json(args.output/'summary.json',dict(report,latency=latency,
        final_reaction_configuration=training['configuration']))
    shutil.copy2(args.latency,args.output/'latency.json')
    labels=dict(views='Просмотры',reactions='Реакции',comments='Комментарии')
    with (args.output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['metric','mean_mse','previous_mse','candidate_mse','gain_to_mean_percent'])
        for name in labels:writer.writerow([name,report['baseline'][name]['mse'],
            report['previous_models_same_rows'][name]['mse'],report['candidate'][name]['mse'],report['gains_percent'][name]])
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':140})
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,(name,label) in zip(axes,labels.items()):
        bars=ax.bar(['Среднее\nпо каналу','До подбора','После\nподбора'],
            [report[key][name]['mse'] for key in ['baseline','previous_models_same_rows','candidate']],
            color=['#afb8c8','#8e78b6','#4965dc'])
        ax.set(title=label,ylabel='MSE, исходные количества²');ax.bar_label(bars,fmt='%.0f',padding=3,fontsize=8);ax.margins(y=.2)
    fig.suptitle(f"Независимая модель реакций: все {report['rows']} статей")
    fig.tight_layout();fig.savefig(args.output/'mse.png');fig.savefig(args.output/'mse.svg');plt.close(fig)
    previous=report['previous_models_same_rows']['reactions']['mse']
    baseline=report['baseline']['reactions']['mse']
    old_gain=100*(1-previous/baseline)
    lines=['# Подбор отдельной модели реакций — 12.11.2026','',
        f"Выигрыш реакций относительно среднего по каналу вырос с **{old_gain:.3f}%** до **{report['gains_percent']['reactions']:.3f}%** по исходной MSE. Порог 1% пройден на всей выборке. Модели просмотров, комментариев и текстовая сеть сохранены.",
        '', '| Метрика | MSE среднего | MSE до подбора | MSE после | Снижение MSE к среднему |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,label in labels.items():lines.append(f"| {label} | {report['baseline'][name]['mse']:.2f} | {report['previous_models_same_rows'][name]['mse']:.2f} | {report['candidate'][name]['mse']:.2f} | {report['gains_percent'][name]:.3f}% |")
    lines.extend(['','## Проверка','',
        f"Использованы те же **{report['rows']} уникальных статей**, пять внешних групповых разбиений и прогнозы текстовой сети, что в [предыдущем сравнении](../2026-11-12-unified-inputs/README.md). Каждая статья проверена ровно один раз вне обучающей группы. Точные нормализованные повторы текста не пересекаются между частями. Веса других двух моделей не менялись; их прогнозы на всех строках совпадают.",
        '', 'В каждой внешней обучающей части параметры реакций выбирались на трёх внутренних групповых разбиениях. Проверены 3000 сочетаний числа соседей, температуры, квантиля, силы поправки и веса контекста. Внешние целевые значения в подбор не передавались. В проверке сохранены исходные счётчики, включая редкие всплески; ограничиваются только обучающие значения при построении модели.',
        '', 'Все три независимые модели получают одну полную матрицу: вектор текста, тему, прогнозную оценку популярности, ресурс, тип источника, аудиторию, дату публикации и срок прогноза. Веса контекста положительны. Другие фактические счётчики не являются входами модели реакций. Для каждой статьи используется одна текущая запись; накопления повторных замеров нет.',
        '', f"Максимальный прогретый полный ответ с SBERT: **{latency['warm_max_seconds']:.3f} с**. Загрузка моделей и первый запрос в лимит не входят. Время обучения не использовалось при выборе.",
        '', '## Итоговая конфигурация реакций','', '```json',json.dumps(training['configuration'],ensure_ascii=False,indent=2),'```',
        '', '![Сравнение MSE](mse.png)','', '## Воспроизведение','', '```powershell',
        'python -m model_tests.validate_unified_metrics --database-only --output model_tests/results/my_validation',
        'python -m src.social_metrics_prediction train','```',
        '', 'Локальная фиксированная выборка и промежуточные модели не включены в GitHub. Исследовательский сценарий `refine_shared_reactions.py` позволяет повторить парное сравнение по сохранённым внешним разбиениям. Рабочее обучение использует ту же процедуру подбора параметров реакций.',
        '', 'Корпус ранее использовался в разработке; это результат повторной групповой проверки, а не независимое подтверждение на будущих датах. 1527 дополнительных статей имеют общее время сбора партии. Глобальная MSE чувствительна к редким большим значениям реакций. Качество на каждом канале и дне отдельно не гарантируется.'])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(gains=report['gains_percent'],output=str(args.output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('report','latency','training','output'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())

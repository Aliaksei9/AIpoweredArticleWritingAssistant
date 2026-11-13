"""Export aggregate comparison and latency without article texts or feature data."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .compare_text_models import write_json


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    report=json.loads(args.report.read_text(encoding='utf8'));latency=json.loads(args.latency.read_text(encoding='utf8'))
    report['latency']=latency;write_json(args.output/'summary.json',report);write_json(args.output/'latency.json',latency)
    shutil.copy2(args.migration,args.output/'database_migration.json')
    names=['views','reactions','comments'];labels=['Просмотры','Реакции','Комментарии']
    with (args.output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['metric','mean_mse','previous_mse_same_rows','unified_mse','gain_to_mean_percent','change_to_previous_percent'])
        for name in names:writer.writerow([name,report['baseline'][name]['mse'],report['previous_models_same_rows'][name]['mse'],report['candidate'][name]['mse'],report['gains_percent'][name],report['change_vs_previous_mse_percent'][name]])
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':140})
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,name,label in zip(axes,names,labels):
        values=[report[key][name]['mse'] for key in ['baseline','previous_models_same_rows','candidate']]
        bars=ax.bar(['Среднее','Прежние\nмодели','Общие\nвходы'],values,color=['#b0b8c8','#8e78b6','#4965dc'])
        ax.set_title(label);ax.set_ylabel('MSE, исходные количества²');ax.bar_label(bars,fmt='%.0f',padding=3,fontsize=8);ax.margins(y=.2)
    fig.suptitle('Одни и те же 11 740 статей, разбиения и оценки текста');fig.tight_layout();fig.savefig(args.output/'mse.png');fig.savefig(args.output/'mse.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.5));durations=[r['seconds'] for r in latency['measurements']]
    ax.bar(range(len(durations)),durations,color=['#4965dc' if r['kind']=='corpus' else '#37a58d' for r in latency['measurements']]);ax.axhline(10,color='#ab4e42',linestyle='--',label='Предел 10 с');ax.legend()
    ax.set(xlabel='Запрос (последние три — 50 000 символов)',ylabel='Прогретый полный ответ, с',title='SBERT → оценка → три прогноза');fig.tight_layout();fig.savefig(args.output/'latency.png');fig.savefig(args.output/'latency.svg');plt.close(fig)
    lines=['# Общие входы и одна запись на статью — 12.11.2026','',
        'Просмотры, реакции и комментарии получают одну и ту же полную матрицу: вектор текста, тема, предсказанная оценка популярности, ресурс, тип источника, аудитория, время публикации и срок прогноза. Различаются обученные параметры моделей.',
        '', 'В рабочей БД удалена таблица истории замеров. В пяти исходных таблицах для каждой статьи остаётся актуальное состояние. Повторный сбор обновляет метрики по ID статьи. Сохраняется только время получения последнего набора счётчиков.',
        '', f"Проверены все **{report['rows']} уникальных статей**; прежние и новые модели обучались на одинаковых частях, получали одинаковые оценки текста и проверялись на одинаковых строках. Пять внешних групповых разбиений, три внутренних разбиения текстовой сети, два разбиения подбора социальных моделей. Целевые счётчики не изменялись.",
        '', '| Метрика | MSE среднего | MSE прежних моделей | MSE общих входов | Изменение MSE к прежним |','| --- | ---: | ---: | ---: | ---: |']
    for name,label in zip(names,labels):lines.append(f"| {label} | {report['baseline'][name]['mse']:.2f} | {report['previous_models_same_rows'][name]['mse']:.2f} | {report['candidate'][name]['mse']:.2f} | {report['change_vs_previous_mse_percent'][name]:+.3f}% |")
    lines.extend(['', 'Отрицательное изменение означает уменьшение ошибки. Просмотры улучшились относительно прежних моделей; реакции и комментарии немного ухудшились. Изменение набора входов не объявляется улучшением всех метрик.',
        '', '| Метрика | Снижение MSE относительно среднего | Цель 2% |','| --- | ---: | --- |'])
    for name,label in zip(names,labels):lines.append(f"| {label} | {report['gains_percent'][name]:.3f}% | {'достигнута' if report['gains_percent'][name]>=2 else 'не достигнута'} |")
    lines.extend(['',f"MSE оценки по тексту: **{report['text_mse']:.6f}**. Текстовая сеть получает только текст и тему; метка каждой статьи берётся из последнего набора её метрик.",
        '',f"Максимум прогретого полного ответа: **{latency['warm_max_seconds']:.3f} с**, включая SBERT. Загрузка моделей и первый запрос исключены из лимита. Время обучения не использовалось как критерий выбора.",
        '', '![MSE](mse.png)','', '![Время ответа](latency.png)','', '## Воспроизведение','', '```powershell',
        'python -m src.database init-db',
        'python -m model_tests.validate_unified_metrics --database-only --output model_tests/results/my_unique_validation',
        'python -m src.social_metrics_prediction train','```','',
        'Перед заменой собственных весов нужно закончить полную проверку. Исследовательские снимки не включены в GitHub; новая проверка использует текущие статьи собственной БД. Рабочий код не читает историю замеров, не выбирает замер по близости к выбранному возрасту и не хранит прежние версии текста.',
        '', 'Для текущего сравнения из локального корпуса заранее оставлена только последняя запись каждого ID статьи. 391 более ранняя запись исключена до обучения. 1527 старых дополнительных статей имеют общее время сбора партии. Корпус ранее использовался в разработке: необходим отдельный контроль будущих дат. Наличие срока прогноза не гарантирует точную калибровку по каждому дню.'])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(rows=report['rows'],gains=report['gains_percent'],paired_change=report['change_vs_previous_mse_percent'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report','latency','migration','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())

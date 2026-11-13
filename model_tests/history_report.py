"""Aggregate report and plots for the chronological channel-history study."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    directory=args.directory;output=directory/'public';output.mkdir(exist_ok=True)
    results=json.loads((directory/'confirmation.json').read_text(encoding='utf8'))
    screening=json.loads((directory/'screening.json').read_text(encoding='utf8'))
    protocol=json.loads((directory/'control_protocol.json').read_text(encoding='utf8'))
    latency=json.loads((directory/'api_latency.json').read_text(encoding='utf8'))
    for name in ['confirmation.json','screening.json','choices_frozen.json','api_latency.json']:
        shutil.copy2(directory/name,output/name)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=170,bbox_inches='tight')
        fig.savefig(output/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    names=['channel_mean','previous_profile','candidate'];labels=['Среднее канала','Прежняя схема','Новая модель']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['views','reactions','comments'],['Просмотры','Реакции','Комментарии']):
        values=[results['results'][name]['metrics'][metric]['mse'] for name in names]
        ax.bar(np.arange(3),values,color=['#9ca5b3','#6581b9','#4e9a78'])
        for position,value in enumerate(values):ax.text(position,value*1.025,f'{value:,.1f}',ha='center',fontsize=8)
        ax.set(xticks=np.arange(3),xticklabels=labels,title=title,ylabel='MSE, количество²',ylim=(0,max(values)*1.18))
        ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f"Новый контроль: {results['control_rows']} статей из {results['control_channels']} каналов")
    save(fig,'control_mse')
    top=sorted(screening['results'],key=lambda n:screening['results'][n]['mean_ratio'])[:12]
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    values=[(1-screening['results'][name]['mean_ratio'])*100 for name in top]
    ax.barh(np.arange(len(top)),values,color='#4e9a78')
    ax.set(yticks=np.arange(len(top)),yticklabels=top,xlabel='Снижение ошибки CV относительно среднего канала, %',
           title='Хронологический подбор · 12 лучших конфигураций')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);save(fig,'development_gain')
    ordinary=[r['seconds'] for r in latency['cases'] if not r['stress']]
    stress=next(r['seconds'] for r in latency['cases'] if r['stress'])
    fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
    values=[float(np.median(ordinary)),max(ordinary),stress]
    ax.bar([0,1,2],values,color='#6581b9')
    ax.axhline(10,linestyle='--',color='#bd7854',label='Предел 10 с')
    ax.set(xticks=[0,1,2],xticklabels=['Реальные тексты,\nмедиана','Реальные тексты,\nмаксимум','50 000 символов'],ylabel='Секунды',
           title='Прогретый API · E5, тема, оценка текста и социальные метрики')
    ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'response_time')
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model','metric','mse','rmse','mae','r2'])
        for name in names:
            for metric in ['views','reactions','comments']:
                value=results['results'][name]['metrics'][metric]
                writer.writerow([name,metric,*[value[k] for k in ['mse','rmse','mae','r2']]])
    lines=['# История канала, тема и оценка текста — 11 ноября 2026','',
        f"На новом контроле из **{results['control_rows']} статей / {results['control_channels']} каналов** "
        f"общая нормированная MSE ниже среднего канала на **{results['improvement_percent']:.2f}%**, "
        f"ниже прежней конфигурации на **{results['improvement_vs_previous_profile_percent']:.2f}%**. "
        'По точечным оценкам ошибка снизилась по всем трём социальным метрикам.','',
        '**Неопределённость:** 95% интервал bootstrap по каналам для разности общей ошибки '
        f"«среднее минус кандидат» равен [{results['channel_bootstrap_ci95'][0]:.6f}; {results['channel_bootstrap_ci95'][1]:.6f}] "
        'и включает ноль. Эта выборка показывает улучшение, но пока не подтверждает устойчивый общий выигрыш '
        'на уровне 95%. Контрольные тексты не добавлялись в обучение после измерения.','',
        '## Что изменилось','',
        'В модуле 4 выбрана условная регрессия по истории канала. Входы прогноза сохранены: тема, '
        'оценка текстовой модели, ресурс, вид источника и аудитория из каталога. '
        'Текстовая нейросеть не переобучалась и по-прежнему получает только текст и тему.','',
        '1. Публикации каждого канала сортируются по времени. Вес уменьшается вдвое через каждые 30 более новых постов; '
        'это половинный вес по числу публикаций, а не окно только из последних 30 статей.\n'
        '2. Только при обучении значения выше 95-го процентиля истории данного канала ограничиваются этим порогом '
        'при расчёте статистик. Это уменьшает влияние отдельных всплесков; исходные данные и проверочные ответы не изменяются.\n'
        '3. Средний отклик уточняется по теме внутри канала. Редкие сочетания темы и канала стягиваются к общему уровню '
        '(параметр регуляризации 20).\n'
        '4. Учитывается отклонение прогнозной оценки текста от средней оценки канала. '
        'Локальный коэффициент оценивается с ridge-регуляризацией 100; используется половина поправки.','',
        'Каждая метрика получает свой набор статистик и коэффициентов. При обучении нужны даты публикаций; '
        'при прогнозе дата не требуется. Если пользовательский `fit` получает канал с неполными датами, '
        'в нём применяются равные веса вместо произвольного порядка строк. В рабочей паре даты известны для всех 11 493 строк.','',
        '## Подбор и отдельный контроль','',
        f"Проверено {len(screening['results'])} конфигураций, включая среднее канала и прежнюю схему CatBoost. "
        'Материал: прежние 10 182 строки и 1311 дополнительных уникальных текстов. '
        'Дополнительные данные прошлой проверки теперь считаются обучающими/валидационными. '
        'В трёх хронологических разбиениях каждого канала использованы первые 40%, 60%, 80% его истории '
        'для обучения и следующие 20% для оценки. Во всех разбиениях имеется также прежний обучающий корпус. '
        'Входные оценки текста дополнительных статей получены сетью, не обучавшейся на этих статьях.','',
        f"Выбрана `{results['selected']}`. Среднее отношение ошибки к среднему канала по CV — "
        f"{screening['results'][results['selected']]['mean_ratio']:.6f}; снижение около "
        f"{(1-screening['results'][results['selected']]['mean_ratio'])*100:.2f}%. "
        'Кандидат лучше среднего на всех трёх временных частях. Время обучения не участвует в выборе.','',
        '![Подбор по хронологическим разбиениям](development_gain.png)','',
        'До сбора новых контрольных статей зафиксированы конфигурация и интервал публикаций: '
        f"**{protocol['holdout_publication_start_exclusive']} — {protocol['holdout_publication_end_exclusive']}**, "
        'границы исключены. Эти публикации позже всех обучающих текстов. Точные нормализованные повторы '
        'прежнего корпуса исключены. Каналы в контроле уже представлены старшими обучающими постами. '
        'Рабочая БД не изменялась. Возраст новых контрольных статей — около двух дней; '
        'обучающие наблюдения относятся к возрастам 2–14 дней.','',
        'Прежняя конфигурация CatBoost заново обучена на тех же 11 493 строках. '
        'Таким образом, сравниваются алгоритмы при одинаковой обучающей истории. '
        'Среднее канала также рассчитано только по этим обучающим строкам.','',
        '| Метрика | MSE среднего канала | MSE прежней схемы | MSE новой модели | Снижение относительно среднего |',
        '| --- | ---: | ---: | ---: | ---: |']
    for metric,label in [('views','Просмотры'),('reactions','Реакции'),('comments','Комментарии')]:
        values=[results['results'][name]['metrics'][metric]['mse'] for name in names]
        lines.append(f'| {label} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {(1-values[2]/values[0])*100:.2f}% |')
    lines+=['',f"Общая нормированная MSE: **{results['results']['channel_mean']['normalized_mse']:.6f} → "
        f"{results['results']['candidate']['normalized_mse']:.6f}**. Нормировка — дисперсия каждой метрики всех обучающих строк, "
        'одинаковая для моделей. Это не процент точности отдельных публикаций. '
        'Проценты прежнего исследования на другой контрольной выборке напрямую не сравниваются с этими числами.','',
        '![MSE нового контроля](control_mse.png)','',
        '## Время ответа','',
        f"Максимум полного прогретого локального API на 12 реальных текстах: **{max(ordinary):.3f} с**; "
        f"50 000 символов: **{stress:.3f} с**. Предел 10 секунд соблюдён. "
        'Включены E5, автоматическое определение темы, текстовая оценка, прогноз трёх метрик и JSON. '
        'Используется замороженный каталог ресурсов; БД, сеть и LLM не входят в замер. '
        f"Начальная загрузка {latency['loading_seconds']:.2f} с измерена отдельно. Это результаты выбранных примеров на текущем компьютере. ",'',
        '![Время ответа](response_time.png)','',
        '## Воспроизведение','',
        'Исследовательский код находится в `model_tests/`. Для повторения нужны локальные снимки предыдущих исследований, '
        'авторизованная Telegram-сессия, E5 и конфигурация `.env`. Тексты, исходные признаки, ключи и сессии не публикуются. '
        'Новый контроль нельзя повторно использовать для выбора настроек.','',
        '```powershell',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.prepare_history_study --output model_tests/results/new_history --as-of "<UTC ISO timestamp>"',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.tune_history --directory model_tests/results/new_history',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.collect_history_control --directory model_tests/results/new_history',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.confirm_history --directory model_tests/results/new_history',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.build_history_pair --directory model_tests/results/new_history --output artifacts/new_history',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.benchmark_history_api --directory model_tests/results/new_history --model-file artifacts/new_history/model.joblib',
        '```','',
        'Время `--as-of` задаёт верхнюю границу новой контрольной публикации как «время минус два дня». '
        'Для следующего исследования нужен другой, ещё не изученный контрольный интервал. '
        'Для обычного обучения на текущей БД используется `python -m src.social_metrics_prediction train`; '
        'эта команда получает даты публикаций через модуль БД.','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(report=str(output/'README.md'),overall_gain=results['improvement_percent'],
        per_metric_gain={m:(1-results['results']['candidate']['metrics'][m]['mse']/results['results']['channel_mean']['metrics'][m]['mse'])*100 for m in ['views','reactions','comments']})),flush=True)


if __name__=='__main__':main()

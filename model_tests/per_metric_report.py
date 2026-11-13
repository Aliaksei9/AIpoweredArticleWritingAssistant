"""Public aggregate report for independent heads and nested full-corpus checks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    directory=args.directory;out=directory/'public';out.mkdir(exist_ok=True)
    report=json.loads((directory/'nested_report.json').read_text(encoding='utf8'))
    audit=json.loads((directory/'audit.json').read_text(encoding='utf8'))
    baseline=json.loads((directory/'text_baseline.json').read_text(encoding='utf8'))
    root=Path(__file__).resolve().parents[1]
    for kind,path in [('social','artifacts/social_metrics/model.joblib'),('text','artifacts/text_models/neural_network.joblib')]:
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==audit[kind+'_sha256']
    for name in ['nested_report.json','audit.json','optimized_modules_audit.json','embedding_storage_check.json','text_baseline.json']:
        shutil.copy2(directory/name,out/name)
    names=['views','reactions','comments'];labels=['Просмотры','Реакции','Комментарии']
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['metric','baseline_mse','model_mse','baseline_rmse','model_rmse','mse_reduction_percent','passes_5_percent'])
        for m in names:
            a=report['results']['baseline'][m];b=report['results']['candidate'][m]
            writer.writerow([m,a['mse'],b['mse'],a['rmse'],b['rmse'],report['gains_percent'][m],report['each_metric_meets_five_percent'][m]])
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.bar(range(3),[report['gains_percent'][m] for m in names],color='#6081b5')
    ax.axhline(5,color='#c67d51',linestyle='--',label='Требуемое снижение MSE 5%')
    ax.set(xticks=range(3),xticklabels=labels,ylabel='Снижение MSE относительно среднего канала, %',title='Вложенная проверка всей выборки · 11 709 статей')
    ax.legend();ax.grid(axis='y',alpha=.2)
    for ext in ['png','svg']:fig.savefig(out/('metric_gains.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
    lines=['# Независимые модели социальных метрик: проверка всей выборки','',
        '**Порог снижения MSE на 5% для каждой метрики не достигнут. Рабочие веса не заменены.** '
        'Реализованы отдельные классы `ViewsModel`, `ReactionsModel`, `CommentsModel`; '
        'каждый получает собственную конфигурацию и собственный выбор по ошибке своей целевой метрики.', '',
        '## Протокол','',
        'Все 11 709 статей (11 549 нормализованных текстов, 146 каналов) проверены ровно один раз '
        'в пяти внешних групповых разбиениях. Совпадающие тексты не пересекаются. '
        'Внутри каждой обучающей части выбрана конфигурация текстовой сети из трёх вариантов, '
        'заново получены её обучающие OOF-прогнозы и прогнозы внешней проверки. '
        'Для каждого социального показателя независимо выбирается одна из 114 конфигураций: '
        'история/тема/локальная калибровка, ridge, ExtraTrees, бустинг или CatBoost; '
        'рассматриваются абсолютные и относительные остатки. '
        'Внешние проверочные ответы не участвуют в подборе. Среднее канала вычисляется на той же обучающей части.', '',
        'Внутренняя настройка социальных моделей использует только обучающие upstream OOF-оценки; '
        'внешний тест исключён из всей цепочки обучения и выбора. Корпус уже исследовался ранее: '
        'это вложенная проверка имеющихся данных, а не новый независимо собранный временной контроль.', '',
        '| Метрика | MSE среднего | MSE модели | Снижение MSE | Порог 5% |',
        '| --- | ---: | ---: | ---: | --- |']
    for m,label in zip(names,labels):
        lines.append(f"| {label} | {report['results']['baseline'][m]['mse']:.2f} | {report['results']['candidate'][m]['mse']:.2f} | {report['gains_percent'][m]:.2f}% | не достигнут |")
    lines+=['','![Снижение ошибки и порог](metric_gains.png)','',
        '## Настройки отдельных моделей','',
        '| Внешняя часть | Просмотры | Реакции | Комментарии | Текстовая сеть |',
        '| --- | --- | --- | --- | --- |']
    for fold in report['folds']:
        lines.append('| '+str(fold['fold'])+' | '+' | '.join(fold['selected'][m] for m in names)+' | '+fold['text_selected']+' |')
    lines+=['','В `outer_1`…`outer_5` локально сохранены отдельные `views.joblib`, `reactions.joblib`, '
        '`comments.joblib` и данные выбора конфигураций. Они не включены в публичную папку и не активированы в приложении.','',
        '## Текстовая нейросеть','',
        f"Фиксированная текущая конфигурация: MSE полной OOF-проверки **{baseline['mse']:.4f}**. "
        f"Внутренний выбор конфигураций: **{report['text_oof_mse']:.4f}**, "
        f"снижение примерно **{(1-report['text_oof_mse']/baseline['mse'])*100:.2f}%**. "
        'Вход остаётся «текст и тема». Поскольку требование качества социальных моделей не выполнено, '
        'согласованная рабочая пара не заменена. Время обучения не используется как критерий выбора.','',
        '## Ограничения и проверка оптимизированных модулей','',
        'Все 151 исходная модульная проверка прошли до изменения конфигурации эмбеддингов. '
        'Численная формула `log-activity-v3` проверена на 10 000 наборов входов; сохранённые метки согласуются '
        'с оптимизированной реализацией. Рабочие артефакты успешно загружаются, выходы конечны и ограничены. '
        'Новые признаки для 1527 дополнительных текстов получены текущей реализацией E5; '
        '10 182 старых признака взяты из локального исторического снимка с известной моделью.','',
        'Прямая проверка БД обнаружила незавершённую миграцию: отсутствует столбец '
        '`article_vectors.embedding_model`. Обучение и RAG по этим векторам блокируются до '
        '`python -m src.database reembed`. Эта проверка использовала замороженный корпус и БД не изменяла.','',
        'В прежней полной OOF-проверке одна статья со 160 347 реакциями давала примерно 96,18% '
        'суммы квадратов ошибок среднего канала по реакциям. Запись сохранена в проверке; '
        'большие наблюдения не удалялись из контрольных ответов ради выполнения порога.','',
        '## Воспроизведение','',
        '```powershell',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.prepare_per_metric --output model_tests/results/new_per_metric',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.nested_per_metric --directory model_tests/results/new_per_metric',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.text_default_full_baseline --directory model_tests/results/new_per_metric',
        '```','',
        'Требуются локальные исторические снимки; после смены эмбеддера старые матрицы признаков '
        'повторно использовать нельзя. Методика вложенного выбора: '
        '[scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(status='reported',promotion_allowed=False,gains_percent=report['gains_percent'])),flush=True)

if __name__=='__main__':main()

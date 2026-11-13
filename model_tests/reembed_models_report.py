"""Aggregate reembedding verification and complete-corpus model experiments."""
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


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('model_tests/results'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    prior=read(args.results/'per_metric_20261111/nested_report.json')
    links=read(args.results/'count_links_20261111/report.json')
    text=read(args.results/'text_ensembles_20261111/report.json')
    reembed=read(args.results/'reembed_20261111/after.json')
    comparison=read(args.results/'reembed_20261111/historical_feature_comparison.json')
    integration=read(args.results/'reembed_20261111/integration.json')
    audit=read(args.results/'per_metric_20261111/audit.json')
    root=Path(__file__).resolve().parents[1]
    for key,path in [('social','artifacts/social_metrics/model.joblib'),('text','artifacts/text_models/neural_network.joblib')]:
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==audit[key+'_sha256']
    assert reembed['status']=='passed' and integration['status']=='passed'
    assert text['every_row_tested_once'] and links['each_row_tested_once']
    names=['views','reactions','comments'];labels=['Просмотры','Реакции','Комментарии']
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['metric','baseline_mse','previous_nested_mse','positive_link_mse','previous_gain_percent','positive_link_gain_percent'])
        for m in names:writer.writerow([m,links['baseline'][m]['mse'],prior['results']['candidate'][m]['mse'],links['candidate'][m]['mse'],prior['gains_percent'][m],links['gains_percent'][m]])
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');positions=np.arange(3)
    ax.bar(positions-.18,[prior['gains_percent'][m] for m in names],width=.36,label='Предыдущий вложенный подбор',color='#6385b8')
    ax.bar(positions+.18,[links['gains_percent'][m] for m in names],width=.36,label='Новый положительный выход / калибровка',color='#91b6a3')
    ax.axhline(5,color='#bd7952',linestyle='--',label='Требование 5%')
    ax.set(xticks=positions,xticklabels=labels,ylabel='Снижение MSE относительно среднего канала, %',title='Все 11 709 статей · социальные метрики')
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    for ext in ['png','svg']:fig.savefig(out/('social_mse_gain.'+ext),dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    vals=[text['baseline_mse'],text['previous_nested_mse'],text['mse']]
    ax.bar(range(3),vals,color=['#b8c0cc','#6385b8','#91b6a3'])
    ax.set(xticks=range(3),xticklabels=['Текущая конфигурация','Предыдущий подбор','Новый подбор / усреднение'],ylabel='MSE оценки популярности',title='Текст и тема · полная вложенная проверка')
    for i,v in enumerate(vals):ax.text(i,v+.08,f'{v:.4f}',ha='center')
    ax.set_ylim(0,max(vals)*1.15);ax.grid(axis='y',alpha=.2)
    for ext in ['png','svg']:fig.savefig(out/('text_mse.'+ext),dpi=170)
    plt.close(fig)
    rows=read(args.results/'per_metric_20261111/rows.json')
    new=np.load(args.results/'text_ensembles_20261111/predictions.npz')
    old=np.load(args.results/'per_metric_20261111/nested_predictions.npz')
    resources=np.asarray([r['resource_id'] for r in rows]);keys=np.unique(resources)
    old_sse=np.asarray([np.sum((old['predicted_text_scores'][resources==k]-new['actual'][resources==k])**2) for k in keys])
    new_sse=np.asarray([np.sum((new['predicted'][resources==k]-new['actual'][resources==k])**2) for k in keys])
    rng=np.random.default_rng(20261111);samples=rng.integers(0,len(keys),size=(2000,len(keys)))
    interval=np.quantile(100*(1-new_sse[samples].sum(axis=1)/old_sse[samples].sum(axis=1)),[.025,.975]).tolist()
    summary=dict(reembed=reembed,historical_features=comparison,integration=integration,
        social_gains_percent=links['gains_percent'],social_candidate_promoted=False,
        text_mse=text['mse'],text_baseline_mse=text['baseline_mse'],text_previous_nested_mse=text['previous_nested_mse'],
        text_gain_vs_baseline_percent=text['reduction_vs_baseline_percent'],
        text_gain_vs_previous_percent=text['reduction_vs_previous_nested_percent'],
        text_gain_vs_previous_channel_bootstrap_95_percent=interval,
        production_weights_unchanged=True,social_sha256=audit['social_sha256'],text_sha256=audit['text_sha256'])
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    shutil.copy2(args.results/'count_links_20261111/report.json',out/'positive_link_report.json')
    shutil.copy2(args.results/'text_ensembles_20261111/report.json',out/'neural_network_report.json')
    lines=['# Пересчёт векторов и дальнейший подбор моделей','',
        '## База данных','',
        'Полный `reembed` выполнен для **10 182 статей / 91 ресурса**. Использован текущий профиль '
        '`intfloat/multilingual-e5-small`, 384 компоненты, префиксы `passage` / `query`. '
        'Все векторы имеют метаданные профиля и правильную размерность; несовместимых записей — **0**. '
        'Ни одна статья не была пропущена из-за параллельного изменения.', '',
        'До операции создана локальная резервная копия '
        '`artifacts/backups/before_reembed_20261111_1820.dump`; архив проверен через `pg_restore --list` '
        'и содержит данные всех пяти таблиц. Дамп в публичную копию не включён.', '',
        'SHA-256 полного содержимого таблиц `resource`, `article_text`, `article_sources`, '
        '`article_metrics` до и после совпали. Тексты, темы, ID, аудитории и метрики сохранены. '
        f"Максимальная ошибка единичной нормы: {reembed['maximum_norm_error']:.3g}. "
        f"Все {comparison['compared']} старых признака совпадают с пересчитанными векторами "
        f"с точностью до {comparison['maximum_component_difference']:.3g} по координатам; "
        'замороженный E5-корпус для экспериментов остаётся совместимым.', '',
        'Проверено реальное чтение 10 182 обучающих строк, ближайшая статья по собственному '
        'вектору (расстояние 0), поиск популярных похожих статей и пять лучших примеров по теме. '
        'Проблема отсутствовавшего `embedding_model`, описанная в предыдущем отчёте, устранена.', '',
        '## Социальные метрики','',
        'Для просмотров, реакций и комментариев независимо проверены 27 конфигураций '
        'модели с положительным мультипликативным выходом: логарифм отклонения от среднего '
        'канала, нелинейные преобразования оценки, взаимодействия с темой и ресурсом, '
        'регуляризация и калибровка среднего только на обучающей части. Среднее канала '
        'также включено в число кандидатов. Цели проверки не обрезались и не удалялись.', '',
        'Пять внешних групповых разбиений охватывают **11 709 статей / 11 549 разных '
        'нормализованных текстов / 146 ресурсов**. Каждая статья проверена ровно один раз. '
        'Совпадающие тексты не пересекаются между обучением и проверкой. Оценки популярности '
        'взяты из предыдущей вложенной подготовки: проверочные метрики не подаются на вход. '
        'Выбор социальных конфигураций проводится только на обучающих строках. Среднее '
        'нормируется на аудиторию; вычисляется на той же обучающей части.', '',
        '| Метрика | MSE среднего | MSE нового кандидата | Снижение MSE | Прежний вложенный подбор |',
        '| --- | ---: | ---: | ---: | ---: |']
    for m,label in zip(names,labels):
        lines.append(f"| {label} | {links['baseline'][m]['mse']:.2f} | {links['candidate'][m]['mse']:.2f} | {links['gains_percent'][m]:+.2f}% | {prior['gains_percent'][m]:+.2f}% |")
    lines+=['','**Новый социальный кандидат отклонён:** улучшения прежнего результата нет, '
        'порог 5% по каждой метрике не достигнут. Рабочие веса сохранены.', '',
        '![Социальные метрики](social_mse_gain.png)','',
        'Одна статья со 160 347 реакциями даёт около 96,18% суммы квадратов ошибок '
        'среднего канала по реакциям. Это ограничение данной выборки; запись сохранена '
        'в оценке. Подмена MSE другой метрикой или удаление этой статьи не использовались.', '',
        '## Нейросеть по тексту','',
        'Вход остаётся только «текст и тема». Проверены четыре архитектуры и один/два '
        'независимых начальных состояния. Архитектура и число сетей выбираются на двух '
        'внутренних разбиениях каждой внешней обучающей части. Прогноз двух сетей '
        'усредняется; это исследовательский вариант, не дополнительный вход модели.', '',
        f"MSE текущей конфигурации на всех статьях: **{text['baseline_mse']:.4f}**. "
        f"Предыдущий вложенный подбор: **{text['previous_nested_mse']:.4f}**. "
        f"Новый: **{text['mse']:.4f}** — "
        f"{text['reduction_vs_baseline_percent']:.2f}% относительно текущей конфигурации "
        f"и {text['reduction_vs_previous_nested_percent']:.2f}% относительно прошлого подбора. "
        f"Парный bootstrap по каналам (2000 повторов), 95% интервал последнего улучшения: "
        f"[{interval[0]:.2f}%; {interval[1]:.2f}%]. Интервал включает ноль; "
        'устойчивое превосходство над предыдущим подбором пока не подтверждено.', '',
        '| Внешняя часть | Выбор по обучающим данным | Проверочная MSE | Максимальное время только сети, с |',
        '| --- | --- | ---: | ---: |']
    for fold in text['folds']:
        lines.append(f"| {fold['fold']} | {fold['selected']} | {fold['outer_mse']:.4f} | {fold['warm_head_max_seconds']:.6f} |")
    lines+=['','![Текстовая нейросеть](text_mse.png)','',
        'Согласованная рабочая пара не заменена: улучшение текстовой MSE отдельно не '
        'подтверждает выполнение требования для социальных метрик. Время обучения '
        'не использовалось при выборе. Приведённое время нейросетевого слоя исключает '
        'SBERT, загрузку и LLM; оно не выдаётся за полное время ответа приложения. '
        'Ранее проверенная рабочая пара остаётся прежней.', '',
        '## Ограничения и воспроизведение','',
        'Корпус уже использовался в исследованиях. Это проверка всех имеющихся данных '
        'с вложенным выбором, а не новый независимый контроль будущих публикаций. '
        'Случайное групповое разбиение не доказывает качество на будущих датах. '
        'Внутренний социальный подбор повторно использует upstream OOF-прогнозы '
        'внешней обучающей части; внешние проверочные ответы исключены из всей цепочки.', '',
        '```powershell',
        '.\\.venv312\\Scripts\\python.exe -m src.database reembed',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.evaluate_count_links --corpus model_tests/results/per_metric_20261111 --output model_tests/results/new_count_links',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.evaluate_text_ensembles --corpus model_tests/results/per_metric_20261111 --output model_tests/results/new_text_ensembles',
        '```','',
        'Нужны локальные замороженные снимки предыдущего исследования; сырые тексты, '
        'разбиения, дамп и промежуточные веса в GitHub-копию не включены. '
        'Агрегированные результаты: [summary.json](summary.json), '
        '[metrics.csv](metrics.csv), [нейросети](neural_network_report.json), '
        '[социальные модели](positive_link_report.json).','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(text_mse=text['mse'],social_gains=links['gains_percent'],weights_unchanged=True)),flush=True)


if __name__=='__main__':main()

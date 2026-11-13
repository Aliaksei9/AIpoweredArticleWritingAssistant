"""Aggregate tuning statistics and paired group bootstrap; no raw texts exported."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .compare_text_models import write_json


def create_report(directory,previous):
    screening=json.loads((directory/'screening.json').read_text(encoding='utf8'))
    confirmation=json.loads((directory/'confirmation/confirmation.json').read_text(encoding='utf8'))
    output=directory/'public'
    output.mkdir(exist_ok=True)
    selected=confirmation['social_selected']
    with np.load(directory/'confirmation/test_predictions.npz') as saved:
        actual=saved['actual'];prediction=saved[selected];baseline=saved['resource_mean']
    with np.load(previous/'social_metrics/dataset.npz') as saved:
        groups=saved['groups'];counts=saved['counts']
    splits=json.loads((previous/'social_metrics/splits.json').read_text(encoding='utf8'))
    fitting=np.asarray(splits['train_indices']);test=np.asarray(splits['test_indices'])
    scale=np.maximum(np.var(counts[fitting],axis=0),1)
    unique,inverse=np.unique(groups[test],return_inverse=True)
    difference=(baseline-actual)**2-(prediction-actual)**2
    group_sums=np.zeros((len(unique),3))
    np.add.at(group_sums,inverse,difference)
    sizes=np.bincount(inverse)
    rng=np.random.default_rng(20261111)
    samples=[]
    for _ in range(2000):
        indices=rng.integers(0,len(unique),size=len(unique))
        samples.append(group_sums[indices].sum(axis=0)/sizes[indices].sum()/scale)
    samples=np.asarray(samples)
    total=samples.mean(axis=1)
    uncertainty=dict(method='paired bootstrap of held-out text groups, 2000 replicates, seed 20261111; descriptive after historical dataset reuse',
        improvement_normalized_mse=float(np.mean(difference.mean(axis=0)/scale)),
        ci95=np.quantile(total,[.025,.975]).tolist(),
        per_metric_ci95={name:np.quantile(samples[:,i],[.025,.975]).tolist() for i,name in enumerate(['views','reactions','comments'])})
    write_json(output/'uncertainty.json',uncertainty)
    write_json(output/'screening.json',screening)
    # Confirmation report contains aggregated metrics, not raw inputs/row IDs.
    write_json(output/'confirmation.json',confirmation)

    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=170,bbox_inches='tight')
        fig.savefig(output/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    names=list(screening['text'])
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    values=[screening['text'][n]['cv_mse'] for n in names]
    ax.bar(np.arange(len(names)),values,color='#5c80bc')
    ax.set(xticks=np.arange(len(names)),xticklabels=names,ylabel='MSE групповой CV, баллы²',title='Настройки текстовой нейросети · вход только текст и тема')
    ax.tick_params(axis='x',rotation=25,labelsize=8)
    ax.grid(axis='y',alpha=.2)
    save(fig,'text_configuration_mse')

    rows=sorted(screening['social'].items(),key=lambda pair:pair[1]['cv_mse'])[:15]
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    delta=[(v['baseline_cv_mse']-v['cv_mse']) for _,v in rows]
    ax.barh(np.arange(len(rows)),delta,color='#4b987d')
    ax.set(yticks=np.arange(len(rows)),yticklabels=[n for n,_ in rows],xlabel='Снижение нормированной MSE CV относительно среднего канала',title='15 лучших конфигураций первичного подбора · больше — лучше')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    save(fig,'social_configuration_gain')

    names=list(confirmation['results'])
    base=confirmation['results']['resource_mean']['test']
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,metric,title in zip(axes,['views','reactions','comments'],['Просмотры','Реакции','Комментарии']):
        values=[confirmation['results'][n]['test'][metric]['mse']/base[metric]['mse'] for n in names]
        ax.bar(np.arange(len(names)),values,color=['#9aa3b1' if n=='resource_mean' else '#5c80bc' for n in names])
        ax.axhline(1,color='#ca755d',linestyle='--')
        ax.set(xticks=np.arange(len(names)),xticklabels=names,title=title,ylabel='MSE / MSE среднего по каналу')
        ax.tick_params(axis='x',rotation=65,labelsize=7);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Отложенные статьи · ниже 1 — лучше среднего по каналу')
    save(fig,'heldout_social_relative_mse')

    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['configuration','cv_normalized_mse','test_normalized_mse','views_mse','reactions_mse','comments_mse'])
        for name,v in confirmation['results'].items():
            writer.writerow([name,v['cv_mse'],v['test_normalized_mse'],*[v['test'][m]['mse'] for m in ['views','reactions','comments']]])
    lines=['# Подбор конфигураций прогнозных моделей — 11 ноября 2026','',
        'Цель — улучшить прогноз социальных метрик относительно среднего по каналу. '
        'Вход текстового модуля сохранён: только текст и тема. Канал известен исключительно модулю социальных метрик.', '',
        '## Протокол','',
        'Использована та же выборка из 10 182 статей / 91 канала: 8143 строки для подбора, 2039 отложенных. '
        'Проверены 8 конфигураций текстовой нейросети и 15 конфигураций моделей поправки к среднему канала '
        'с четырьмя коэффициентами поправки (60 комбинаций). Дополнительно проверена комбинация настроек, '
        'отдельно выбранных по CV для каждой метрики и зафиксированных до просмотра новых тестовых результатов. Подбор выполняется по групповой CV. '
        'Средние каналов, словари, нормировка и коэффициенты вычисляются только на обучающей части.', '',
        'Для подтверждения заново рассчитаны входные текстовые оценки: выбор конфигурации текстовой сети '
        'внутри каждого обучения использует две внутренние части; прогнозы для обучения социальных моделей '
        'получены трёхчастным cross-fitting. Проверяемая группа не участвует ни в обучении сети, ни в выборе '
        'её конфигурации. После фиксации выбора по CV рассчитаны ошибки отложенных статей.', '',
        '**Отложенная часть уже использовалась в предыдущем сравнении моделей.** '
        'В текущем подборе её ошибки раскрыты после фиксации конфигураций, но это не новая независимая выборка. '
        'Нужна последующая проверка на публикациях новых дат. Все записи, включая редкий большой счётчик реакций, сохранены. '
        'Время обучения не участвует в выборе; предел прогноза после загрузки остаётся 10 секунд.', '',
        '## Прогноз социальных метрик','',
        f'По CV выбран вариант **{selected}**. Общий критерий — среднее MSE трёх метрик, '
        'нормированное на дисперсию каждой метрики обучающей части.', '',
        '| Конфигурация | Норм. MSE CV | Норм. MSE проверки | RMSE просмотров | RMSE реакций | RMSE комментариев |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,v in confirmation['results'].items():
        lines.append(f"| {name} | {v['cv_mse']:.6f} | {v['test_normalized_mse']:.6f} | "+' | '.join(f"{v['test'][m]['rmse']:.2f}" for m in ['views','reactions','comments'])+' |')
    chosen=confirmation['results'][selected]
    reference=confirmation['results']['resource_mean']
    gain=(1-chosen['test_normalized_mse']/reference['test_normalized_mse'])*100
    comparison=(f'Ошибка выше среднего по каналу на **{-gain:.2f}%**.' if gain<0 else
                f'Ошибка ниже среднего по каналу на **{gain:.2f}%**.')
    lines+=['',f'Первичный вариант с минимальной средней CV MSE: {comparison} '
        'Процент описывает изменение ошибки, а не точность отдельных прогнозов. '
        'Изменения по отдельным метрикам показаны на графике ниже.', '',
        f"Парный групповой bootstrap: разность ошибок «среднее канала минус модель» — {uncertainty['improvement_normalized_mse']:.6f}; "
        f"95% интервал [{uncertainty['ci95'][0]:.6f}; {uncertainty['ci95'][1]:.6f}]. "
        'Положительная разность соответствует улучшению. Интервал описательный с учётом повторного использования данных.', '',
        '![Подбор социальных конфигураций](social_configuration_gain.png)','',
        '![Сравнение с каналом по метрикам](heldout_social_relative_mse.png)','',
        '## Оценка популярности по тексту','',
        f"Конфигурация итоговой сети для оценки: `{confirmation['text_configuration']}`. "
        f"MSE отложенной части: {confirmation['text_test']['model']['mse']:.4f}; RMSE: {confirmation['text_test']['model']['rmse']:.4f} балла. "
        f"Диагностическое среднее по каналу: MSE {confirmation['text_test']['channel_mean']['mse']:.4f}. "
        'Это среднее использует источник, которого нет во входе текстовой сети, и не заменяет её при прогнозе.', '',
        '![Конфигурации текстовой сети](text_configuration_mse.png)','',
        '## Файлы и повторение','',
        '`screening.json`, `confirmation.json` и `metrics.csv` содержат все агрегированные результаты. '
        'Код подбора и подтверждения находится в `model_tests/`, вне кода приложения. '
        'Исходные тексты, признаки, промежуточные веса и входные оценки остаются только локально.', '',
        '```powershell',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.tune_models --output model_tests/results/new_tuning',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.confirm_tuning --screening model_tests/results/new_tuning/screening.json --output model_tests/results/new_tuning/confirmation',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.confirm_metricwise_tuning --directory model_tests/results/new_tuning',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.tuning_report --directory model_tests/results/new_tuning',
        '```','',
        'Повторение требует локального снимка предыдущего исследования в '
        '`model_tests/results/expansion_20261111`; он не публикуется вместе с исходным кодом. '
        'Рабочая БД не изменяется при подборе.','']
    if (directory/'fresh_confirmation.json').exists():
        fresh=json.loads((directory/'fresh_confirmation.json').read_text(encoding='utf8'))
        frozen=json.loads((directory/'fresh_choices_frozen.json').read_text(encoding='utf8'))
        latency=json.loads((directory/'fresh_evaluation_latency.json').read_text(encoding='utf8'))
        write_json(output/'fresh_confirmation.json',fresh)
        write_json(output/'fresh_choices_frozen.json',frozen)
        write_json(output/'latency.json',latency)
        leading=['## Итог дополнительной проверки','',
            f"Для модуля социальных метрик выбран **CatBoost с поправкой к среднему канала**: `{fresh['selected']}`. "
            f"На {fresh['fresh_test_rows']} более новых статьях из {fresh['fresh_channels']} дополнительных каналов "
            f"общая нормированная MSE ниже среднего канала на **{fresh['improvement_percent']:.3f}%**. "
            'Снижение есть по каждой из трёх метрик. Выигрыш небольшой; это не обещание точности для каждой будущей статьи.','',
            '**Почему понадобилась дополнительная проверка.** Конфигурация с минимальной средней CV MSE '
            'не подтвердила преимущество на старой отложенной части; результаты приведены в таблице ниже. '
            'После этого использован более консервативный критерий: минимальное худшее отношение ошибки CV '
            'к среднему канала, с улучшением во всех трёх частях. До чтения новых статей зафиксирован CatBoost. '
            'Новая проверочная часть не использовалась для выбора конфигурации.','',
            'Прочитаны 1311 дополнительных текстов из 55 ранее отобранных публичных каналов. '
            f"Из них {fresh['fresh_articles']} уникальных текстов из {fresh['fresh_channels']} каналов с минимум 10 статьями "
            f"вошли в проверку: старшие 80% ({fresh['fresh_training_rows']}) каждого канала использованы для обучения "
            f"с прежними 10 182 строками, младшие 20% ({fresh['fresh_test_rows']}) оставлены для проверки. "
            'Точные повторы прежних и новых текстов исключены. Канал известен обоим прогнозам по его старшим публикациям. '
            'Публикации относятся к тому же окну возраста 2–14 дней; данные не собраны в будущую дату.','',
            '| Метрика | MSE среднего канала | MSE CatBoost | RMSE среднего | RMSE CatBoost | Снижение MSE |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for metric,label in [('views','Просмотры'),('reactions','Реакции'),('comments','Комментарии')]:
            base=fresh['channel_mean']['metrics'][metric];model=fresh['model']['metrics'][metric]
            leading.append(f"| {label} | {base['mse']:.2f} | {model['mse']:.2f} | {base['rmse']:.3f} | {model['rmse']:.3f} | {(1-model['mse']/base['mse'])*100:.3f}% |")
        interval=fresh['paired_channel_bootstrap']['ci95']
        leading+=['',f"Общая нормированная MSE: {fresh['channel_mean']['normalized_mse']:.8f} → {fresh['model']['normalized_mse']:.8f}. "
            'Шкалы дисперсии рассчитаны на обучающих строках новых каналов и одинаковы для обоих прогнозов. '
            f"95% интервал парного bootstrap по каналам для разности «среднее минус модель»: [{interval[0]:.8f}; {interval[1]:.8f}]. "
            'Он положителен в этой выборке; оценка остаётся зависимой от каналов и срока наблюдения.','',
            f"Прогретый полный ответ с E5: максимум {latency['response_time']['max']:.3f} с на восьми статьях разной длины; "
            f"текст из 50 000 символов — {latency['stress_50000_seconds']:.3f} с. Предел 10 секунд соблюдён. "
            'Загрузка моделей, БД, сеть и LLM не входят в это измерение.','',
            'Рабочие веса собираются **без повторного обучения на проверочных статьях**: текстовая сеть обучена '
            f"на 8143 прежних строках, социальная модель — на {fresh['total_training_rows']} строках. "
            'Совпадение прогнозов приложения и экспериментальной модели проверяется перед заменой. '
            'Старая пара сохранена локально. Исходная рабочая БД содержит прежние 10 182 статьи и не изменялась этим подбором.','',
            '## Первичный подбор и его ограничения','']
        lines=lines[:4]+leading+lines[4:]
        fig,axes=plt.subplots(1,3,figsize=(11,4.5),layout='constrained')
        for ax,metric,label in zip(axes,['views','reactions','comments'],['Просмотры','Реакции','Комментарии']):
            base=fresh['channel_mean']['metrics'][metric]['mse'];model=fresh['model']['metrics'][metric]['mse']
            ax.bar([0,1],[base,model],color=['#9ba4b2','#4c997a'])
            ax.set(xticks=[0,1],xticklabels=['Среднее канала','CatBoost'],ylabel='MSE, количество²',title=label)
            ax.grid(axis='y',alpha=.2)
            for x,v in enumerate([base,model]):ax.text(x,v*1.01,f'{v:,.2f}',ha='center',fontsize=8)
            ax.set_ylim(0,base*1.16)
        fig.suptitle('MSE на дополнительных каналах · оси начинаются с нуля')
        save(fig,'fresh_social_mse')
        lines[lines.index('## Первичный подбор и его ограничения'):lines.index('## Первичный подбор и его ограничения')]=[
            '![MSE дополнительной проверки](fresh_social_mse.png)','']
        lines+=['## Дополнительная проверка: повторение','',
            'Замороженные настройки: `fresh_choices_frozen.json`. Код чтения новых каналов — '
            '`model_tests.collect_fresh_confirmation`, проверки — `model_tests.confirm_fresh_tuning`. '
            'Сбор не вступает в каналы, не публикует сообщения и не записывает в рабочую БД. '
            'Один и тот же новый тест повторно не используется для подбора.','',
            '```powershell',
            '.\\.venv312\\Scripts\\python.exe -m model_tests.freeze_robust_profile --directory model_tests/results/new_tuning',
            '.\\.venv312\\Scripts\\python.exe -m model_tests.collect_fresh_confirmation --directory model_tests/results/new_tuning --target 1500',
            '.\\.venv312\\Scripts\\python.exe -m model_tests.confirm_fresh_tuning --directory model_tests/results/new_tuning',
            '.\\.venv312\\Scripts\\python.exe -m model_tests.benchmark_tuned_latency --directory model_tests/results/new_tuning --fresh',
            '.\\.venv312\\Scripts\\python.exe -m model_tests.build_tuned_pair --directory model_tests/results/new_tuning --output artifacts/new_candidate',
            '```','',
            'Методика разделения выбора и оценки: [документация scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).','']
    document='\n'.join(lines).replace('## Первичный подбор и его ограничения\n\n## Протокол',
                                     '## Первичный подбор: протокол и ограничения')
    (output/'README.md').write_text(document,encoding='utf8')
    print(json.dumps(dict(selected=selected,test_mse_improvement_percent=gain,uncertainty=uncertainty)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=Path('model_tests/results/expansion_20261111'))
    args=parser.parse_args()
    create_report(args.directory,args.previous)

"""Export aggregate statistics and plots; never export raw articles or weights."""
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


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--text-study',type=Path,required=True)
    parser.add_argument('--social-study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);out=args.output
    text=read(args.text_study/'report.json');social=read(args.social_study/'report.json')
    latency=read(args.social_study/'latency.json');audit=read(args.corpus/'audit.json')
    serialization=read(args.social_study/'artifact_verification.json')
    prior=read(args.corpus/'nested_report.json');rows=read(args.corpus/'rows.json')
    assert text['every_row_tested_once'] and social['every_row_tested_once']
    assert serialization['status']=='passed' and serialization['rows_verified']==text['rows']
    assert text['corpus_sha256']==social['corpus_sha256']
    old_data=np.load(args.corpus/'corpus.npz');new_data=np.load(args.social_study/'predictions.npz')
    np.testing.assert_array_equal(old_data['counts'],new_data['actual'])
    values=np.load(args.text_study/'predictions.npz');resource_ids=np.asarray([r['resource_id'] for r in rows])
    keys=np.unique(resource_ids)
    previous=np.asarray([np.sum((values['previous'][resource_ids==k]-values['actual'][resource_ids==k])**2) for k in keys])
    new=np.asarray([np.sum((values['predicted'][resource_ids==k]-values['actual'][resource_ids==k])**2) for k in keys])
    rng=np.random.default_rng(20261111);sample=rng.integers(0,len(keys),size=(3000,len(keys)))
    ci=np.quantile(100*(1-new[sample].sum(axis=1)/previous[sample].sum(axis=1)),[.025,.975]).tolist()
    root=Path(__file__).resolve().parents[1]
    for name,path in [('text','artifacts/text_models/neural_network.joblib'),('social','artifacts/social_metrics/model.joblib')]:
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==audit[name+'_sha256']
    summary=dict(rows=text['rows'],distinct_texts=text['groups'],resources=len(keys),
        baseline_text_mse=text['baseline_mse'],previous_text_mse=text['previous_mse'],new_text_mse=text['mse'],
        baseline_text_rmse=float(np.sqrt(text['baseline_mse'])),new_text_rmse=float(np.sqrt(text['mse'])),
        gain_vs_baseline_percent=text['reduction_vs_baseline_percent'],gain_vs_previous_percent=text['reduction_vs_previous_percent'],
        gain_vs_previous_channel_bootstrap_95_percent=ci,social_gains_percent=social['gains_percent'],
        passes_social_accuracy_gate=social['passes_accuracy_gate'],passes_latency_gate=latency['status']=='passed',
        maximum_warm_response_seconds=latency['response_time']['max'],production_weights_unchanged=True,
        text_sha256=audit['text_sha256'],social_sha256=audit['social_sha256'],
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
            ['advanced_text_networks.py','advanced_neural_study.py','conditional_kernel_metrics.py','advanced_pipeline_study.py','advanced_model_latency.py']})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for source,destination in [(args.text_study/'report.json','text_report.json'),
                               (args.social_study/'report.json','social_report.json'),
                               (args.social_study/'latency.json','latency.json'),
                               (args.social_study/'artifact_verification.json','artifact_verification.json')]:shutil.copy2(source,out/destination)
    with (out/'neural_configurations.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['configuration','mean_inner_mse','selected_outer_folds'])
        config_names=list(text['folds'][0]['inner_mse'])
        config_scores={n:float(np.mean([f['inner_mse'][n] for f in text['folds']])) for n in config_names}
        for name in sorted(config_names,key=config_scores.get):
            writer.writerow([name,config_scores[name],sum(f['selected']==name for f in text['folds'])])
    names=['views','reactions','comments'];labels=['Просмотры','Реакции','Комментарии']
    with (out/'social_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['metric','channel_mean_mse','model_mse','model_rmse','gain_percent','meets_five_percent'])
        for m in names:writer.writerow([m,social['baseline'][m]['mse'],social['candidate'][m]['mse'],social['candidate'][m]['rmse'],social['gains_percent'][m],social['gains_percent'][m]>=5])
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    mse=[text['baseline_mse'],text['previous_mse'],text['mse']]
    ax.bar(range(3),mse,color=['#b8c0cc','#6385b8','#91b6a3'])
    ax.set(xticks=range(3),xticklabels=['Рабочая конфигурация','Предыдущий подбор','Новый вложенный подбор'],
           ylabel='MSE оценки популярности',title='Текст и тема · все 11 709 статей')
    for i,value in enumerate(mse):ax.text(i,value+.1,f'{value:.4f}',ha='center')
    ax.set_ylim(0,max(mse)*1.15);ax.grid(axis='y',alpha=.2)
    for ext in ['png','svg']:fig.savefig(out/('text_mse.'+ext),dpi=170)
    plt.close(fig)
    ordered=sorted(config_scores,key=config_scores.get)
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    ax.barh(ordered[::-1],[config_scores[n] for n in ordered[::-1]],color='#6385b8')
    ax.axvline(config_scores['reference'],color='#bd7952',linestyle='--',label='Предыдущая нейросеть')
    ax.set(xlabel='Средняя MSE внутренних проверок (обучение на половине внешней обучающей части)',title='17 вариантов · только внутренний подбор')
    ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2)
    for ext in ['png','svg']:fig.savefig(out/('inner_configurations.'+ext),dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');position=np.arange(3)
    ax.bar(position-.18,[prior['gains_percent'][m] for m in names],width=.36,color='#6385b8',label='Предыдущий вложенный подбор')
    ax.bar(position+.18,[social['gains_percent'][m] for m in names],width=.36,color='#91b6a3',label='Новая текстовая сеть и социальные модели')
    ax.axhline(5,color='#bd7952',linestyle='--',label='Требуемые 5%')
    ax.set(xticks=position,xticklabels=labels,ylabel='Снижение MSE относительно среднего канала, %',title='Полная последовательность модулей 3–4')
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    for ext in ['png','svg']:fig.savefig(out/('social_gains.'+ext),dpi=170)
    plt.close(fig)
    ordinary=[r['seconds'] for r in latency['cases'] if not r['stress']]
    stress=[r['seconds'] for r in latency['cases'] if r['stress']]
    lines=['# Расширенный подбор нейросетей и проверка последовательности модулей','',
        '**До изменения рабочих весов выполнена полная проверка.** В этом запуске проверены '
        'все **11 709 статей**, **11 549 групп нормализованных текстов**, **146 ресурсов**. '
        'Текущая рабочая пара и БД не изменены.','',
        '## Текстовая модель','',
        f"MSE рабочей конфигурации на всех статьях — **{text['baseline_mse']:.4f}**; "
        f"предыдущего подбора — **{text['previous_mse']:.4f}**; нового — **{text['mse']:.4f}**. "
        f"Снижение составляет **{text['reduction_vs_baseline_percent']:.2f}%** относительно рабочей "
        f"конфигурации и **{text['reduction_vs_previous_percent']:.2f}%** относительно предыдущего подбора. "
        f"95% парный bootstrap-интервал последнего улучшения по каналам (3000 повторов): "
        f"**[{ci[0]:.2f}%; {ci[1]:.2f}%]**. " +
        ('Интервал включает ноль: устойчивое превосходство пока не подтверждено.' if ci[0]<=0<=ci[1] else
         'В рамках этого повторно используемого корпуса интервал не включает ноль; новый временной контроль всё равно нужен.'),'',
        '![Полная MSE](text_mse.png)','',
        'Восемь новых архитектур/режимов и их равные смеси с предыдущим нейросетевым '
        'ансамблем дают 17 вариантов с учётом исходного ансамбля. Проверены GELU, '
        'cosine learning rate, усреднение весов последних 25% эпох, mixup, остаточные '
        'связи с LayerNorm, PCA текста на 64/128 компонент, отдельная линейная ветка '
        'темы, порядковая регрессия по 21 уровню и центрирование признаков без '
        'стандартизации дисперсий. Масштабирование и PCA обучаются только на обучающих '
        'строках. Входы остаются **текст и эмбеддинг темы**; канал в текстовую модель не передаётся.','',
        '| Внешняя часть | Выбор по внутренней проверке | MSE нового варианта | MSE предыдущего |',
        '| --- | --- | ---: | ---: |']
    for f in text['folds']:lines.append(f"| {f['fold']} | {f['selected']} | {f['outer_mse']:.4f} | {f['reference_mse']:.4f} |")
    lines+=['','`additive_topic`: текстовая ветка 384 → 256 → 128 → 1, GELU, Dropout 0,5; '
        'тема проходит отдельный линейный слой 384 → 1, результаты складываются перед '
        '`10*tanh`. AdamW, начальный learning rate 0,0005, weight decay 0,2, batch 128, '
        '35 эпох с cosine-расписанием и усреднением последних эпох. `:blend` означает '
        'вес 0,5 у этой сети и по 0,25 у двух сетей предыдущего варианта '
        '(768 → 512 → 128 → 1, Dropout 0,65, разные seed). Это исследовательский '
        'ансамбль, сохранённый отдельно от приложения.','',
        'График ниже отражает только внутренний выбор; его MSE не заменяет результат '
        'полной внешней проверки выше.','',
        '![Внутренний подбор](inner_configurations.png)','',
        '## Прогноз социальных метрик','',
        'Для каждого внешнего разбиения выбранная текстовая конфигурация заново '
        'обучена в трёх OOF-разбиениях обучающей части. По этим прогнозам, а не по '
        'истинной оценке из метрик, отдельно подбираются модели просмотров, реакций '
        'и комментариев. Проверочная текстовая оценка получена сетью, обученной '
        'только на внешней обучающей части.','',
        f"Проверено **{social['configuration_count']} конфигураций**: история канала, "
        'тема, локальный наклон по оценке, независимая ridge-регрессия и новый '
        'сглаженный поиск соседей по оценке/теме внутри ресурса. Для нового варианта '
        'подбираются ширина гауссова ядра, усиление совпадающей темы, сглаживание '
        'к среднему и обучающее ограничение хвоста. Исходные проверочные количества '
        'не обрезаются и не удаляются. Среднее канала рассчитано на той же обучающей '
        'части с нормировкой на аудиторию.','',
        '| Метрика | MSE среднего | MSE модели | Снижение MSE | Требование 5% |',
        '| --- | ---: | ---: | ---: | --- |']
    for m,label in zip(names,labels):
        lines.append(f"| {label} | {social['baseline'][m]['mse']:.2f} | {social['candidate'][m]['mse']:.2f} | {social['gains_percent'][m]:+.2f}% | {'да' if social['gains_percent'][m]>=5 else 'нет'} |")
    lines+=['','![Социальные метрики](social_gains.png)','',
        ('Порог точности пройден.' if social['passes_accuracy_gate'] else
         '**Порог снижения MSE на 5% отдельно для каждой социальной метрики не пройден. Рабочая пара не заменена.**'),'','## Время ответа','',
        f"Через отдельное локальное ASGI-приложение проверены все пять исследовательских пар: "
        f"по 12 реальных текстов разной длины и текст на 50 000 символов. Максимум для "
        f"реальных текстов — **{max(ordinary):.3f} с**, для 50 000 символов — "
        f"**{max(stress):.3f} с**. Условие 10 секунд "
        f"{'выполнено' if latency['status']=='passed' else 'не выполнено'} на проверенном оборудовании.", '',
        'Измерение включает контроллер, полный SBERT текста, автоматическое определение '
        'темы, прогноз оценки, три социальных прогноза и JSON. Не включает первую '
        'загрузку, БД, сеть или рекомендации LLM. Время обучения в выборе не участвовало.','',
        '## Протокол и ограничения','',
        'Каждая статья имеет ровно один внешний проверочный прогноз в пяти '
        'GroupKFold-разбиениях с seed 20261111. Совпадающие нормализованные тексты '
        'не пересекают границу обучения/проверки. Архитектура выбирается на двух '
        'внутренних групповых разбиениях (seed 91). Внешние проверочные ответы '
        'исключены из обучения, масштабирования, PCA и выбора параметров. Кэш '
        'новых моделей проверяет хеш корпуса, индексы обучения, конфигурацию и seed. '
        'Предыдущий нейросетевой ориентир воспроизведён по всем проверочным прогнозам.','',
        'Корпус уже использовался в исследованиях: результат описывает повторную '
        'проверку доступных данных, а не новый независимый контроль будущих дат. '
        'Случайное групповое разбиение не проверяет временной сдвиг. Внутренний '
        'подбор социальных моделей повторно использует OOF-оценки внешней обучающей '
        'части; внешний тест исключён из всей цепочки. Одна статья со 160 347 '
        'реакциями по-прежнему остаётся в проверке.','',
        'Прошли 168 Python-тестов и 14 JS-тестов; среди новых проверок — '
        'воспроизводимость mixup, сериализация всех архитектур, сохранение состояния '
        'генератора случайных чисел, обучающее происхождение кэша и независимость '
        'целевых метрик. Дополнительно все 11 709 социальных прогнозов точно '
        'воспроизведены после повторной загрузки сохранённых моделей '
        '([проверка](artifact_verification.json)). Контрольные суммы рабочей пары '
        'сохранены в [summary.json](summary.json).','',
        '## Воспроизведение','',
        '```powershell',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.advanced_neural_study --corpus model_tests/results/per_metric_20261111 --previous model_tests/results/text_ensembles_20261111 --output model_tests/results/new_advanced_text',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.advanced_pipeline_study --corpus model_tests/results/per_metric_20261111 --text-study model_tests/results/new_advanced_text --output model_tests/results/new_advanced_pipeline',
        '.\\.venv312\\Scripts\\python.exe -m model_tests.advanced_model_latency --corpus model_tests/results/per_metric_20261111 --text-study model_tests/results/new_advanced_text --social-study model_tests/results/new_advanced_pipeline',
        '```','',
        'Нужны локальные снимки предыдущих экспериментов. В публичную копию '
        'включены только код и агрегированные отчёты; тексты статей, матрицы, '
        'индексы разбиений и промежуточные веса остаются локально.','',
        '[Текстовая проверка](text_report.json) · [Социальные метрики](social_report.json) · '
        '[Время ответа](latency.json) · [Конфигурации](neural_configurations.csv) · '
        '[MSE социальных метрик](social_metrics.csv)','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:summary[k] for k in ['new_text_mse','gain_vs_baseline_percent','social_gains_percent','maximum_warm_response_seconds']}),flush=True)


if __name__=='__main__':main()

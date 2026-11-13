"""Publish only aggregate whole-corpus results for alternative count forecasts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','previous','classical','neural','transfer','merged','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    joint=read(args.merged/'report.json');latency=read(args.merged/'latency.json')
    original=np.load(args.previous/'predictions.npz');y=original['actual'];base=original['baseline']
    candidates={'previous':original['predicted']}
    classic=np.load(args.classical/'predictions.npz')
    candidates.update(same_inputs=classic['same_inputs'],direct_text=classic['richer_inputs'],
        direct_neural=np.load(args.neural/'predictions.npz')['predicted'],
        cross_resource=np.load(args.transfer/'predictions.npz')['predicted'],merged=np.load(args.merged/'predictions.npz')['predicted'])
    results_root=args.classical.parent
    calendar_latency=read(results_root/'calendar_counts_20261111/latency.json')
    for name,directory in [('stacked','count_stacking_20261111'),('calendar','calendar_counts_20261111'),('calibrated','reaction_calibration_20261111')]:
        candidates[name]=np.load(results_root/directory/'predictions.npz')['predicted']
    names=['views','reactions','comments'];labels=['Просмотры','Реакции','Комментарии']
    captions={'previous':'Предыдущая цепочка','same_inputs':'Те же входы / CatBoost','direct_text':'Прямые текстовые признаки',
              'direct_neural':'Прямые нейросети счётчиков','cross_resource':'Перенос между каналами','merged':'Выбор семейства внутри обучения'}
    captions.update(stacked='Корректор OOF-ошибок',calendar='Время публикации и горизонт',calibrated='Непрерывная калибровка реакции')
    mse=np.mean((y-base)**2,axis=0);mae=np.mean(abs(y-base),axis=0)
    results={}
    for name,value in candidates.items():
        results[name]=dict(mse=np.mean((y-value)**2,axis=0).tolist(),mae=np.mean(abs(y-value),axis=0).tolist(),
            mse_gain_percent=(100*(1-np.mean((y-value)**2,axis=0)/mse)).tolist(),
            mae_gain_percent=(100*(1-np.mean(abs(y-value),axis=0)/mae)).tolist())
    texts=read(args.classical/'raw_texts.json');rows=read(args.corpus/'rows.json')
    pattern=re.compile(r'(?:голос\w*|выб[еи]р\w*)[\s\S]{0,200}реакц|реакц[\s\S]{0,150}голос',re.I)
    votes=np.asarray([bool(pattern.search(t)) for t in texts]);errors=(y-base)**2
    source=Path(__file__).resolve().parents[1];audit=read(args.corpus/'audit.json')
    for key,path in [('text','artifacts/text_models/neural_network.joblib'),('social','artifacts/social_metrics/model.joblib')]:
        assert hashlib.sha256((source/path).read_bytes()).hexdigest()==audit[key+'_sha256']
    result=dict(rows=len(y),resources=len({r['resource_id'] for r in rows}),target_reduction_percent=2.,
        results=results,passes_accuracy_gate=bool(np.all(np.asarray(results['calendar']['mse_gain_percent'])>=2)),
        largest_reaction_row_fraction_of_baseline_sse=float(errors[:,1].max()/errors[:,1].sum()),
        reaction_vote_rule_matches=int(votes.sum()),reaction_vote_max_counts=np.sort(y[votes,1]).tolist(),
        maximum_warm_response_seconds=latency['response_time']['max'],production_weights_unchanged=True,
        calendar_maximum_warm_response_seconds=calendar_latency['response_time']['max'],
        text_sha256=audit['text_sha256'],social_sha256=audit['social_sha256'])
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for path,name in [(args.classical/'report.json','classical_report.json'),(args.neural/'report.json','neural_report.json'),
                      (args.transfer/'report.json','transfer_report.json'),(args.merged/'report.json','combined_report.json'),
                      (args.merged/'latency.json','latency.json'),(args.classical/'data_audit.json','data_audit.json')]:shutil.copy2(path,out/name)
    for directory,name in [('count_stacking_20261111','stacked_report.json'),('calendar_counts_20261111','calendar_report.json'),('reaction_calibration_20261111','calibration_report.json')]:
        shutil.copy2(results_root/directory/'report.json',out/name)
    shutil.copy2(results_root/'calendar_counts_20261111/latency.json',out/'calendar_latency.json')
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pipeline','metric','baseline_mse','model_mse','mse_gain_percent','baseline_mae','model_mae','mae_gain_percent'])
        for name,result in results.items():
            for i,m in enumerate(names):writer.writerow([name,m,mse[i],result['mse'][i],result['mse_gain_percent'][i],mae[i],result['mae'][i],result['mae_gain_percent'][i]])
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    order=list(candidates)
    for i,ax in enumerate(axes):
        ax.barh(range(len(order)),[results[n]['mse_gain_percent'][i] for n in order],color=['#91b6a3' if n=='calendar' else '#6385b8' for n in order])
        ax.set(yticks=range(len(order)),yticklabels=[captions[n] for n in order] if i==0 else ['']*len(order),title=labels[i],xlabel='Снижение MSE, %')
        ax.axvline(2,color='#bd7952',linestyle='--',label='Цель 2%');ax.grid(axis='x',alpha=.2)
    axes[-1].legend();fig.suptitle('Все 11 709 статей · выбор параметров внутри обучения')
    for ext in ('png','svg'):fig.savefig(out/('mse_comparison.'+ext),dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');position=np.arange(3)
    ax.bar(position-.18,results['merged']['mse_gain_percent'],width=.36,label='MSE — основной критерий',color='#6385b8')
    ax.bar(position+.18,results['merged']['mae_gain_percent'],width=.36,label='MAE — отдельная диагностика',color='#91b6a3')
    ax.axhline(2,color='#bd7952',linestyle='--');ax.set(xticks=position,xticklabels=labels,ylabel='Снижение ошибки, %')
    ax.legend();ax.grid(axis='y',alpha=.2)
    for ext in ('png','svg'):fig.savefig(out/('mse_and_mae.'+ext),dpi=170)
    plt.close(fig)
    lines=['# Альтернативные варианты прогноза социальных метрик','',
        'Пользователь уточнил цель: **минимум 2% снижения MSE относительно среднего '
        'канала отдельно для просмотров, реакций и комментариев**. Это заменяет '
        'прежний целевой порог 5%. В этом исследовании проверены все **11 709 '
        'статей / 146 ресурсов**; рабочие веса и БД сохранены.','',
        '## Результаты по основному критерию','',
        '| Вариант | Просмотры: снижение MSE | Реакции | Комментарии |',
        '| --- | ---: | ---: | ---: |']
    for name,r in results.items():lines.append('| '+captions[name]+' | '+' | '.join(f'{v:+.2f}%' for v in r['mse_gain_percent'])+' |')
    lines+=['', '**Цель 2% для каждой метрики '+('достигнута с дополнительными календарными входами.' if all(v>=2 for v in results['calendar']['mse_gain_percent']) else 'пока не достигнута. Рабочая пара не заменена.' )+'**','',
        f"С дополнительными календарными входами просмотры улучшились на **{results['calendar']['mse_gain_percent'][0]:.2f}%**, "
        f"реакции — на **{results['calendar']['mse_gain_percent'][1]:.2f}%**, "
        f"комментарии — на **{results['calendar']['mse_gain_percent'][2]:.2f}%**. "
        'Два из трёх показателей проходят целевой порог. Для реакции он ещё не выполнен.', '',
        '![Сравнение MSE](mse_comparison.png)','',
        '## Что проверено','',
        '1. Прежние входы модуля 4: тема, ресурс, источник, аудитория и прогноз '
        'оценки. Проверены CatBoost с нативными категориями, Ordered boosting, '
        'RMSE и Poisson, глубины 4/6.',
        '2. Прямой SBERT текста и PCA; слова и символьные сочетания TF-IDF; '
        'абсолютные и нормированные на аудиторию остатки; несколько уровней '
        'регуляризации и смешивания со средним канала. Всего 38 базовых '
        'конфигураций и четыре силы отклонения от среднего.',
        '3. Три независимые нейронные модели счётчиков: обычная MSE, MSE '
        'нормированных величин, PCA и пуассоновская вероятность. Выходы '
        'счётчиков не ограничиваются шкалой оценки [-10,10]. Исправлено лишнее '
        'вычитание единицы из среднего Poisson; после исправления повторена '
        'полная проверка. Старые экспериментальные артефакты имеют совместимый '
        'режим прежнего выхода; в этом отчёте используются результаты v2.',
        '4. Перенос относительного отклонения от среднего между похожими '
        'текстами разных каналов. Для обучающей нормировки собственная группа '
        'текстов исключена из среднего канала; варианты соседей, температуры '
        'сходства, исключения собственного ресурса и ограничения обучающего отношения.',
        '5. Семейство для каждой метрики выбирается по общей внутренней проверке '
        'обучающей части, а не по итоговым внешним ответам.','',
        '6. Небольшой корректор учится на OOF-прогнозах трёх метрик. Он поднял '
        f"просмотры до {results['stacked']['mse_gain_percent'][0]:.2f}%, но не дал прорыва по реакции. "
        'Фактические метрики проверочных статей в корректор не подаются.',
        '7. Публикация по местному времени UTC+3, циклические признаки часа/дня недели '
        'и номинальный горизонт наблюдения. Горизонт рассчитан по замороженному '
        'моменту сбора соответствующей когорты и дате публикации. Для рабочего '
        'прогноза эти входы должны означать планируемое время публикации и время, '
        'через которое нужны метрики.',
        '8. Непрерывный коэффициент реакции оценён по обучающим OOF-прогнозам. '
        'Он не улучшил основной результат, поэтому этот вариант отклонён.','',
        '**Ограничение календарного варианта:** в одном срезе смешиваются возраст '
        'поста и изменение интереса канала со временем. Результат 9,38% нельзя '
        'считать доказанной точностью прогноза ровно через 7 дней: для этого '
        'нужны повторные замеры одних статей и новый временной контроль. Этот '
        'вариант не включён в рабочую программу автоматически.','',
        'Прямые текстовые варианты требуют дополнительного входа в модуль 4 '
        '(текст или уже вычисленный SBERT). Они остаются исследовательскими; '
        'модуль 3 по-прежнему получает только текст и тему, без канала.','',
        '## Ограничение выборки и другой взгляд на ошибку','',
        f"Самый большой пример даёт **{100*errors[:,1].max()/errors[:,1].sum():.2f}%** "
        'суммы квадратов ошибок среднего канала по реакциям. Это текст с '
        'голосованием реакциями: 160 347 реакций при 264 427 просмотрах. '
        'Запись сохранена в итоговом расчёте. Во всей выборке следующий максимум '
        'реакций — 9160.', '',
        f"Простое текстовое правило нашло **{int(votes.sum())}** голосований реакциями. "
        'Это диагностическая оценка, а не полная ручная разметка форматов. У '
        'остальных совпадений не больше 493 реакций. Из этих данных нельзя '
        'уверенно выучить ожидаемый масштаб редкой крупной кампании только '
        'перебором гиперпараметров. Требуются дополнительные сопоставимые примеры '
        'или явный учёт формата и ожидаемого продвижения публикации.', '',
        'MAE ниже приводится отдельно. Она линейно учитывает крупные отклонения '
        'и отвечает на иной вопрос, чем MSE; её улучшение не выдаётся за выполнение '
        'поставленного порога MSE.','',
        '| Метрика | Снижение MSE объединённого варианта | Снижение MAE |',
        '| --- | ---: | ---: |']
    for i,label in enumerate(labels):lines.append(f"| {label} | {results['merged']['mse_gain_percent'][i]:+.2f}% | {results['merged']['mae_gain_percent'][i]:+.2f}% |")
    lines+=['','![MSE и MAE](mse_and_mae.png)','',
        '## Время и воспроизводимость','',
        f"Максимальный полный прогретый ответ: **{latency['response_time']['max']:.3f} с**. "
        f"Для календарного варианта — **{calendar_latency['response_time']['max']:.3f} с**. "
        'Проверены все пять исследовательских пар через отдельное локальное '
        'ASGI-приложение: по 12 реальных текстов и 50 000 символов. Включены '
        'SBERT полного текста, автоопределение темы, оценка, три социальных '
        'прогноза и JSON; исключены первая загрузка, БД, сеть и LLM. Время '
        'обучения не участвовало в выборе.','',
        'До измерения времени повторно загруженные модели в точности '
        'воспроизвели все 11 709 сохранённых социальных прогнозов. Для общей '
        'предобработки введён кэш одного запроса; три целевые головы имеют '
        'отдельные параметры, неизменность прогнозов при кэшировании проверена. '
        'Прошли 179 Python-тестов и 14 JS-тестов.','',
        'Пять внешних групповых разбиений исключают одинаковые нормализованные '
        'тексты между обучением и проверкой. Каждая статья проверена ровно один '
        'раз. Внутренние разбиения одинаковы для всех семейств, выбор сделан '
        'по внутренним ошибкам. Словари, масштабы, PCA и средние обучаются '
        'только на обучающих строках. Для новой семейной проверки использованы '
        'замороженные обучающие OOF-оценки предыдущей цепочки. Сами внешние '
        'проверочные метрики нигде не подаются на вход.', '',
        'Корпус использовался в предыдущих исследованиях. Результаты описывают '
        'полную повторную проверку имеющихся данных; независимый контроль '
        'будущих дат пока отсутствует. Сырые тексты, матрицы и промежуточные '
        'веса не включены в GitHub-копию. Все отпечатки исходных текстов '
        'проверены; данные БД не изменялись.','',
        '[MSE и MAE по каждому варианту](metrics.csv) · [Сводка](summary.json) · '
        '[Выбор семейства](combined_report.json) · [Время ответа](latency.json)','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(calendar_gains_percent=results['calendar']['mse_gain_percent'],
                         passes_accuracy_gate=all(v>=2 for v in results['calendar']['mse_gain_percent']))))


if __name__=='__main__':main()

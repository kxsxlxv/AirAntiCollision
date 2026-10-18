"""Generate a small offline Russian video gallery from real saved outputs."""
from html import escape
from pathlib import Path

cards=[
('M2_final','Разные решения для разных целей','14 с: зелёные признаки прохода у цели №2, жёлтая неопределённость у других. Около 27 с растущая цель получает красное предупреждение. Класс детектора часто ошибочен.','frame_14s.png'),
('M1_final','Предупреждение и неопределённость вместе','Около 26 с приближающаяся цель красная; вторая остаётся жёлтой. Мелкие цели часто пропускаются.','frame_26s.png'),
('M6_failure','Ошибка на размеченной категории other','На 6.16–7.16 с этого отрывка (96.16–97.16 с источника) есть предупреждение по другому самолёту. Это 26 кадров / 1.04 с. Отрисовка записанных решений полного прогона, без повторной нейросетевой оценки.','preview.png'),
('S2_final','Трудный случай: нет свежего трека','На 10 с цель ещё не отслеживается. Позже появляется предупреждение. Пустой экран не означает отсутствие угрозы.','frame_10s.png'),
('S2_zoom_final','Дополнительные нейросетевые наблюдения','Тот же RGB: предложения небольших фрагментов проверяются YOLO. Сравните раннюю часть с S2 выше. Больше наблюдений ещё не означает более достоверную геометрию.','frame_10s.png'),
('formation_final','Воздушная FPV-съёмка, подвижная камера','Исходное 1920×1080: наклон и движение камеры. Все оценки неопределённы; фоновое выравнивание не разделяет перенос и поворот камеры. Меток столкновения нет.','preview.png'),
('helicopter_final','Реальный беспилотный вертолёт','Меняется ракурс; все оценки неопределённы. Детекция и сопровождение не устанавливают геометрию сближения. Меток столкновения нет.','preview.png')]
body=[]
for folder,title,description,preview in cards:
    path=Path('demo')/folder/'annotated.mp4'
    if not path.exists() or not path.stat().st_size:raise FileNotFoundError(path)
    body.append(f'<article><h2>{escape(title)}</h2><video controls preload="none" poster="{folder}/{preview}"><source src="{folder}/annotated.mp4" type="video/mp4"></video><p>{escape(description)}</p><a href="{folder}/annotated.mp4">Открыть MP4</a></article>')
html='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RGB: оценка воздушных встреч</title>
<style>body{margin:0;background:#0c131e;color:#dce6f1;font:17px/1.55 system-ui,sans-serif}main{max-width:1350px;margin:auto;padding:28px}h1{font-size:36px;line-height:1.2;margin:12px 0}h2{font-size:23px;margin:0 0 14px}a{color:#8bbfff}p{margin:12px 0}.sub{color:#a8b6c8}.note{border-left:4px solid #f6c353;padding:12px 18px;background:#192333}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:24px 0}article{background:#142030;padding:20px;border-radius:12px}video,img{display:block;width:100%;height:auto;border-radius:5px;background:#000}.chips{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}.red{color:#ff5e67}.green{color:#43d6aa}.yellow{color:#f6c353}footer{font-size:15px;color:#a8b6c8;margin:25px 0}@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:16px}h1{font-size:29px}}</style>
<main><p class="sub">Исследовательский офлайн-прототип • реальные RGB-видео • 18 октября 2026</p><h1>Что видно о воздушной встрече — и чего камера не знает</h1>
<p>Система сравнивает поперечный дрейф и рост видимого размера каждой цели. Причины справа — реальные вычисленные показатели. Цвет не является вероятностью столкновения.</p>
<div class="chips"><span class="red">● Возможная угроза</span><span class="green">● Признаки прохода мимо</span><span class="yellow">● Недостаточно данных</span></div>
<div class="note"><strong>Направление полёта не рекомендуется.</strong> Внизу роликов сравниваются условные изменения дрейфа при одинаковой глубине целей. Дальности, динамика аппарата, прочие препятствия и связь осей камеры с аппаратом неизвестны.</div>
<div class="grid">'''+''.join(body)+'''</div>
<h2>Проверка шире удачного фрагмента</h2><img src="../reports/results_cadence_full.png" alt="Фактические результаты 15 полных последовательностей QUT">
<p>Основная оценка: 15 полных QUT, 37.02 минуты, 11 113 обработанных кадров с шагом 5. Все 15 номинальных подходов имеют предупреждение; у 13 событий other предупреждений нет. Полный покадровый прогон всех 15 источников выявил 40 кадров с предупреждением по категории other: 26 кадров / 1.04 с в M6 и 14 кадров / 0.56 с в M7. Два из 13 событий имеют предупреждения; только M6 достигает критерия 0.8 с. Нельзя переносить нулевой результат между частотами обработки.</p>
<p>QUT снят неподвижной наземной камерой. Категория approach не доказывает физического столкновения, а other не задаёт истинный CPA. M3 имеет противоречивую разметку дальности. Данные уже изучались при разработке; независимого теста нет. Пропуски и неопределённость включены в таблицу.</p>
<p>H.264/yuv420p, исходные интервалы показа сохранены. Текущий M2 на RTX 4060: 20.08 обработанного кадра/с с запуском Python, 23.81 кадра/с в основном цикле; FPV 1920×1080: 10.66 и 12.26 кадра/с соответственно. Это офлайн-производительность, не гарантия потоковой задержки.</p>
<p><a href="../reports/warning_timeline.png">Когда появляются предупреждения: все 15 подходов</a></p>
<p><a href="README.md">Краткий русский путеводитель</a> • <a href="../RESULTS.md">Результаты и ограничения (English)</a> • <a href="../README.md">Запуск и воспроизведение</a></p>
<footer>Фиксированные параметры, реальный детектор, сохранённые решения. Ground truth не поступает в работающий алгоритм. Исходные неудачные версии сохранены отдельно. Прототип не предназначен для управления полётом.</footer></main></html>'''
Path('demo/index.html').write_text(html,encoding='utf-8')
print('demo/index.html')

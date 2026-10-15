# Риски и меры

Пять рисков из брифа М.Видео (слайд 15) — с мерами, местом в коде и проверкой. Статус: **есть** — код в репозитории
и тест; **план** — будет сделано. Цифры — holdout, отчёт v2: весь отложенный сплит с эталоном, 23 391 карточка
(`submission/REPORT.md`, [EVALUATION.md](EVALUATION.md) §9), если не помечены «dev» (сплит настройки, §8).
Продуктовый контекст — [PRD.md](PRD.md), внедрение — [DEPLOYMENT.md](DEPLOYMENT.md).

| Риск | Последствие | Главная мера |
|---|---|---|
| 1. Галлюцинации LLM | ложный атрибут ломает фильтр и расходится по поиску | цитата + проверка заземления + справочник + калиброванный порог; значение только от LLM — всегда на проверку |
| 2. Стоимость на масштабе | LLM на каждую карточку — дорого | LLM только для хвоста и только об упомянутых атрибутах, кэш, лимиты, on-prem модель |
| 3. Оценка не покрывает хвост | метрика красивая, качество плохое | метрики по атрибутам и категориям, head / torso / tail, извлекаемый recall, веса спроса |
| 4. Дрейф и новые категории | качество падает со временем | классификатор категории, таксономия без кода, LLM со статусом «на проверку», петля правок, мониторинг `attrx monitor` |
| 5. Несправедливость к продавцам | мелкие продавцы и редкие категории хуже обработаны | извлечение из текста, справочник брендов всего каталога, метрики по сегментам |

## 1. Галлюцинации LLM

| Мера | Где | Проверка | Статус |
|---|---|---|---|
| LLM вызывается только для пустых, слабых (уверенность < `LLM_ROUTE_THRESHOLD`) и конфликтных атрибутов и только если атрибут упомянут в тексте (префильтр) | `src/attrx/pipeline.py`, `src/attrx/extractors/llm.py` | `tests/test_llm.py::test_llm_only_for_unresolved_attrs`, `::test_llm_not_called_when_text_has_nothing_to_extract` | есть |
| Строгий JSON по схеме атрибутов категории, temperature 0 | `src/attrx/extractors/llm.py`, `src/attrx/llm/client.py` | `tests/test_llm_client_mock.py::test_strict_json_schema_is_sent_with_temperature_zero` | есть |
| Допустимые значения справочника — в промпте; значение вне справочника отбрасывается. В JSON Schema список не зашиваем: по замеру это провоцирует вынужденный выбор (`LLM_SCHEMA_ENUM=false`, [LLM.md](LLM.md) §2) | `src/attrx/extractors/llm.py`, `src/attrx/normalize.py` | `tests/test_llm.py::test_out_of_dictionary_llm_value_is_reported_as_dictionary_candidate` | есть |
| Цитата обязательна и выравнивается с текстом карточки: числа — точно, слова — с допуском на окончания и знаки (`ALIGN_MIN` 0,85); значение должно следовать из найденного фрагмента, иначе отброшено с предупреждением. Смещения цитаты — `evidence_span` | `src/attrx/verify.py` (`align`, `ground`) | `tests/test_grounding.py::test_align_rejects_fabricated_quote_and_other_numbers`, `::test_value_must_follow_from_aligned_span`, `tests/test_llm.py::test_hallucinated_value_rejected` | есть |
| **Значение, которое предложила только LLM, никогда не принимается автоматически** — статус `review` при любой уверенности. Подтверждая значение продавца, правил или NER, LLM повышает общую уверенность (noisy-OR) | `pipeline.UNVERIFIED_ONLY` → `Extractor._finalize`; то же правило при выборе порога (`src/attrx/eval/calibrate.py`) | `tests/test_grounding.py::test_llm_only_value_goes_to_review`, `::test_llm_corroborating_rules_can_be_accepted` | есть |
| Уверенность откалибрована на dev (изотоническая регрессия по группам «категория / атрибут / источники»); порог 0,70 выбран под precision ≥ 0,95. Весь holdout: точность принятых 0,954 (95% ДИ от 0,952), принято 94% найденного, ECE 0,019. **Слабое место — слой у порога:** принятые значения с уверенностью 0,70–0,80 верны лишь в 57% случаев (2 008 значений — 4% принятого, но 38% ошибок среди принятого; 91% — цвет и модель хвоста); на dev то же. Кандидат исправления найден только на dev — калибровка на всех карточках dev, порог 0,82: точность принятых 0,982 при доле автопринятия 85,5%. Не применён: выбор после просмотра holdout нарушил бы протокол | `src/attrx/confidence.py`, `src/attrx/eval/calibrate.py` → `config/calibration.json` (`attrx calibrate`) | `submission/calibration.json` (`accepted_band`), `threshold.json`; [EVALUATION.md](EVALUATION.md) §9.5 | есть; кандидат — решение по тени |
| Значения нет в тексте → статус `missing`, не заполняем | `src/attrx/pipeline.py` | `tests/test_pipeline.py::test_no_hallucination_when_absent` | есть |
| Generative UI не показывает значений вне результата; невалидная спецификация экрана отклоняется | `src/attrx/genui/`, [GENUI.md](GENUI.md) | `tests/test_llm.py::test_genui_rejects_invalid_spec` | есть |
| Точность LLM-значений и доля отброшенных измерены. Holdout, подвыборка 662 карточки (qwen3:4b-instruct): LLM спросили для 31% карточек (205 вызовов), 61 ответ без опоры в тексте отброшен; из 71 значения одной LLM верны 38% (13 из 34 с эталоном) — поэтому они идут на проверку. Если бы их принимали автоматически: P 0,962 → 0,958 ради +0,003 полноты | `pipeline.UNVERIFIED_ONLY`, `src/attrx/verify.py` | `submission/REPORT.md` §4, `compare.json` | есть |

## 2. Стоимость на масштабе

| Мера | Где | Проверка | Статус |
|---|---|---|---|
| Гибридный роутинг: продавец и правила → NER → LLM только для слабых атрибутов | `src/attrx/pipeline.py` | `tests/test_llm.py::test_llm_only_for_unresolved_attrs` | есть |
| Префильтр: LLM не спрашивают об атрибутах, о которых в тексте нет ни слова (для чисел — ни цифры рядом): на 20 реальных карточках −56% вызовов при том же результате | `LLM_PREFILTER`, `src/attrx/extractors/llm.py` | `tests/test_llm.py::test_llm_not_called_when_text_has_nothing_to_extract`; [LLM.md](LLM.md) §1 | есть |
| Кэш ответов LLM (SQLite): повторная карточка не оплачивается | `src/attrx/llm/cache.py`, `LLM_CACHE` | `tests/test_llm_client_mock.py::test_stats_tokens_cache_and_latency` | есть |
| Лимиты вызовов на процесс и на сутки | `LLM_MAX_CALLS`, `LLM_DAILY_MAX_CALLS` | `tests/test_llm_client_mock.py::test_daily_and_process_limits` | есть |
| Короткий промпт: в LLM идут только релевантные предложения (`LLM_MAX_INPUT_CHARS=3500`), ответ ограничен (`LLM_MAX_TOKENS`) | `src/attrx/extractors/llm.py` (`fit_text`) | `tests/test_llm_prompt.py::test_fit_text_keeps_relevant_sentences_verbatim` | есть |
| Замер на holdout: LLM для 31% карточек, 969 + 52 токена и 3,3 с на вызов (p95 5,0 с). За 1000 карточек гибрид — 0,83 ₽ на своей GPU, 20,6 ₽ GigaChat Lite, 63,2 ₽ YandexGPT Lite против 5,6 / 84,6 / 260,2 ₽ у «LLM на каждую карточку» — в 4–7 раз дешевле; режим в сдаче без LLM — < 0,01 ₽. Цель «LLM ≤ 20% карточек» пока не выполнена | `src/attrx/llm/cost.py`, `config/llm_prices.yaml`, `attrx llm-bench` | `submission/cost.json`, [LLM.md](LLM.md) §6; `tests/test_llm_prompt.py::test_cost_per_1000_cards_every_card_vs_hybrid` | есть |
| On-prem модель без оплаты за токен | `docker-compose.yml`, профиль `llm` (Ollama) | `docker compose --profile llm up` | есть |
| Дистилляция: подтверждённые ответы LLM → обучающие примеры NER, доля вызовов LLM падает | `src/attrx/extractors/ner_distant.py` | — | план |

Формула: стоимость гибрида ≈ p_LLM × стоимость «LLM на каждую карточку», где p_LLM — доля карточек, попавших в хвост.

## 3. Оценка не покрывает хвост

| Мера | Где | Проверка | Статус |
|---|---|---|---|
| P / R / F1 по каждому атрибуту и ячейке «категория × атрибут», macro-F1; общую accuracy не используем | `src/attrx/eval/metrics.py` | `tests/test_report.py` | есть |
| Эталон из каталога (241 355 карточек с офферами); holdout фиксирован по md5(sku_id), правила и калибровка на нём не настраиваются. Основной набор — весь holdout с эталоном (23 391 карточка, 94% — хвост каталога), а не только «удобные» категории | `src/attrx/data/`, `src/attrx/eval/gold.py` | `tests/test_eval_gold.py` | есть |
| Извлекаемый recall — только по значениям, которые есть в тексте (их 83%); заполнения, которых нет в неполном эталоне, не считаются ошибкой (`new_fills`). Holdout: 0,927; из 10 956 пропусков 8 401 (77%) — значения нет в тексте | `src/attrx/eval/metrics.py`, `src/attrx/eval/gold.py` | `tests/test_eval_gold.py::test_extractable_numbers_in_text`, `::test_incomplete_gold_new_fills_not_false_positives` | есть |
| Разрез по размеру категории: head (≥ 1000 офферов), torso (≥ 100), tail. Holdout, F1: 0,870 / 0,844 / 0,818 (в базовой линии на dev было 0,616 / 0,257 / 0,227) | `SIZE_BANDS` в `src/attrx/eval/gold.py` | `submission/segments.json` | есть |
| Срез «новые названия» против запоминания: название карточки дословно есть в train у 28% holdout; на остальных F1 0,841 (16 803 карточки) против 0,912 у повторов, полнота по извлекаемому 0,922 против 0,941 | `src/attrx/eval/final.py` (`seen_title_ids`) | `submission/REPORT.md` §5 | есть |
| Веса спроса из 31 млн поисковых кликов: 255 категорий, вес атрибута = доля кликов категории, где он назван | `attrx traffic` (`src/attrx/data/traffic.py`) → `config/traffic_weights.yaml`, `submission/traffic.json` | `tests/test_traffic.py`, `tests/test_eval_gold.py::test_traffic_weights_from_clicks_and_yaml` | есть |
| Ручной аудит выборки эталона на редких категориях: LLM предразмечает, человек проверяет | — | — | план |

## 4. Дрейф и новые категории

| Мера | Где | Проверка | Статус |
|---|---|---|---|
| Категория по тексту для 75% карточек без оффера и кликов: TF-IDF + линейная модель «раздел → лист каталога». Holdout: точность категории таксономии 0,9998 против 0,921 у ключевых слов; ниже порога 0,9 — все значения карточки на проверку | `src/attrx/category.py` (`attrx train-category`), `Taxonomy.classify`; [CATEGORY.md](CATEGORY.md) | `tests/test_category.py` | есть |
| Категории, атрибуты и словари — YAML; новая категория без изменения кода; лист вне таксономии — `generic` | `config/taxonomy.yaml`, `config/dictionaries/` | `tests/test_eval_gold.py::test_taxonomy_validation`, `tests/test_pipeline.py::test_unknown_category_falls_back_to_generic` | есть |
| LLM извлекает атрибуты новой категории zero-shot по схеме; без второго источника значения уходят на проверку | `src/attrx/extractors/llm.py`, `pipeline.UNVERIFIED_ONLY` | `tests/test_grounding.py::test_llm_only_value_goes_to_review` | есть |
| Петля: правки модератора → `feedback.jsonl` → `attrx learn` → синонимы в `learned.yaml` | `src/attrx/feedback.py`, `POST /api/feedback` | `attrx learn` | есть |
| Словари из данных, а не руками: 4 822 бренда каталога, соответствие значений справочника значениям каталога (`enum_map.json`), стиль модели по бренду — пересобираются одной командой | `attrx build-dicts` (`src/attrx/dictgen.py`) | [EVALUATION.md](EVALUATION.md) §8.1–8.2 | есть |
| Мониторинг дрейфа: `attrx monitor` сравнивает поток с эталонным прогоном — доля `review` и `missing`, уверенность, конфликты, неуверенная категория по категориям; PSI и JS распределения категорий; новые листья каталога; доля LLM и отброшенных значений → алерт с действием (`--fail-on-alert` для cron) | `src/attrx/monitor.py`, [MONITORING.md](MONITORING.md) | `tests/test_monitor.py` | есть |
| Выборочный аудит precision принятого в проде (в том числе слоя уверенности 0,70–0,80) и откат категории в тень | журнал решений модератора (`feedback`) | — | план (пилот) |

## 5. Несправедливость к продавцам

| Мера | Где | Проверка | Статус |
|---|---|---|---|
| Извлекаем из свободного текста; поля продавца — лишь один из источников (блок «Характеристики» есть у 6,6% карточек с эталоном) | `src/attrx/pipeline.py`, `src/attrx/extractors/seller.py` | — | есть |
| Справочник брендов всего каталога одинаково покрывает всех продавцов: абляция — без него F1 −0,175, у маркетплейса −0,26 | `config/dictionaries/brands.yaml` (`attrx build-dicts`) | [EVALUATION.md](EVALUATION.md) §8.2 | есть |
| Метрики по сегментам: тип продавца, длина описания, размер категории. Весь holdout, F1: М.Видео 0,873, маркетплейс 0,853 — разрыв 2,0 п.п. (в базовой линии на dev 0,654 и 0,447) | `src/attrx/eval/report.py`, `src/attrx/eval/gold.py` | `tests/test_report.py::test_segments_and_errors`; `submission/segments.json` | есть |
| Конфликт продавца и текста → на проверку | `src/attrx/confidence.py` (`cross_conflict`), `pipeline._finalize` | `tests/test_pipeline.py::test_seller_conflict_goes_to_review` | есть |
| Цели: разрыв F1 по продавцам ≤ 5 п.п., между head и tail ≤ 10 п.п. — выполнены (2,0 и 5,2 п.п.); tail — самый слабый сегмент (точность 0,933 против 0,962 у head, полнота по извлекаемому 0,893 против 0,942) — следующая итерация словарей; отстающему сегменту — свой порог маршрутизации в хвост | `submission/segments.json` | — | есть (цели); tail — план |

## Риски сверх кейса

| Риск | Мера | Статус |
|---|---|---|
| LLM-провайдер недоступен | ошибка перехватывается, карточка обрабатывается без LLM; предохранитель не ждёт упавший сервер (`tests/test_llm_client_mock.py::test_connection_error_opens_circuit`, `tests/test_llm_prompt.py::test_extract_reports_unavailable_llm_as_warning`); `LLM_ENABLED=false` — штатный режим без LLM; провайдер меняется переменной | есть |
| Данные уходят из контура | on-prem LLM (vLLM / Ollama), без LLM внешних вызовов нет; в карточках нет персональных данных | есть |
| Подгонка под holdout | правила, словари, NER и калибровка — на train / dev. Holdout прогнан дважды: v2 — после исправлений, выбранных только на dev и на демо-сэмпле вне eval-набора; на тех же 2 416 карточках F1 v1 → v2 0,843 → 0,842; слабость калибровки, найденная на holdout, не исправлялась; обе версии раскрыты ([EVALUATION.md](EVALUATION.md) §9, §9.9). Весь dev → весь holdout: F1 0,859 → 0,858 | есть |
| Калибровка обучена на продавце, правилах и NER; сочетания с LLM получают кривую типа атрибута | значение одной LLM всё равно идёт на проверку; калибровку с LLM переобучить после замера на модели организаторов | есть (NER) / план (LLM) |
| Эталон неполон или ошибается | извлекаемый recall и `new_fills`; ручной аудит выборки | есть / план |
| Модель LLM слабая (локальная 4B): одна LLM верна в 38% случаев | LLM — подсказки модератору, не автоматический источник; следующий шаг — тот же замер с моделью организаторов на ВМ и 7–8B в vLLM | есть / план |
| Уверенность у порога завышена: слой 0,70–0,80 верен на 57%, 91% — цвет и модель хвоста (калибровку учили на стратифицированном dev, где хвоста мало) | аудит слоя в тени; кандидат с dev — калибровка на всех карточках dev (порог 0,82: точность принятых 0,982 при доле автопринятия 85,5%, на 8–9% больше работы модератору); включаем по данным тени и `attrx monitor`, а не по holdout | найдено и раскрыто / кандидат — план |

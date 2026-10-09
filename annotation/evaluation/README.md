# Оценка captions

CIDEr, BLEU-4 и ROUGE-L — `pycocoevalcap==1.2`, реализация COCO caption
[авторов](https://github.com/tylin/coco-caption). PTBTokenizer использует Java
Stanford CoreNLP из пакета; каталог временных файлов в образе доступен UID пользователя.
Метрики вычисляются из неизменных `raw_caption`, токенизация создаёт отдельное представление.
CIDEr сохраняется в исходной шкале, без умножения на 100.

CHAIR: [исходная реализация авторов](https://github.com/LisaAnne/Hallucination/tree/6e4d33c4bedf6dfcd24c61bf527f40714717be47),
`utils/chair.py` и `data/synonyms.txt`. Авторы указывают BSD 2-Clause в README.
Перенесены только конструктор словаря и caption_to_words; правила пар слов и
исключения сохранены. Старый загрузчик COCO-2014 заменён объединением categories
из `instances_val2017.json` и слов эталонных captions каждого оцениваемого image_id.
CHAIRs — доля описаний хотя бы с одним неподтверждённым объектом из словаря;
CHAIRi — доля таких упоминаний среди всех распознанных упоминаний объектов.
Повторные упоминания считаются повторно; GT включает crowd и non-crowd.

Singularize — только соответствующие функции/таблицы
[Pattern](https://github.com/clips/pattern/blob/af754685cca3713db0abc4f020f2e94467c19d85/pattern/text/en/inflect.py)
с сохранённым лицензионным уведомлением (MIT). Не заменён WordNet-лемматизатором.
Токенизация NLTK 3.9.2 с офлайн punkt_tab. Версии и исходные SHA — sources.json.
Проверяются plural people/bicycles, baby bird, passenger jet, traffic light,
wine glass, hot dog и исключение toilet seat.

`object_recall_micro` — дополнительная полнота относительно объединённых
объектов GT; не стандартная метрика CHAIR и не экспертная полнота.
Описания действий, качеств и вне-COCO объектов CHAIR не проверяет.
Экспертные оценки этим показателем не заменяются. SPICE не активирован:
его evaluator и ресурсы в этом протоколе не валидированы.

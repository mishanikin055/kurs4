# Полная таблица сравнения детекторов

Все доступные показатели из сохранённых test-v2: COCO test, 4500 изображений на повтор, три повтора, FP32, batch 1. По столбцам — модели, по строкам — показатели. Исходные экспериментальные файлы не изменены. Полная точность чисел — в detection_full.csv; CSV записан в UTF-8 с BOM для Excel.

| Показатель | YOLO26s | RT-DETRv2 R18 | RF-DETR Small | Faster R-CNN ResNet50 FPN V2 |
| --- | --- | --- | --- | --- |
| status | complete | complete | complete | complete |
| completed_repeats | 3 | 3 | 3 | 3 |
| n_images | 4500 | 4500 | 4500 | 4500 |
| mAP | 0.475269 | 0.480841 | 0.528356 | 0.47044 |
| AP50 | 0.640917 | 0.651094 | 0.720145 | 0.681364 |
| AP75 | 0.514008 | 0.520507 | 0.568858 | 0.513322 |
| AP_small | 0.297947 | 0.300668 | 0.323945 | 0.312032 |
| AP_medium | 0.517686 | 0.508812 | 0.579995 | 0.508019 |
| AP_large | 0.643883 | 0.642342 | 0.730622 | 0.606489 |
| AR100 | 0.642714 | 0.700713 | 0.703659 | 0.64776 |
| precision | 0.889206 | 0.702671 | 0.887656 | 0.602194 |
| recall | 0.477934 | 0.63834 | 0.570007 | 0.731765 |
| F1 | 0.621709 | 0.668963 | 0.694221 | 0.660687 |
| mAP_ci95_low | 0.468615 | 0.473682 | 0.521961 | 0.464199 |
| mAP_ci95_high | 0.488354 | 0.494171 | 0.541833 | 0.483403 |
| p50_ms | 17.4697 | 27.7688 | 26.1391 | 172.985 |
| p95_ms | 23.9026 | 41.9077 | 38.2723 | 190.701 |
| images_per_second | 55.0926 | 33.5486 | 35.9539 | 5.78477 |
| cold_load_ms_mean | 410.088 | 2432.25 | 5205.31 | 503.37 |
| VRAM_allocated_MiB | 122.803 | 183.971 | 172.992 | 1131.49 |
| VRAM_reserved_MiB | 184 | 262 | 204 | 2068 |
| RAM_peak_MiB | 1552.43 | 1719.23 | 1893.14 | 1626.83 |
| parameters | 9496140 | 20174608 | 32111170 | 43712278 |
| weight_MiB | 19.4766 | 77.1567 | 368.162 | 167.104 |
| device | cuda:0 | cuda:0 | cuda:0 | cuda:0 |
| precision_mode | fp32 | fp32 | fp32 | fp32 |
| dataset_split | test | test | test | test |
| model_revision | assets/v8.4.0 | 5650961749fa93567c0d46fc7f43ea4f9e914107 | gcs-generation-1753220474114031 | COCO_V1-dd69338a |
| run_ids | yolo26s_repeat1;yolo26s_repeat2;yolo26s_repeat3 | rtdetr_v2_r18vd_repeat1;rtdetr_v2_r18vd_repeat2;rtdetr_v2_r18vd_repeat3 | rf_detr_small_repeat1;rf_detr_small_repeat2;rf_detr_small_repeat3 | fasterrcnn_resnet50_fpn_v2_repeat1;fasterrcnn_resnet50_fpn_v2_repeat2;fasterrcnn_resnet50_fpn_v2_repeat3 |
| weight_sha256 | 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b | d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115 | d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33 | dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf |
| git_commit | — | — | — | — |
| git_dirty | True | True | True | True |
| container_image_id | sha256:72b472f15ea77dc0f8526857fd50a6ada2a53de827e822cf8122a13898e0bc42 | sha256:72b472f15ea77dc0f8526857fd50a6ada2a53de827e822cf8122a13898e0bc42 | sha256:72b472f15ea77dc0f8526857fd50a6ada2a53de827e822cf8122a13898e0bc42 | sha256:72b472f15ea77dc0f8526857fd50a6ada2a53de827e822cf8122a13898e0bc42 |
| source_hash | 32cd65ac23fbbae6dbf98874c1a3d71116675de0f478e011da7d98f5a77c8c86 | 32cd65ac23fbbae6dbf98874c1a3d71116675de0f478e011da7d98f5a77c8c86 | 5e74cd9c73e0ca0782befd074a9161f3cfab84406c2c56e72edf61a2d4182414 | 32cd65ac23fbbae6dbf98874c1a3d71116675de0f478e011da7d98f5a77c8c86 |
| failures | — | — | — | — |
| reused_from | reports/detection/test-v1/yolo26s_repeat1;reports/detection/test-v1/yolo26s_repeat2;reports/detection/test-v1/yolo26s_repeat3 | reports/detection/test-v1/rtdetr_v2_r18vd_repeat1;reports/detection/test-v1/rtdetr_v2_r18vd_repeat2;reports/detection/test-v1/rtdetr_v2_r18vd_repeat3 | — | reports/detection/test-v1/fasterrcnn_resnet50_fpn_v2_repeat1;reports/detection/test-v1/fasterrcnn_resnet50_fpn_v2_repeat2;reports/detection/test-v1/fasterrcnn_resnet50_fpn_v2_repeat3 |
| excluded_coco_gap_predictions | 0 | 0 | 60 | 0 |
| AR1 | 0.359545 | 0.370745 | 0.393309 | 0.370718 |
| AR10 | 0.591214 | 0.631222 | 0.646275 | 0.604159 |
| AR_small | 0.451391 | 0.52379 | 0.492675 | 0.502817 |
| AR_medium | 0.699693 | 0.736826 | 0.770736 | 0.683943 |
| AR_large | 0.795611 | 0.858016 | 0.902691 | 0.782777 |
| TP | 15562 | 20785 | 18560 | 23827 |
| FP | 1939 | 8795 | 2349 | 15740 |
| FN | 16999 | 11776 | 14001 | 8734 |
| operating_threshold | 0.5 | 0.5 | 0.5 | 0.5 |
| operating_iou | 0.5 | 0.5 | 0.5 | 0.5 |
| operating_max_detections | 100 | 100 | 100 | 100 |
| protocol_version | detection-v1 | detection-v1 | detection-v1 | detection-v1 |
| batch_size | 1 | 1 | 1 | 1 |
| seed | 42 | 42 | 42 | 42 |
| warmup | 20 | 20 | 20 | 20 |
| score_floor | 0.001 | 0.001 | 0.001 | 0.001 |
| max_detections | 300 | 300 | 300 | 300 |
| cpu_threads | 4 | 4 | 4 | 4 |
| bootstrap_samples | 1000 | 1000 | 1000 | 1000 |
| preprocess_mean_ms | 1.95285 | 3.59554 | 3.13342 | 6.1945 |
| forward_mean_ms | 12.9699 | 22.1419 | 20.9422 | 164.279 |
| postprocess_mean_ms | 3.22856 | 4.07002 | 3.73774 | 2.3938 |
| read_mean_ms | 2.33751 | 2.30585 | 2.33978 | 2.33545 |
| write_mean_ms | 3.46671 | 8.45713 | 5.46551 | 5.75811 |
| offline_inference | yes: network_mode=none | yes: network_mode=none | yes: network_mode=none | yes: network_mode=none |
| parameters_note | post-hoc metadata, CPU, no inference, same pinned image and weights | recorded runtime network | recorded runtime network | recorded runtime network |

## Как читать показатели

- mAP, AP50/AP75, AP_small/medium/large, AR1/10/100 и AR_small/medium/large, precision/recall/F1 и TP/FP/FN относятся к первому повтору. AP/AR представлены в диапазоне 0–1. Precision/recall/F1 и ошибки рассчитаны при score ≥ 0,5, IoU ≥ 0,5, max detections 100; настройки также перечислены в таблице.
- p50_ms/p95_ms — объединённые измерения трёх повторов: preprocess + forward + postprocess. images_per_second = 1000 / среднее этого времени; чтение и запись исключены. Средние времена отдельных стадий, чтения и записи — взвешенные по числу изображений средние сохранённых повторов. cold_load_ms_mean — средняя холодная загрузка.
- VRAM_allocated_MiB — пик памяти тензоров PyTorch, VRAM_reserved_MiB — пик его резерва; значения не складываются и не включают полную память CUDA-контекста/драйвера. RAM_peak_MiB — пик RSS процесса, включая загрузку. Пики берутся как максимум по повторам. weight_MiB — размер файлов весов на диске.
- parameters — число параметров исполняемой сети. У YOLO исходная оболочка сохранила некорректный ноль; уточнение внутренней сети выполняется отдельным офлайн CPU-подсчётом, с проверкой SHA весов и ID исходного контейнера. Если уточнения нет, число отсутствует, а не принимается равным нулю.
- Ошибки по всем 80 классам (AP, TP/FP/FN, precision/recall/F1) — detection_per_class.csv. Парные доверительные интервалы — detection_paired.csv; подробности bootstrap — detection_analysis.md и исходный bootstrap.json.

## Покрытие плана и ограничения

Основные метрики сравнения детекторов из раздела 5 плана включены. Сбои текущей версии отражает failures; три исходных сбоя RF-DETR версии test-v1 сохранены в recovery.json и исходных логах. Девять проходов перенесены без изменения; происхождение показывает reused_from. Офлайн-инференс выполнен в контейнерах без сети. Полные штатные настройки препроцессинга, разрешения и нормализации находятся в native_config.json каждого запуска.

Полное потребление GPU по nvidia-smi в этих прогонах не измерялось. Сравнение на искажённых изображениях и отдельном внешнем наборе, классификация, captions и полный конвейер ещё не выполнены. Эти результаты нельзя получить только перестроением отчёта.

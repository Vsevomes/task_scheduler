# Матрица эксперimentов для отчёта

Скрипт: `scripts/run_report_experiments.sh` — **36 прогонов** (3 режима × 4 сценария).

## Общие настройки

| Параметр | Значение |
|----------|----------|
| Режимы | `native_cpu`, `native_gpu`, `starpu_hybrid` |
| StarPU | `STARPU_SCHED=dmda` (по умолчанию) |
| Результаты | `results/report/*.json` |
| Лог | `results/report_run.log` |

---

## 1. Matmul (9 прогонов)

| Параметр | Значения |
|----------|----------|
| **size (N×N)** | **512**, **2048**, **8192** |
| tile-size | 256 |
| Задач StarPU | 4 / 64 / 1024 тайлов |

---

## 2. Independent (9 прогонов)

| Параметр | Значения |
|----------|----------|
| **tasks** | **500**, **3000**, **5000** |
| block-size | 4096 |
| total elements | 2M / 12.3M / 20.5M |

---

## 3. Heterogeneous (9 прогонов)

| Параметр | Значения |
|----------|----------|
| **tasks** | **1000**, **3000**, **5000** |

### Распределение классов задач

| Класс | Доля | Размер (элементов) | Пример при 3000 tasks |
|-------|------|--------------------|------------------------|
| **Light** | 50% | 2048 | 1500 задач |
| **Medium** | 30% | 16384 | 900 задач |
| **Heavy** | 20% | 65536 | 600 задач |

Параметры CLI:

```text
--light-ratio 0.5 --medium-ratio 0.3
--light-size 2048 --medium-size 16384 --heavy-size 65536
```

---

## 4. Image (9 прогонов)

| Разрешение | Пикселей | MP |
|------------|----------|-----|
| **7680×4320** (8K) | 33.2 M | ~33 |
| **15360×8640** (16K) | 132.7 M | ~133 |
| **30720×17280** (32K) | 530.8 M | ~531 |

| Параметр | Значение |
|----------|----------|
| op | `blur` |
| mixed-ops | да (5 ops по тайлам) |
| tile-size | 64 |

---

## Сравнение со smoke-тестами

| Сценарий | Было | Для отчёта |
|----------|------|------------|
| matmul | 512, 1024, 2048 | 512, **2048**, **8192** |
| independent | 100, 1000 | **500**, **3000**, **5000** |
| heterogeneous | 300 tasks | **1000**, **3000**, **5000** |
| image | 640×480, 720p | **8K**, **16K**, **32K** |

---

## Запуск

```bash
cd ~/task_scheduler

# вся матрица (долго: matmul 8192 CPU + image 32K)
./scripts/run_report_experiments.sh

# по одному сценарию
./scripts/run_report_experiments.sh matmul
./scripts/run_report_experiments.sh independent
./scripts/run_report_experiments.sh heterogeneous
./scripts/run_report_experiments.sh image
```

Переменные окружения:

```bash
STARPU_SCHED=dmda STARPU_PROF=0 ./scripts/run_report_experiments.sh
REPORT_LOG=results/my_report.log ./scripts/run_report_experiments.sh image
```

---

## Оценка времени (грубо)

| Прогон | Риск |
|--------|------|
| matmul 8192 `native_cpu` | **десятки минут – часы** |
| image 32K `native_gpu` | **десятки минут** (sequential `cudaMalloc` на каждый тайл) |
| image 16K `starpu_hybrid` | ~1 мин |
| Остальное | минуты |

---

## Сводка прогонов

| Сценарий | Размеры / tasks | Режимы | Прогонов |
|----------|-----------------|--------|----------|
| matmul | 512, 2048, 8192 | 3 | 9 |
| independent | 500, 3000, 5000 | 3 | 9 |
| heterogeneous | 1000, 3000, 5000 | 3 | 9 |
| image | 8K, 16K, 32K | 3 | 9 |
| **Итого** | | | **36** |

---

## Настройка параметров

Все константы задаются в начале `scripts/run_report_experiments.sh`:

- размеры matmul, independent, heterogeneous, image;
- добавить другие ops для image (`grayscale`, `edge`, …);
- изменить `INDEPENDENT_BLOCK` (например, 8192);
- исключить тяжёлые точки (8192 CPU, 32K GPU), если прогон слишком долгий.

---

## Формат результатов

Каждый прогон создаёт:

- `results/report/<scenario>_<mode>_<param>.json` — время, метрики, параметры;
- `results/system_<scenario>_*.csv` — CPU/GPU/RAM;
- `results/system_<scenario>_*_cores.csv` — загрузка по ядрам;
- сводка в консоли через `scripts/summarize_metrics.sh` (память, ядра, overhead StarPU).

# Анализ экспериментов: StarPU на RTX 4060

**Дата сбора данных:** 6–7 июня 2026  
**Машина:** Ubuntu, `nikitos@192.168.1.218`  
**GPU:** NVIDIA GeForce RTX 4060 (sm_89), драйвер 610.43.02  
**Стек:** CUDA 12.4, StarPU 1.4.9 (Docker `starpu-cuda-rtx4060`), OpenBLAS  
**Сырые данные:** [`experiment_results/`](experiment_results/) — **128 JSON-файлов**

> В таблицах ниже — **последний прогон** для каждой уникальной комбинации (сценарий + режим + параметры). Если прогонов несколько, указано в скобках.

---

## 1. Резюме (главные выводы)

### 1.1. Что работает стабильно

- StarPU **инициализируется**, CUDA codelet'ы выполняются, JSON-метрики пишутся.
- **OpenBLAS (native_cpu)** — лучший режим для **монолитного matmul** на всех размерах 512–4096.
- **starpu_gpu** стабильно **лучше native_gpu** за счёт `starpu_cublas` и синхронизации потоков.
- **bench_hybrid tiled** при `dmdas` + `starpu_malloc` + perf models: **hybrid побеждает** (247 GFLOPS vs 144 GPU-only).

### 1.2. Где hybrid не выигрывает

- **Монолитный matmul** (`bench_matmul`, 1 task): hybrid ≈ gpu, но **оба проигрывают native_cpu**.
- **Мелкие independent tasks** (matvec 256×256): CPU-only StarPU быстрее GPU и hybrid.
- **Старый heterogeneous** (100 tasks, 256/384): **starpu_gpu** (50 ms) быстрее hybrid (117 ms).
- **Классический hybrid** без tiled/malloc/models — планировщик `lws`/`dmda` не даёт выигрыша.

### 1.3. Ключевой тезис для диплома

> **StarPU hybrid оправдан**, когда выполнены условия: **много независимых task'ов**, **разная сложность**, **`starpu_malloc`**, **performance models**, планировщик **`dmdas`**, **разбиение монолита на тайлы**.  
> Без этого CPU (OpenBLAS) или GPU-only часто быстрее.

---

## 2. Инфраструктура и методология

| Параметр | Значение |
|----------|----------|
| Запуск | Docker `starpu-cuda-rtx4060`, `USE_GPU=1`, `-e STARPU_HOME=/workspace/.starpu-home` |
| Режимы | `native_cpu`, `native_gpu`, `starpu_cpu`, `starpu_gpu`, `starpu_hybrid` |
| Метрики | `total_time_ms`, `gflops`, `avg_task_ms`, `megapixels` |
| Путь результатов (на Ubuntu) | `build/results/<сценарий>/<режим>/<timestamp>.json` |
| Локальная копия | `experiment_results/` |

**Ограничения анализа:**

- Часть сценариев прогонялась **многократно** (калибровка, сравнение `lws`/`dmda`) — в таблицах взяты **последние** значения.
- Нет усреднения по 5+ повторам с доверительными интервалами.
- `native_gpu` на малых matmul включает **полный H2D/D2H** каждый раз — занижает GFLOPS.

---

## 3. Hello — проверка StarPU

**Параметры:** `size=1048576`, 1 task (add_one).

| Режим | total_time_ms | Прогонов |
|-------|---------------|----------|
| starpu_hybrid | **0.37** | 3 |
| starpu_cpu | 0.41 | 3 |
| starpu_gpu | 1.01 | 1 |

**Вывод:** инфраструктура исправна. GPU медленнее из‑за инициализации CUDA на тривиальной задаче. **Для сравнения производительности не использовать.**

---

## 4. Matmul — сценарий 1 (основной)

**Бинарник:** `bench_matmul` — **один** task DGEMM на всю матрицу.

### 4.1. GFLOPS по размеру (последний прогон)

| N×N | native_cpu | native_gpu | starpu_cpu | starpu_gpu | starpu_hybrid | Лидер |
|-----|-----------|-----------|-----------|-----------|--------------|-------|
| 512 | **136.6** | 1.1 | 59.3 | 8.0 | 29.4 | native_cpu |
| 1024 | **198.5** | 10.0 | 108.1 | 48.1 | 114.1 | native_cpu |
| 2048 | **245.1** | 57.6 | 136.5 | 143.8 | 146.0 | native_cpu |
| 4096 | **286.8** | 160.8 | 174.3 | **208.4** | 147.6 | native_cpu |

### 4.2. Время (ms)

| N×N | native_cpu | native_gpu | starpu_cpu | starpu_gpu | starpu_hybrid |
|-----|-----------|-----------|-----------|-----------|--------------|
| 512 | **2.0** | 239.1 | 4.5 | 33.5 | 9.1 |
| 1024 | **10.8** | 214.4 | 19.9 | 44.6 | 18.8 |
| 2048 | **70.1** | 298.2 | 125.8 | 119.5 | 117.7 |
| 4096 | **479.2** | 854.7 | 788.6 | 659.4 | 931.2 |

### 4.3. Выводы matmul

1. **native_cpu (OpenBLAS) побеждает на всех N** — до **286.8 GFLOPS** на 4096².
2. **native_gpu** на N=512: **1.1 GFLOPS** — доминирует overhead `cudaMalloc` + копирование (~239 ms).
3. **starpu_gpu vs native_gpu** на 4096: **208 vs 161 GFLOPS** — StarPU + cuBLAS даёт **+29%**.
4. **starpu_hybrid ≈ starpu_gpu** на 2048 (146 vs 144 GFLOPS) — **один task нельзя разделить** между CPU и GPU; hybrid не даёт преимущества.
5. На 4096 **starpu_gpu (208) > starpu_cpu (174)**, но оба **ниже native_cpu (287)**.
6. **Crossover starpu_gpu vs starpu_cpu:** между 2048 и 4096.

**Speedup starpu_gpu относительно native_gpu (4096):** 659/855 = **0.77× по времени** (быстрее на 23%).

---

## 5. Independent — сценарий 2 (пакет задач)

**Суть:** много одинаковых matvec-задач.

### 5.1. Классические прогоны (size=256)

| Режим | tasks | total_ms | avg_task_ms | Прогонов |
|-------|-------|----------|-------------|----------|
| starpu_cpu | 100 | 505.1 | 5.05 | 2 |
| starpu_cpu | 1000 | **473.3** | 0.47 | 2 |
| starpu_gpu | 100 | 624.1 | 6.24 | 2 |
| starpu_gpu | 1000 | 658.4 | 0.66 | 2 |
| starpu_hybrid | 100 | 633.4 | 6.33 | 2 |
| starpu_hybrid | 1000 | 635.1 | 0.64 | 4 |

### 5.2. Крупный пакет (report-эксперимент, block-size=4096)

| Режим | tasks | total_ms | avg_task_ms |
|-------|-------|----------|-------------|
| native_gpu | 5000 | **168.9** | 0.034 |
| starpu_hybrid | 5000 | 267.4 | 0.053 |

### 5.3. Выводы independent

1. При **1000 tasks × size 256**: время **почти не растёт** vs 100 tasks (473 vs 505 ms) — **параллелизм работает**.
2. **starpu_cpu быстрее GPU/hybrid** на мелких задачах — overhead transfer > выигрыш GPU.
3. На **5000 крупных блоках** `native_gpu` (**169 ms**) быстрее `starpu_hybrid` (**267 ms**) — чистый CUDA batch эффективнее StarPU scheduling на этом сценарии.
4. **avg_task_ms ~0.47 ms (CPU)** vs **~60 µs overhead** (см. §9) — полезная работа >> overhead только при 1000+ tasks.

---

## 6. Heterogeneous — сценарий 3 (разная сложность)

### 6.1. Базовый прогон (100 tasks, light 256 / heavy 384, ratio 70%)

| Режим | total_ms | Прогонов |
|-------|----------|----------|
| **starpu_gpu** | **50.3** | 2 |
| starpu_cpu | 113.4 | 2 |
| starpu_hybrid (dmda) | 117.4 | 5 |

*Лучший разовый hybrid (dmda): **59.9 ms** — нестабильно.*

### 6.2. Report-эксперименты (крупная нагрузка)

| Конфигурация | Режим | tasks | total_ms |
|--------------|-------|-------|----------|
| light=512, medium=4096, heavy=131072 | **native_gpu** | 2000 | **414.9** |
| то же | starpu_hybrid | 2000 | 1824.6 |
| light=128, medium=2048, heavy=8M el. | **native_gpu** | 100 | **967.3** |
| то же | starpu_hybrid | 100 | 11358.6 |

### 6.3. Выводы heterogeneous

1. На **малых** задачах (100×256/384) **GPU-only StarPU выигрывает** у CPU и hybrid.
2. **Hybrid с dmda нестабилен** (59–117 ms на одной конфигурации).
3. На **report-крупных** задачах **native_gpu доминирует** — StarPU hybrid добавляет overhead без выигрыша от распределения.
4. Тяжёлые задачи с `heavy_size=8388608` на hybrid **11.4 с** — вероятно неоптимальное размещение данных / копирование.

---

## 7. Image — сценарий 4 (обработка изображений)

**Фильтр:** преимущественно `blur`; одно `grayscale` 1280×720.

### 7.1. Время по разрешению (blur, последний прогон)

| Разрешение | MP | native_cpu | native_gpu | starpu_hybrid |
|------------|-----|-----------|-----------|--------------|
| 3840×2160 (4K) | 8.3 | **62 ms** | 495 ms | 237 ms |
| 7680×4320 (8K) | 33.2 | **249–256 ms** | 1380–4549 ms* | **337–633 ms** |
| 15360×8640 (16K) | 132.7 | 986 ms | 4929 ms | **822 ms** |

\* Разброс native_gpu на 8K — разные прогоны (cold start / повтор).

### 7.2. Пропускная способность (ms на мегапиксель, blur)

| Режим | 4K | 8K | 16K |
|-------|-----|-----|------|
| native_cpu | **7.5** | **7.5** | 7.4 |
| native_gpu | 59.7 | 41.6–137 | 37.1 |
| starpu_hybrid | 28.5 | **10.2–19.1** | **6.2** |

### 7.3. Выводы image

1. **4K:** CPU-native **в 4–8× быстрее** GPU и hybrid — tile overhead StarPU велик на среднем размере.
2. **16K:** **starpu_hybrid (822 ms) побеждает** native_cpu (986 ms) и особенно native_gpu (4929 ms) — на **очень больших** изображениях гетерогенное расписание + тайлы окупаются.
3. **8K:** результаты неоднозначны (337 vs 633 ms hybrid) — нужны повторы.
4. Grayscale 1280×720 hybrid: **592 ms** — аномально долго (вероятно cold calibration).

---

## 8. Hybrid stress — bench_hybrid (усиленные условия)

**Условия:** `starpu_malloc`, `STARPU_HISTORY_BASED` perf models, планировщик **`dmdas`**, warmup.

### 8.1. Mixed — 2000 tasks (80% light matvec 128, 20% heavy matmul 768)

| Режим | total_ms | Относительно лучшего |
|-------|----------|----------------------|
| starpu_cpu | 3926 | 2.37× медленнее |
| **starpu_gpu** | **1659** | **1.00× (лидер)** |
| starpu_hybrid | 1961 | 1.18× медленнее |

**Вывод:** при 80% лёгких задач scheduler всё ещё отправляет часть на GPU → hybrid **между CPU и GPU**, но **не побеждает GPU-only**.

### 8.2. Tiled matmul — 4096², tile 256, 256 tasks

| Режим | total_ms | GFLOPS | Относительно лучшего |
|-------|----------|--------|----------------------|
| starpu_cpu | 2236 | 61.5 | 4.0× медленнее |
| starpu_gpu | 957 | 143.7 | 1.7× медленнее |
| **starpu_hybrid** | **557** | **246.9** | **1.00× (лидер)** |

**Вывод:** **главный положительный результат проекта** — hybrid на tiled matmul:

- **1.72× быстрее** starpu_gpu (957 → 557 ms)
- **4.0× быстрее** starpu_cpu
- **246.9 GFLOPS** — сопоставимо с **native_cpu 286.8** на монолитном matmul

Это подтверждает: hybrid выигрывает, когда **CPU и GPU заняты разными тайлами одновременно**.

---

## 9. Overhead — накладные расходы StarPU

**Параметры:** `kind=noop`, `tasks=10000`, `bytes=1048576`, `starpu_hybrid`.

| Метрика | Значение |
|---------|----------|
| total_time_ms | 598.8 (2 прогона, σ≈13.5 ms) |
| **avg_task_us** | **59.9 µs** |

**Вывод:** базовая цена одного пустого task ≈ **60 µs**. Задачи короче **0.1 ms** невыгодны для StarPU. При avg_task_ms **0.47 ms** (independent CPU) overhead составляет ~**13%** теоретически; на GPU transfer увеличивает долю.

---

## 10. Сводная таблица: кто побеждает по сценариям

| Сценарий | Лучший режим | Комментарий |
|----------|--------------|-------------|
| hello | starpu_hybrid | Только sanity, не performance |
| matmul (монолит) | **native_cpu** | На всех N=512–4096 |
| matmul (tiled 4096) | **starpu_hybrid** | 247 GFLOPS, dmdas+malloc |
| independent (256, 1k tasks) | **starpu_cpu** | GPU проигрывает |
| independent (5k blocks) | **native_gpu** | 169 ms vs hybrid 267 ms |
| heterogeneous (100, малый) | **starpu_gpu** | 50 ms |
| heterogeneous (report, 2k) | **native_gpu** | 415 ms |
| image 4K blur | **native_cpu** | 62 ms |
| image 16K blur | **starpu_hybrid** | 822 ms vs CPU 986 ms |
| hybrid mixed 2k | **starpu_gpu** | 1659 ms |
| overhead | — | ~60 µs/task baseline |

---

## 11. Ответы на исследовательские вопросы

### В1. Работает ли StarPU на RTX 4060 (sm_89)?

**Да.** Все режимы отрабатывают в Docker с CUDA 12.4 и StarPU 1.4.9.

### В2. Когда GPU быстрее CPU?

- Монолитный matmul: **почти никогда** в наших данных (OpenBLAS слишен).
- starpu_gpu vs starpu_cpu: при **N ≥ 4096** (208 vs 174 GFLOPS).
- Tiled matmul hybrid: GPU+CPU **вместе** дают 247 GFLOPS.
- Image **16K**: hybrid быстрее CPU.
- Heterogeneous малый: **starpu_gpu**.

### В3. Оправдан ли hybrid?

| Условие | Выполнено? | Результат |
|---------|------------|-----------|
| Много task'ов | tiled 256 / mixed 2000 | tiled ✅, mixed ⚠️ |
| Разная тяжесть | mixed 128/768 | частично |
| starpu_malloc | bench_hybrid | ✅ |
| perf models + dmdas | bench_hybrid | ✅ |
| Параллельная загрузка CPU+GPU | tiled | **✅ 247 GFLOPS** |

### В4. Главные bottlenecks

1. **Host↔device transfer** (warnings StarPU про `starpu_malloc` даже после его использования в части путей).
2. **OpenBLAS на CPU** — очень высокий потолок FP64.
3. **Планировщик** без calibrate → нестабильные результаты hybrid.
4. **Монолитный matmul** — принципиально не подходит для hybrid.

---

## 12. Рекомендации для следующих экспериментов

1. **Повторы:** 5 прогонов, median + σ для каждой точки.
2. **Warmup:** отдельный прогон до замера (особенно native_gpu, image).
3. **Matmul report:** использовать **tiled** вместо монолита для сравнения hybrid.
4. **Mixed hybrid:** снизить `light_ratio` до 0.5, увеличить `heavy_size` до 1024.
5. **Pinned memory** везде, не только bench_hybrid.
6. **Графики для диплома:**
   - GFLOPS vs N (matmul, 5 режимов)
   - bar chart bench_hybrid tiled (3 режима)
   - image: ms/MP vs разрешение
   - independent: total_time vs tasks

---

## 13. Структура `experiment_results/`

```
experiment_results/
├── README.md
├── _summary.json          # агрегат для скриптов
├── hello/
├── matmul/                # 71 файл
├── independent/           # 16
├── heterogeneous/         # 13
├── image/                 # 13
├── overhead/              # 2
└── hybrid/
    ├── mixed/             # 3 режима
    └── tiled/             # 3 режима
```

---

## 14. Заключение

Эксперименты на **128 прогонах** показывают **двухрежимную картину**:

1. **Без специальной подготовки** (монолитный matmul, мелкие tasks, `lws`/ранний `dmda`) — **OpenBLAS CPU или GPU-only** выигрывают; **hybrid не оправдан**.

2. **При выполнении условий** (`bench_hybrid`: tiled decomposition, `starpu_malloc`, history perf models, `dmdas`) — **hybrid достигает 246.9 GFLOPS** на matmul 4096² и **лучшего времени на image 16K**, демонстрируя **реальное параллельное использование CPU+GPU**.

Для дипломной работы рекомендуется строить аргументацию вокруг **условий применимости** StarPU hybrid, а не вокруг утверждения «hybrid всегда быстрее».

---

*Сгенерировано по данным `experiment_results/`. Для обновления: `rsync` с Ubuntu + пересчёт `_summary.json`.*

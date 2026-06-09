# Полный отчёт по эксперimentам task_scheduler

Платформа: Ubuntu, RTX 4060, CUDA 12.4, StarPU dmda (hybrid).

Источники данных:
- `results/report/*.json` — 51 прогон
- `results/system_*.csv` — CPU/GPU/RAM
- `results/system_*_cores.csv` — загрузка по ядрам
- Логи: `report_run.log` (полная матрица), `report_x3.log` (independent/heterogeneous ×3)

---

## 1. Executive summary

| Метрика | Значение |
|---------|----------|
| Всего JSON-прогонов | 51 |
| Полная матрица (36) | matmul + image + independent/hetero (старые tasks) |
| Доп. прогон ×3 (18) | independent 1500/9000/15000, heterogeneous 3000/9000/15000 |
| Суммарное время всех прогонов | 15.46 h |
| Победитель по min time (51 точка) | native_gpu: 12, starpu_hybrid: 2, native_cpu: 3 |

**Ключевые выводы:**
- **native_gpu** доминирует на matmul, independent (крупные N), heterogeneous.
- **starpu_hybrid** выигрывает на **image 16K и 32K** (blur, mixed-ops).
- **native_cpu** быстрее на мелких задачах (independent 500–1500, image 8K).
- StarPU overhead на hybrid: init ~200 ms + registration/submit/unregister; на image 16K overhead ~94%, на matmul 8192 wait фаза аномально длинная (CPU-тайлы).

---

## 2. Matmul

Параметры: tile=256, sizes 512 / 2048 / 8192, режимы native_cpu / native_gpu / starpu_hybrid.

| Size | native_cpu | native_gpu | starpu_hybrid | GFLOPS (cpu/gpu/st) | Speedup CPU→GPU | Speedup CPU→StarPU |
|-----:|----------:|-----------:|--------------:|--------------------:|----------------:|-------------------:|
| 512 | 758.3 ms | 206.2 ms | 380.2 ms | 0.35 / 1.3 / 0.71 | 4× | 2.0× |
| 2048 | 3.8 min | 246.9 ms | 294.0 ms | 0.08 / 69.6 / 58.44 | 925× | 777.1× |
| 8192 | 14.72 h | 4.65 s | 29.3 min | 0.02 / 236.6 / 0.63 | 11406× | 30.2× |

### 2.1 Анализ matmul
- GPU на 8192: **~237 GFLOPS**, CPU **~14.7 ч** — baseline CPU неприменим на больших N.
- StarPU 8192: **~29 мин**, 0.63 GFLOPS — dmda отправляет часть тайлов на CPU; wait **~29 мин**.
- StarPU 2048/512: отстаёт от GPU в **1.5–2×**, но быстрее CPU на 2048.

## 3. Independent

block_size=4096. Два набора task counts: **базовый** (500/3000/5000) и **×3** (1500/9000/15000).

### 3.1 Базовый набор

| tasks | native_cpu | native_gpu | starpu_hybrid | elems | Speedup→GPU |
|------:|----------:|-----------:|--------------:|------:|------------:|
| 500 | 38.9 ms | 184.2 ms | 210.2 ms | 2.0M | 0.21× |
| 3000 | 236.5 ms | 180.7 ms | 262.8 ms | 12.3M | 1.31× |
| 5000 | 390.6 ms | 196.6 ms | 319.1 ms | 20.5M | 1.99× |

### 3.2 Набор ×3

| tasks | native_cpu | native_gpu | starpu_hybrid | elems | Speedup→GPU |
|------:|----------:|-----------:|--------------:|------:|------------:|
| 1500 | 126.9 ms | 243.4 ms | 261.8 ms | 6.1M | 0.52× |
| 9000 | 769.5 ms | 225.5 ms | 415.1 ms | 36.9M | 3.41× |
| 15000 | 1.17 s | 273.5 ms | 598.7 ms | 61.4M | 4.28× |

### 3.3 Масштабируемость independent (native_gpu, ms vs tasks)

| tasks | base | ×3 series |
|------:|-----:|----------:|
| 500 → 1500 | 184.2 ms | 243.4 ms |
| 3000 → 9000 | 180.7 ms | 225.5 ms |
| 5000 → 15000 | 196.6 ms | 273.5 ms |

### 3.4 StarPU overhead (hybrid, 15000 tasks)
- total: 598.7 ms; init: 217 ms (36.3%); wait: 87 ms

**Вывод:** при tasks≥3000 GPU быстрее CPU; StarPU проигрывает из‑за init (~190–240 ms). ×3 линейно масштабирует время CPU/GPU.

## 4. Heterogeneous

Mix: light 50% (2048), medium 30% (16384), heavy 20% (65536).

### 4.1 Базовый набор (1000 / 3000 / 5000)

| tasks | native_cpu | native_gpu | starpu_hybrid | CPU/GPU | StarPU/GPU |
|------:|----------:|-----------:|--------------:|--------:|-----------:|
| 1000 | 18.51 s | 479.6 ms | 485.9 ms | 39× | 1.01× |
| 3000 | 55.76 s | 998.5 ms | 1.03 s | 56× | 1.03× |
| 5000 | 1.5 min | 1.49 s | 1.61 s | 62× | 1.07× |

### 4.2 Набор ×3 (3000 / 9000 / 15000)

| tasks | native_cpu | native_gpu | starpu_hybrid | CPU/GPU | StarPU/GPU |
|------:|----------:|-----------:|--------------:|--------:|-----------:|
| 3000 | 55.76 s | 998.5 ms | 1.03 s | 56× | 1.03× |
| 9000 | 2.8 min | 2.54 s | 2.69 s | 66× | 1.06× |
| 15000 | 4.6 min | 4.06 s | 4.32 s | 68× | 1.06× |

**Вывод:** GPU **39–68×** быстрее CPU. StarPU в пределах **1.01–1.07×** от GPU на всех точках (после калибровки dmda). Параметр 3000 tasks есть в обеих сериях (один JSON — последний прогон).

## 5. Image (blur, mixed-ops, tile=64)

| Resolution | MP | native_cpu | native_gpu | starpu_hybrid | MP/s (cpu/gpu/st) | Best |
|------------|---:|----------:|-----------:|--------------:|------------------:|------|
| 7680x4320 | 33 | 252.8 ms | 1.38 s | 344.4 ms | 131/24/96 | native_cpu |
| 15360x8640 | 133 | 999.6 ms | 4.88 s | 835.6 ms | 133/27/159 | starpu_hybrid |
| 30720x17280 | 531 | 3.98 s | 19.10 s | 2.65 s | 133/28/200 | starpu_hybrid |

### StarPU 16K breakdown: total 835.6 ms, init 229 ms, wait 38 ms, overhead 95%

**Вывод:** StarPU побеждает на **16K и 32K**; native_gpu проигрывает из‑за sequential cudaMalloc на тайл; 8K быстрее на CPU.

## 6. Системные метрики (RAM, GPU util, CPU cores)

Примеры репрезентативных прогонов:

| Прогон | RAM peak | GPU util max | Top core peak | Active cores |
|--------|----------:|-------------:|--------------:|--------------|
| matmul 8192 native_gpu | 1923 MB | 7% | CPU0 90% | CPU0, CPU1, CPU7, CPU4, CPU8, CPU2 |
| matmul 8192 starpu_hybrid | 1859 MB | 8% | CPU6 90% | CPU6, CPU7, CPU8, CPU9, CPU10, CPU11 |
| image 15360x8640 starpu_hybrid | 303 MB | 7% | CPU6 90% | CPU6, CPU4, CPU11, CPU7, CPU8, CPU5 |
| heterogeneous 15000 starpu_hybrid | 4005 MB | 8% | CPU10 97% | CPU10, CPU1, CPU7, CPU11, CPU0, CPU6 |
| independent 15000 native_gpu | 486 MB | 7% | CPU0 90% | CPU0, CPU6, CPU5, CPU2, CPU1, CPU8 |

> GPU util в nvidia-smi занижен (семпл 0.5 с); peak core load надёжнее для CPU/GPU balance.

## 7. Накладные расходы StarPU (hybrid)

| Сценарий | param | total | init | reg | submit | wait | unreg | overhead% |
|----------|-------|------:|-----:|----:|-------:|-----:|------:|----------:|
| matmul | 512 | 380.2 ms | 186 | 1 | 0 | 192 | 1 | 50% |
| matmul | 8192 | 29.3 min | 213 | 259 | 128 | 29.3 min | 153 | 2%† |
| independent | 15000 | 598.7 ms | 217 | 53 | 84 | 87 | 158 | 85% |
| independent | 5000 | 319.1 ms | 188 | 18 | 29 | 33 | 52 | 90% |
| heterogeneous | 15000 | 4.32 s | 239 | 53 | 64 | 3538 | 431 | 18% |
| heterogeneous | 5000 | 1.61 s | 226 | 18 | 22 | 1198 | 142 | 25% |
| image | 15360x8640 | 835.6 ms | 229 | 315 | 137 | 38 | 112 | 95% |

† matmul 8192: wait **~29 мин** (CPU-тайлы) доминирует; overhead без wait **~2%**.

---

## 8. Рекомендации для текста диплома

1. **Matmul:** сравнивать GPU vs StarPU на 512–2048; 8192 CPU исключить или привести как asymptotic baseline.
2. **Independent:** показать порог tasks≈3000, где GPU обгоняет CPU; StarPU — отдельно overhead init.
3. **Heterogeneous:** GPU vs StarPU на 5000/15000 (и 1000 base); StarPU ≈ GPU после калибровки dmda.
4. **Image:** главный кейс победы StarPU — 16K/32K, throughput MP/s, active cores.
5. **Метрики:** JSON + CSV cores покрывают время, RAM, load; GPU util — с оговоркой о sampling.

---

*Сгенерировано автоматически из 51 JSON в `results/report/`.*

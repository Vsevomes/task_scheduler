# Сырые результаты экспериментов

Скопировано с Ubuntu (`nikitos@192.168.1.218:~/task_scheduler/build/results/`) **6 июня 2026**.

## Содержимое

| Путь | Описание |
|------|----------|
| `hello/` | Sanity-check StarPU (1M элементов) |
| `matmul/` | Умножение матриц N×N, 5 режимов |
| `independent/` | Пакет независимых задач |
| `heterogeneous/` | Смешанная нагрузка (light/heavy/medium) |
| `image/` | Фильтры blur/grayscale на изображениях |
| `overhead/` | Накладные расходы StarPU (noop tasks) |
| `hybrid/mixed/` | bench_hybrid: 2000 mixed tasks |
| `hybrid/tiled/` | bench_hybrid: tiled matmul 4096² |

## Формат JSON

```json
{
  "scenario": "matmul",
  "mode": "starpu_hybrid",
  "params": { "size": "2048" },
  "metrics": {
    "total_time_ms": 117.66,
    "gflops": 146.01
  }
}
```

Имя файла: `YYYYMMDD_HHMMSS.json` — время прогона (UTC+3 на машине nikitos).

## Статистика

- **Всего файлов:** 128 JSON
- **Полный анализ:** см. [`../ANALYSIS_EXPERIMENTS.md`](../ANALYSIS_EXPERIMENTS.md)
- **Сводка (машиночитаемая):** `_summary.json`

## Повторная синхронизация с Ubuntu

```bash
rsync -avz nikitos@192.168.1.218:~/task_scheduler/build/results/ experiment_results/
```

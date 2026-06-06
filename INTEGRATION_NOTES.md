# Integration Notes

Qualitative assessment of StarPU integration complexity for the research thesis.

## Platform deviation from spec

The thesis specifies CUDA 10.2. RTX 4060 (Ada Lovelace, sm_89) requires CUDA ≥ 11.8.
We use **CUDA 12.4** as the sole target. This is documented in the report as a hardware constraint, not a design choice.

HIP/AMD paths from the spec are out of scope (NVIDIA-only platform).

## Build environment

| Aspect | Difficulty | Notes |
|--------|------------|-------|
| StarPU from source in Docker | Medium | ~20 min first build; Dockerfile handles deps |
| CMake + pkg-config | Low | `pkg-config starpu-1.4` after install |
| CUDA arch sm_89 | Low | `-DCMAKE_CUDA_ARCHITECTURES=89` |
| Native host build | High | `setup_rtx4060.sh` installs runtime only, not dev headers |

**Recommendation:** use Docker as the primary build path.

## Code structure per scenario

Each scenario follows the same pattern:

1. **Native CPU** — direct call to OpenBLAS or hand-written loops
2. **Native GPU** — cuBLAS or CUDA kernels via `cudaMalloc`/`cublas*`
3. **StarPU** — `starpu_codelet` with `.cpu_funcs` + `.cuda_funcs`, data handles, `starpu_task_submit`

### Lines of code (approximate, excluding common/)

| Scenario | Native paths | StarPU path | Codelets |
|----------|-------------|-------------|----------|
| matmul | ~40 lines | ~50 lines | 1 (CPU+cuda) |
| independent | — | ~80 lines | 1 |
| heterogeneous | — | ~120 lines | 1 (dual logic) |
| image | ~60 lines CPU + ~40 GPU | ~80 lines | 1 (tiled) |
| overhead | — | ~60 lines | 2 (noop, memcpy) |

StarPU adds roughly **1.5–2×** code volume vs native for equivalent compute, mostly boilerplate (handles, task creation, init/shutdown).

## API usability

**Pros:**
- Single codelet definition covers CPU and GPU variants
- Data handles abstract migration between memories
- Built-in profiling (`STARPU_PROF=1`) for scheduling analysis

**Cons:**
- Must register data before submitting tasks
- `cl_arg` lifetime must outlive tasks (use persistent storage)
- Separate CUDA compilation unit required for `__global__` kernels / cuBLAS in codelets
- Worker restriction (`ncpu=0` / `ncuda=0`) requires understanding of `starpu_conf`

## Debugging & profiling

| Tool | Purpose |
|------|---------|
| `STARPU_PROF=1` + `STARPU_FXT_PREFIX` | Task timeline traces |
| `scripts/collect_metrics.sh` | CPU/GPU/memory sampling |
| `std::chrono` in benchmarks | Wall-clock JSON output |
| CUDA events | Native GPU kernel timing |
| `nvidia-smi`, `mpstat` | External utilization (thesis spec) |

## Known issues

1. **Remote driver mismatch** — if `nvidia-smi` fails with "Driver/library version mismatch", reboot the machine after driver update.
2. **Docker GPU** — requires `nvidia-container-toolkit` and `--gpus all`.
3. **Image blur on GPU** — simplified to grayscale in CUDA tile path; full convolution blur runs on CPU/StarPU-CPU path.

## Metrics mapping to thesis

| Thesis metric | Implementation |
|---------------|----------------|
| Execution time | `MetricsWriter` JSON (`total_time_ms`, `gflops`) |
| CPU/GPU load | `collect_metrics.sh` → CSV |
| Memory usage | `collect_metrics.sh` (RAM + VRAM columns) |
| StarPU overhead | `bench_overhead` (noop vs memcpy tasks) + StarPU traces |
| Integration complexity | this document |

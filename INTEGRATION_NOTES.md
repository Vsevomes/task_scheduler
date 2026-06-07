# Integration Notes

Qualitative assessment of StarPU integration complexity for the research thesis.

## Platform Deviation From Spec

The thesis specifies CUDA 10.2. RTX 4060 (Ada Lovelace, sm_89) requires CUDA >= 11.8.
Experiments use **CUDA 12.4** as a hardware constraint, not as a methodology change.

HIP/AMD paths from the spec are out of scope for this NVIDIA-only platform.

## Build Environment

| Aspect | Difficulty | Notes |
|--------|------------|-------|
| Docker StarPU/CUDA build | Medium | Dockerfile handles StarPU and CUDA dependencies |
| CMake + pkg-config | Low | Uses `starpu-1.4`, falling back to `starpu-1.3` |
| CUDA arch sm_89 | Low | `-DCMAKE_CUDA_ARCHITECTURES=89` |
| Native host build | High | Requires matching CUDA dev packages and StarPU contrib |

**Recommendation:** use Docker as the primary build path.

## Methodology Shape

The final matrix uses three modes:

1. **Native CPU** — simple scalar CPU loops, no StarPU.
2. **Native GPU** — straightforward CUDA kernels, no StarPU.
3. **StarPU hybrid** — one StarPU codelet with CPU and CUDA implementations, data handles, and many submitted tasks where the scenario supports tiling/blocking.

StarPU-only CPU/GPU modes are excluded because they evaluate StarPU as a wrapper rather than as a heterogeneous scheduler.

## Scenario Notes

| Scenario | Native paths | StarPU path | Main limitation |
|----------|--------------|-------------|-----------------|
| matmul | scalar CPU loop + simple CUDA kernel | independent output-tile tasks | simple kernels are intentionally not BLAS/cuBLAS-optimized |
| independent | linear array loop + flat CUDA kernel | one task per array block | synthetic arithmetic workload |
| heterogeneous | sequential class-based CPU loop + CUDA task grid | mixed light/medium/heavy tasks submitted together | task classes are synthetic but isolate scheduler behavior |
| image | whole-image CPU/GPU functions | one task per image tile | convolution borders use simplified per-tile no-halo handling |
| overhead | none | noop/memcpy microbenchmarks | auxiliary only, not a replacement for per-scenario overhead analysis |

## Integration Complexity

StarPU adds complexity mostly through:

- data handle registration and unregistering;
- `starpu_task` creation and lifetime;
- stable `cl_arg` storage until task completion;
- separate CUDA compilation units for kernels used by codelets;
- interpreting traces and separating scheduling time from compute/data movement.

The native paths are intentionally plain so the comparison focuses on scheduling and data movement rather than tuned library performance.

## Scheduling And Data Movement Overhead

The main JSON outputs record total runtime and task/block/tile counts. StarPU traces are enabled by default through `STARPU_PROF=1`, and each JSON result links to the trace prefix through `starpu_trace_prefix`.

System metrics are sampled by `scripts/collect_metrics.sh`; each JSON result links the matching CSV through `system_metrics_csv`.

`bench_overhead` remains as an auxiliary microbenchmark for noop task submission and memcpy-style data movement. It should be reported separately from the main scenario comparisons.

## Debugging And Profiling

| Tool | Purpose |
|------|---------|
| `STARPU_PROF=1` + `STARPU_FXT_PREFIX` | Task timeline traces |
| `scripts/collect_metrics.sh` | CPU/GPU/memory sampling |
| Benchmark JSON | Runtime, parameters, task counts, result file linkage |
| `nvidia-smi`, `mpstat` | External utilization checks |

## Known Issues And Assumptions

1. **Remote driver mismatch** — if `nvidia-smi` fails with "Driver/library version mismatch", reboot after driver update.
2. **Docker GPU** — requires `nvidia-container-toolkit` and `--gpus all`.
3. **Image tile borders** — StarPU image blur/convolution uses simplified per-tile borders without halo exchange; this is recorded in result metadata.
4. **Synthetic workloads** — independent and heterogeneous scenarios are designed to expose scheduling behavior, not to model a specific production application.

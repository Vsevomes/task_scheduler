# task_scheduler

StarPU research benchmarks for heterogeneous CPU+GPU scheduling on **NVIDIA RTX 4060** (sm_89).

## Stack

| Component | Version |
|-----------|---------|
| GPU | RTX 4060 (compute capability 8.9) |
| CUDA | 12.4 |
| StarPU | 1.3.9 contrib (Docker) / 1.4.9 (thesis target) |
| OS | Ubuntu 22.04 |
| Build | Docker (recommended) |

> **Note on CUDA 10.2 (thesis spec):** RTX 4060 requires CUDA >= 11.8. Experiments run on CUDA 12.4; see `INTEGRATION_NOTES.md` for rationale.

## Quick start

```bash
# Build Docker image
./scripts/build-docker.sh

# Build all benchmarks
./starpu-rtx4060-build

# Build with GPU passthrough
USE_GPU=1 ./starpu-rtx4060-build

# Run a single experiment with system metrics sampling
./scripts/run_experiment.sh matmul --mode starpu_hybrid --size 2048 --tile-size 256

# Run full experiment matrix
./scripts/run_all_experiments.sh
```

## Execution Modes

- `native_cpu` — simple scalar CPU loops, no StarPU.
- `native_gpu` — straightforward CUDA kernels, no StarPU.
- `starpu_hybrid` — StarPU dynamic scheduling across CPU and CUDA workers.

StarPU-only CPU/GPU wrapper modes are intentionally excluded from the main matrix. StarPU is evaluated as a heterogeneous scheduler.

## Benchmarks

| Binary | Scenario | Modes |
|--------|----------|-------|
| `bench_matmul` | Tiled matrix multiplication | native_cpu, native_gpu, starpu_hybrid |
| `bench_independent` | Independent array-processing blocks | native_cpu, native_gpu, starpu_hybrid |
| `bench_heterogeneous` | Mixed light/medium/heavy array tasks | native_cpu, native_gpu, starpu_hybrid |
| `bench_image` | Tiled image operations | native_cpu, native_gpu, starpu_hybrid |
| `bench_overhead` | Auxiliary noop/memcpy StarPU microbenchmarks | starpu_hybrid |

### Matrix Multiplication

Sizes in the default matrix: 512x512, 1024x1024, 2048x2048.

`starpu_hybrid` submits one independent task per output tile and records `tile_size` and `submitted_tasks` in JSON.

### Independent Tasks

Each block computes:

```text
y[i] = sin(x[i]) + sqrt(x[i]) + x[i]^2
```

The task count and block size are varied to study scheduler behavior under different parallelism levels.

### Heterogeneous Workload

The scenario submits light, medium, and heavy array-processing tasks together. Task classes differ by input size and arithmetic intensity, creating uneven load for dynamic scheduling.

### Image Scenario

Operations:

- `grayscale`
- `blur`
- `edge`
- `convolution`
- `filter`

StarPU image execution uses tiles. Blur/convolution borders are handled per tile with simplified no-halo borders; this policy is recorded in JSON.

## Results

JSON timing results: `results/<scenario>/<mode>/<timestamp>.json`

Every JSON file includes scenario, mode, params, metrics, `system_metrics_csv`, and `starpu_trace_prefix`.

System metrics: `results/system_*.csv`

StarPU traces when `STARPU_PROF=1`: `results/trace_*`

## Remote Machine

```bash
rsync -avz --exclude build --exclude results . nikitos@192.168.1.218:~/task_scheduler/
ssh nikitos@192.168.1.218 'cd ~/task_scheduler && ./scripts/build-docker.sh && ./starpu-rtx4060-build'
```

If `nvidia-smi` reports **Driver/library version mismatch**, reboot the machine:

```bash
sudo reboot
```

After reboot:

```bash
nvidia-smi
cd ~/task_scheduler
./scripts/build-docker.sh
./starpu-rtx4060-build
./scripts/run_all_experiments.sh
```

## Project Layout

```text
src/common/        timers, metrics JSON, StarPU init
src/matmul/        scenario 1
src/independent/   scenario 2
src/heterogeneous/ scenario 3
src/image/         scenario 4
src/overhead/      auxiliary StarPU overhead microbenchmarks
scripts/           build and experiment runners
docker_rtx4060/    CUDA 12.4 + StarPU Dockerfile
```

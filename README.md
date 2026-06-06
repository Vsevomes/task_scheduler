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

> **Note on CUDA 10.2 (thesis spec):** RTX 4060 requires CUDA ≥ 11.8. Experiments run on CUDA 12.4; see `INTEGRATION_NOTES.md` for rationale.

## Quick start

```bash
# Build Docker image (uses cached nvidia/cuda base + apt StarPU contrib)
./scripts/build-docker.sh

# Build all benchmarks (no GPU needed for compile)
./starpu-rtx4060-build

# Run with GPU passthrough (after driver fix on host)
USE_GPU=1 ./starpu-rtx4060-build

# Run a single experiment with system metrics sampling
./scripts/run_experiment.sh matmul --mode starpu_hybrid --size 2048

# Run full experiment matrix
./scripts/run_all_experiments.sh
```

## Benchmarks

| Binary | Scenario | Modes |
|--------|----------|-------|
| `hello_starpu` | StarPU sanity check | starpu_cpu, starpu_gpu, starpu_hybrid |
| `bench_matmul` | Matrix multiplication | all 5 modes |
| `bench_independent` | N identical matvec tasks | starpu_* |
| `bench_heterogeneous` | Mixed light/heavy tasks | starpu_* |
| `bench_image` | Grayscale / blur / threshold | all 5 modes |
| `bench_overhead` | Empty & memcpy StarPU tasks | starpu_* |

### Execution modes

- `native_cpu` — OpenBLAS, no StarPU
- `native_gpu` — cuBLAS, no StarPU
- `starpu_cpu` — StarPU, CPU workers only (`STARPU_NCUDA=0`)
- `starpu_gpu` — StarPU, GPU workers only (`STARPU_NCPU=0`)
- `starpu_hybrid` — StarPU dynamic scheduling

## Results

JSON timing results: `results/<scenario>/<mode>/<timestamp>.json`

System metrics (CPU/GPU/memory): `results/system_*.csv`

StarPU traces (when `STARPU_PROF=1`): `results/trace_*`

## Remote machine (192.168.1.218)

```bash
rsync -avz --exclude build --exclude results . nikitos@192.168.1.218:~/task_scheduler/
ssh nikitos@192.168.1.218 'cd ~/task_scheduler && ./scripts/build-docker.sh && ./starpu-rtx4060-build'
```

If `nvidia-smi` reports **Driver/library version mismatch**, reboot the machine:

```bash
sudo reboot
```

After reboot, verify GPU and rebuild/run:

```bash
nvidia-smi
cd ~/task_scheduler
./scripts/build-docker.sh    # if Dockerfile changed
./starpu-rtx4060-build
./build/hello_starpu --mode starpu_hybrid --size 1048576
```

If Docker build fails with `TLS handshake timeout`, retry when network to Docker Hub is stable, or pull the base image manually:

```bash
docker pull nvidia/cuda:12.4.0-base-ubuntu22.04
./scripts/build-docker.sh
```

Docker image uses `libstarpu-contrib-dev` (CUDA StarPU 1.3.9) and CUDA 12.4 dev packages (`nvcc`, `nvml`, `cusparse`, `cublas`).

## Project layout

```
src/common/       timers, metrics JSON, StarPU init
src/matmul/       scenario 1
src/independent/  scenario 2
src/heterogeneous/ scenario 3
src/image/        scenario 4
src/overhead/     StarPU overhead micro-benchmarks
scripts/          build & experiment runners
docker_rtx4060/   CUDA 12.4 + StarPU Dockerfile
```

#include "starpu_runtime.hpp"

#include <starpu.h>
#include <starpu_cublas.h>

int init_starpu_for_mode(ExecutionMode mode)
{
  struct starpu_conf conf;
  starpu_conf_init(&conf);

  switch (mode) {
  case ExecutionMode::StarpuCpu:
    conf.ncuda = 0;
    break;
  case ExecutionMode::StarpuGpu:
    conf.ncpus = 0;
    break;
  default:
    break;
  }

  const int ret = starpu_init(&conf);
  if (ret == 0 && mode != ExecutionMode::StarpuCpu)
    starpu_cublas_init();
  return ret;
}

void shutdown_starpu() { starpu_shutdown(); }

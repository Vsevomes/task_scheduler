#include "starpu_runtime.hpp"

#include <starpu.h>
#include <starpu_cublas.h>

int init_starpu_for_mode(ExecutionMode mode)
{
  struct starpu_conf conf;
  starpu_conf_init(&conf);

  const int ret = starpu_init(&conf);
  if (ret == 0 && mode == ExecutionMode::StarpuHybrid)
    starpu_cublas_init();
  return ret;
}

void shutdown_starpu() { starpu_shutdown(); }

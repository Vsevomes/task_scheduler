#include "starpu_runtime.hpp"

#include <starpu.h>

int init_starpu_for_mode(ExecutionMode mode)
{
  (void)mode;
  struct starpu_conf conf;
  starpu_conf_init(&conf);

  return starpu_init(&conf);
}

void shutdown_starpu() { starpu_shutdown(); }

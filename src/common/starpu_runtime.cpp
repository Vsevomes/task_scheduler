#include "starpu_runtime.hpp"

#include <cstdlib>
#include <starpu.h>

int init_starpu_for_mode(ExecutionMode mode)
{
  (void)mode;
  struct starpu_conf conf;
  starpu_conf_init(&conf);

  const char *sched = std::getenv("STARPU_SCHED");
  conf.sched_policy_name = (sched != nullptr && sched[0] != '\0') ? sched : "dmda";

  return starpu_init(&conf);
}

void shutdown_starpu() { starpu_shutdown(); }

#pragma once

#include "execution_mode.hpp"

int init_starpu_for_mode(ExecutionMode mode);
void shutdown_starpu();

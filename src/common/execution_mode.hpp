#pragma once

#include <cstdlib>
#include <string>

enum class ExecutionMode {
  NativeCpu,
  NativeGpu,
  StarpuHybrid,
};

inline ExecutionMode parse_execution_mode(const char *value)
{
  std::string mode(value);
  if (mode == "native_cpu")
    return ExecutionMode::NativeCpu;
  if (mode == "native_gpu")
    return ExecutionMode::NativeGpu;
  if (mode == "starpu_hybrid")
    return ExecutionMode::StarpuHybrid;
  std::abort();
}

inline const char *execution_mode_name(ExecutionMode mode)
{
  switch (mode) {
  case ExecutionMode::NativeCpu:
    return "native_cpu";
  case ExecutionMode::NativeGpu:
    return "native_gpu";
  case ExecutionMode::StarpuHybrid:
    return "starpu_hybrid";
  }
  return "unknown";
}

inline bool uses_starpu(ExecutionMode mode)
{
  return mode == ExecutionMode::StarpuHybrid;
}

inline ExecutionMode execution_mode_from_env()
{
  const char *value = std::getenv("EXECUTION_MODE");
  if (value == nullptr)
    return ExecutionMode::StarpuHybrid;
  return parse_execution_mode(value);
}

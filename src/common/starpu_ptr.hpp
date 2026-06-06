#pragma once

#include <cstdint>
#include <starpu.h>

template <typename T>
inline T *starpu_vector_ptr(void *buffer)
{
  return reinterpret_cast<T *>(static_cast<uintptr_t>(STARPU_VECTOR_GET_PTR(buffer)));
}

template <typename T>
inline T *starpu_matrix_ptr(void *buffer)
{
  return reinterpret_cast<T *>(static_cast<uintptr_t>(STARPU_MATRIX_GET_PTR(buffer)));
}

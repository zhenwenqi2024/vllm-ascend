// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <torch/library.h>

#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/hisparse_flash_attention/hisparse_flash_attention_torch_adpt.h"

#ifdef HISPARSE_DIRECT_STANDALONE
// A private build must not depend on the TLS definitions in the original
// extension. Build with hidden visibility so these names cannot interpose it.
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
#endif

// CMake's csrc/*.cpp source discovery includes this translation unit. The
// separate namespace cannot replace the existing _C_ascend SFA oracle.
TORCH_LIBRARY(hisparse_direct, ops) {
  ops.def("sparse_flash_attention(Tensor query, Tensor packed_cache, "
          "Tensor physical_indices, Tensor query_ends, Tensor valid_counts, "
          "Tensor query_rope, float scale_value, int cache_block_size, "
          "int cache_block_stride, int cache_row_capacity) -> Tensor");
  ops.impl("sparse_flash_attention", torch::kPrivateUse1,
           &vllm_ascend::hisparse_direct::sparse_flash_attention);
}

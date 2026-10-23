// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef HISPARSE_FLASH_ATTENTION_TORCH_ADPT_H
#define HISPARSE_FLASH_ATTENTION_TORCH_ADPT_H

#include <cmath>
#include <cstdint>
#include <limits>

namespace vllm_ascend::hisparse_direct {

inline at::Tensor sparse_flash_attention(
    const at::Tensor& query, const at::Tensor& packed_cache,
    const at::Tensor& physical_indices, const at::Tensor& query_ends,
    const at::Tensor& valid_counts, const at::Tensor& query_rope,
    double scale_value, int64_t cache_block_size, int64_t cache_block_stride,
    int64_t cache_row_capacity) {
  constexpr int64_t kLatent = 512;
  constexpr int64_t kRope = 64;
  constexpr int64_t kWidth = kLatent + kRope;
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  TORCH_CHECK(query.device().is_privateuseone(), "HiSparse direct SFA requires NPU inputs");
  TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && query.size(2) == kLatent,
              "query must be [queries, heads, 512]");
  const int64_t rows = query.size(0);
  const int64_t heads = query.size(1);
  TORCH_CHECK(heads >= 8 && heads <= 64 && heads % 8 == 0,
              "HiSparse direct SFA supports 8..64 query heads divisible by 8");
  TORCH_CHECK(rows <= 1048576, "HiSparse direct query count exceeds 1048576");
  TORCH_CHECK(query_rope.dim() == 3 && query_rope.size(0) == rows &&
                  query_rope.size(1) == heads && query_rope.size(2) == kRope,
              "query_rope must be [queries, heads, 64]");
  TORCH_CHECK(packed_cache.dim() == 1, "packed_cache must be a contiguous raw storage alias");
  for (const at::Tensor* tensor : {&query, &packed_cache, &query_rope}) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->scalar_type() == at::kBFloat16 &&
                    tensor->is_contiguous(),
                "All HiSparse direct payloads must be contiguous BF16 on the same NPU");
  }
  TORCH_CHECK(physical_indices.dim() == 3 && physical_indices.size(0) == rows &&
                  physical_indices.size(1) == 1,
              "physical_indices must be [queries, 1, top_k]");
  const int64_t topk = physical_indices.size(2);
  TORCH_CHECK(topk >= 128 && topk <= 32768 && topk % 128 == 0,
              "top_k must be divisible by 128 and at most 32768");
  for (const at::Tensor* tensor : {&physical_indices, &query_ends, &valid_counts}) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->scalar_type() == at::kInt &&
                    tensor->is_contiguous(),
                "HiSparse direct metadata must be contiguous int32 on the same NPU");
  }
  TORCH_CHECK(query_ends.dim() == 1 && query_ends.size(0) == rows &&
                  valid_counts.dim() == 1 && valid_counts.size(0) == rows,
              "query_ends and valid_counts must have one entry per query");
  TORCH_CHECK(std::isfinite(scale_value) && scale_value > 0,
              "HiSparse direct scale must be positive and finite");
  TORCH_CHECK(cache_block_size > 0 && cache_block_size <= 4095 &&
                  cache_block_stride >= cache_block_size * kWidth && cache_row_capacity > 0 &&
                  cache_row_capacity <= std::numeric_limits<int32_t>::max() &&
                  cache_row_capacity % cache_block_size == 0,
              "Invalid HiSparse packed page dimensions or stride");
  const int64_t blocks = cache_row_capacity / cache_block_size;
  const int64_t tail = cache_block_size * kWidth;
  TORCH_CHECK(blocks - 1 <= (kMax - tail) / cache_block_stride,
              "HiSparse packed storage span exceeds signed 64-bit addressing");
  const int64_t span = (blocks - 1) * cache_block_stride + tail;
  TORCH_CHECK(packed_cache.numel() >= span, "HiSparse packed alias is smaller than its addressable span");
  TORCH_CHECK(packed_cache.storage_offset() >= 0 &&
                  static_cast<uint64_t>(packed_cache.storage_offset()) + static_cast<uint64_t>(span) <=
                      packed_cache.storage().nbytes() / sizeof(at::BFloat16),
              "HiSparse packed alias exceeds its owning storage");

  auto output = at::empty(query.sizes(), query.options());
  auto softmax_max = at::empty({0}, query.options().dtype(at::kFloat));
  auto softmax_sum = at::empty({0}, query.options().dtype(at::kFloat));
  EXEC_NPU_CMD(aclnnHisparseFlashAttention, query, packed_cache, physical_indices,
               query_ends, valid_counts, query_rope, scale_value, cache_block_size,
               cache_block_stride, cache_row_capacity, output, softmax_max, softmax_sum);
  return output;
}

}  // namespace vllm_ascend::hisparse_direct
#endif

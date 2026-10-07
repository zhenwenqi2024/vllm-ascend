# SPDX-License-Identifier: Apache-2.0
"""Lightweight LIM kernel bounds and Copy-SFA cache-layout contract.

The LIM values mirror the BF16 and quantized kernel constants headers under
``csrc/attention/fused*_lightning_indexer_manage/op_kernel``. Source-checkout
tests enforce parity; runtime configuration does not load NPU operators or
depend on installed C++ source files.
"""

# LIM kernel constraints, independent of the draft checkpoint or serving layout.
LIM_TOPK = 2048
LIM_MAX_QUERY_ROWS = 14
LIM_MAX_HOT_TOKENS = 32640
LIM_CACHE_BLOCK_SIZE = 128

# Copy-SFA's serving layout reserves a two-block circular tail. Its period is
# also the hot-region alignment needed by the dense-to-circular transition.
# LIM itself only requires cache-block alignment, not this stronger alignment.
COPY_SFA_TAIL_BLOCKS = 2
COPY_SFA_TAIL_TOKENS = COPY_SFA_TAIL_BLOCKS * LIM_CACHE_BLOCK_SIZE

# HiSparse direct packed-KV attention

This independent Ascend 950 operator reuses the repository SparseFlashAttention
Cube and softmax implementation. It accepts BF16 Q `[Q,H,512]`, RoPE Q
`[Q,H,64]`, an unmodified contiguous 1D alias of packed cache storage, physical
row indices `[Q,1,K]`, query ends `[1,...,Q]`, and per-query valid counts.

The cache uses 576-element rows. A row's element address is
`(row / block_size) * block_stride + (row % block_size) * 576`, with 64-bit
arithmetic before every address product. `block_stride` is in elements. The
alias begins at the original layer view's data pointer and must cover every
page. It must not be made contiguous from the logical strided cache view.

Indices have a valid prefix and retain selection order, including duplicates.
The caller filters logical causal positions before mapping to physical rows;
the kernel never compares a physical row ID to a logical sequence position.
An empty selection produces zero output. Kernel bounds checks prevent invalid
physical IDs from reading outside the declared capacity. The first version
supports 8 through 64 query heads in multiples of 8, BF16, and RoPE width 64.

The new operator type, tiling data, ACLNN API, and Torch namespace coexist with
the original SFA. They do not replace installed CANN binaries or change their
ABI. Hardware numerical and graph validation is required before enabling it.

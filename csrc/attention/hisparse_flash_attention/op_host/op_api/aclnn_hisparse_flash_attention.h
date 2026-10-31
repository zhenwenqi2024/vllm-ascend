/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_hisparse_flash_attention.h
 * \brief
 */

#ifndef ACLNN_HISPARSE_FLASH_ATTENTION_H
#define ACLNN_HISPARSE_FLASH_ATTENTION_H
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnHisparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *packedCache,
    const aclTensor *physicalIndices, const aclTensor *queryEnds,
    const aclTensor *validCounts, const aclTensor *queryRope,
    double scaleValue, int64_t cacheBlockSize, int64_t cacheBlockStride,
    int64_t cacheRowCapacity, const aclTensor *attentionOut,
    const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnHisparseFlashAttention(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, const aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif

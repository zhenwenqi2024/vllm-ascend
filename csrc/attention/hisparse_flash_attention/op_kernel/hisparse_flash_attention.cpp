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
 * \file hisparse_flash_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "hisparse_flash_attention_template_tiling_key.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/hisparse_flash_attention_kernel_mla.h"

using namespace AscendC;

template<int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T,
         int TEMPLATE_MODE, int IS_SPLIT_G>
__global__ __aicore__ void hisparse_flash_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *packedCache,
    __gm__ uint8_t *physicalIndices, __gm__ uint8_t *queryEnds,
    __gm__ uint8_t *validCounts, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    using Cube = HiSparseBaseApi::HSFAMatmulService<bfloat16_t, bfloat16_t, float, bfloat16_t,
        FLASH_DECODE, PAGE_ATTENTION, static_cast<HSFA_LAYOUT>(LAYOUT_T),
        static_cast<HSFA_LAYOUT>(KV_LAYOUT_T), static_cast<HSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G>;
    using CubeDummy = HiSparseBaseApi::HSFAMatmulServiceDummy<bfloat16_t, bfloat16_t, float, bfloat16_t,
        FLASH_DECODE, PAGE_ATTENTION, static_cast<HSFA_LAYOUT>(LAYOUT_T),
        static_cast<HSFA_LAYOUT>(KV_LAYOUT_T), static_cast<HSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G>;
    using Vector = HiSparseBaseApi::HSFAVectorService<bfloat16_t, bfloat16_t, float, bfloat16_t,
        FLASH_DECODE, PAGE_ATTENTION, static_cast<HSFA_LAYOUT>(LAYOUT_T),
        static_cast<HSFA_LAYOUT>(KV_LAYOUT_T), static_cast<HSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G>;
    using VectorDummy = HiSparseBaseApi::HSFAVectorServiceDummy<bfloat16_t, bfloat16_t, float, bfloat16_t,
        FLASH_DECODE, PAGE_ATTENTION, static_cast<HSFA_LAYOUT>(LAYOUT_T),
        static_cast<HSFA_LAYOUT>(KV_LAYOUT_T), static_cast<HSFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G>;
    using CubeBlock = typename std::conditional<g_coreType == AscendC::AIC, Cube, CubeDummy>::type;
    using VecBlock = typename std::conditional<g_coreType == AscendC::AIC, VectorDummy, Vector>::type;
    HiSparseBaseApi::HisparseFlashAttentionKernelMla<CubeBlock, VecBlock> op;
    GET_TILING_DATA_WITH_STRUCT(HisparseFlashAttentionTilingDataMla, tilingData, tiling);
#if defined(__DAV_C310_CUBE__)
    const HisparseFlashAttentionTilingDataMla *localTiling = nullptr;
#else
    const HisparseFlashAttentionTilingDataMla *localTiling = &tilingData;
#endif
    op.Init(query, packedCache, packedCache, physicalIndices, queryEnds, validCounts,
            nullptr, queryRope, nullptr, attentionOut, softmaxMax, softmaxSum,
            user, localTiling, tiling, &tPipe);
    op.Process();
}
#endif

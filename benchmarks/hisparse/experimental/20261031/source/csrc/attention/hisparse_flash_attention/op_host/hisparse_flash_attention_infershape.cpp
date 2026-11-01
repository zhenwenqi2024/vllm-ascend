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
 * \file hisparse_flash_attention_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "err/ops_err.h"

namespace ops {
static ge::graphStatus InferShapeHisparseFlashAttention(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("HisparseFlashAttention", "Null shape context"),
                return ge::GRAPH_FAILED);
    const auto *query = context->GetInputShape(0);
    auto *out = context->GetOutputShape(0);
    auto *max = context->GetOutputShape(1);
    auto *sum = context->GetOutputShape(2);
    OP_CHECK_IF(query == nullptr || out == nullptr || max == nullptr || sum == nullptr,
                OP_LOGE("HisparseFlashAttention", "Missing shape"), return ge::GRAPH_FAILED);
    *out = *query;
    max->SetDimNum(1);
    max->SetDim(0, 0);
    sum->SetDimNum(1);
    sum->SetDim(0, 0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeHisparseFlashAttention(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("HisparseFlashAttention", "Null dtype context"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HisparseFlashAttention)
    .InferShape(InferShapeHisparseFlashAttention)
    .InferDataType(InferDataTypeHisparseFlashAttention);
} // namespace ops

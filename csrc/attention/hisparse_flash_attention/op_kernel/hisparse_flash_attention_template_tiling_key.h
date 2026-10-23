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
 * \file hisparse_flash_attention_template_tiling_key.h
 * \brief
 */

#ifndef HISPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H
#define HISPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define HSFA_LAYOUT_BSND 0
#define HSFA_LAYOUT_TND 1
#define HSFA_LAYOUT_PA_BSND 2

#define ASCENDC_TPL_4_BW 4

#define C_TEMPLATE 0
#define V_TEMPLATE 1

// 模板参数支持的范围定义
ASCENDC_TPL_ARGS_DECL(HisparseFlashAttention, // 算子OpType
ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1),
ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 0, 1),
ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, HSFA_LAYOUT_BSND, HSFA_LAYOUT_TND),
ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, HSFA_LAYOUT_BSND, HSFA_LAYOUT_TND,
                      HSFA_LAYOUT_PA_BSND),
ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, C_TEMPLATE, V_TEMPLATE),
ASCENDC_TPL_BOOL_DECL(IS_SPLIT_G, 0, 1),
);

// Only the registered direct packed-cache contract is compiled.
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0),
        ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1),
        ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, HSFA_LAYOUT_TND),
        ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, HSFA_LAYOUT_PA_BSND),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, V_TEMPLATE),
        ASCENDC_TPL_BOOL_SEL(IS_SPLIT_G, 0),
    ),
);
#endif // HISPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H

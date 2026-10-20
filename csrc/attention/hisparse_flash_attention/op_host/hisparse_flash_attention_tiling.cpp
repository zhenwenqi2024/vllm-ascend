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
 * \file hisparse_flash_attention_tiling.cpp
 * \brief
 */

#include <cmath>
#include <climits>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "../op_kernel/hisparse_flash_attention_template_tiling_key.h"
#include "hisparse_flash_attention_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
ge::graphStatus HSFAMlaTiling::SetBlockDim(uint32_t blockDim) const
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HSFAMlaTiling::SetTilingKey(uint64_t tilingKey) const
{
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);     // 1: batchmode模式
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HSFAMlaTiling::SetWorkspaceSize(uint64_t workspaceSize) const
{
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HSFAMlaTiling::SetTilingData(TilingDef &tilingData) const
{
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HSFAMlaTiling::GetPlatformInfo()
{
    OP_CHECK_IF(sfaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void HSFAMlaTiling::GenTilingKey()
{
    uint32_t inputQType = static_cast<uint32_t>(sfaInfo_->inputQType);
    uint32_t inputKvType = static_cast<uint32_t>(sfaInfo_->inputKvType);
    uint32_t outputType = static_cast<uint32_t>(sfaInfo_->outputType);
    uint32_t layoutQuery = static_cast<uint32_t>(sfaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(sfaInfo_->kvLayout);
    uint32_t pageAttention = 0U;
    if (sfaInfo_->kvLayout == HSFALayout::PA_BSND) {
        pageAttention = 1U;
    }

    tilingKey_ = GET_TPL_TILING_KEY(0U, pageAttention, layoutQuery, layoutKV,
        perfMode_ == HSFAPerfMode::V_TEMPLATE_MODE, static_cast<uint32_t>(sfaInfo_->gSize > 64)); // N1 > 128时核间切G

    OP_LOGI(sfaInfo_->opName, "HSFA tilingKey_: %lu.", tilingKey_);
}

void HSFAMlaTiling::ZeroTensorProcess()  const
{
    if (sfaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        sfaInfo_->s2Size = 1024;
    }
}

void HSFAMlaTiling::InitParams()
{
    if (sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize <= 4) { // 4:当前支持范围
        perfMode_ = HSFAPerfMode::V_TEMPLATE_MODE;
    } else {
        perfMode_ = HSFAPerfMode::C_TEMPLATE_MODE;
    }

    coreNum_ = aicNum_;

    headDimAlign_ = Align(sfaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    ZeroTensorProcess();
}

void HSFAMlaTiling::CalcUbBmm()
{
    uint32_t cubeMSize = sfaInfo_->gSize * sfaInfo_->s1Size;
    uint32_t maxMSize = mBaseSize_;
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存

    qPreSizeMla_ = sfaInfo_->gSize * (headDimAlign_ + 64U) * sfaInfo_->s1Size;
}

void HSFAMlaTiling::CheckUbSpace()
{
    CalcUbBmm();
}

void HSFAMlaTiling::CalcInnerSize(uint32_t s2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && sfaInfo_->qLayout != HSFALayout::TND) {
        if (s2Size == 256) {   // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128; // 128:sInnerSize_值为s2Size的一半，均分为2份进行计算，
        } else if (s2Size > 256 && s2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2; // 2:减半
        }
    }

    sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > s2Size) {
        sInnerSize_ = s2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CheckUbSpace();
}

void HSFAMlaTiling::SplitBalanced()
{
    CalcInnerSize(sfaInfo_->s2Size);

    HiSparseInnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = sfaInfo_->gSize;
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void HSFAMlaTiling::Split()
{
    SplitBalanced();
}

void HSFAMlaTiling::FillTilingBaseParamsMla()
{
    tilingData_.baseParams.set_batchSize(sfaInfo_->bSize);
    tilingData_.baseParams.set_seqSize(sfaInfo_->s2Size);
    tilingData_.baseParams.set_qSeqSize(sfaInfo_->s1Size);
    tilingData_.baseParams.set_blockSize(sfaInfo_->blockSize);
    tilingData_.baseParams.set_cacheBlockStride(sfaInfo_->cacheBlockStride);
    tilingData_.baseParams.set_cacheRowCapacity(sfaInfo_->cacheRowCapacity);
    tilingData_.baseParams.set_maxBlockNumPerBatch(sfaInfo_->maxBlockNumPerBatch);
    tilingData_.baseParams.set_scaleValue(sfaInfo_->scaleValue);
    tilingData_.baseParams.set_nNumOfQInOneGroup(sfaInfo_->n1Size / sfaInfo_->n2Size);
    tilingData_.baseParams.set_actualLenDimsQ(sfaInfo_->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(sfaInfo_->actualLenDimsKV);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(sfaInfo_->outLayout));
    tilingData_.baseParams.set_sparseMode(sfaInfo_->sparseMode);
    tilingData_.baseParams.set_preTokens(sfaInfo_->preTokens);
    tilingData_.baseParams.set_nextTokens(sfaInfo_->nextTokens);
    tilingData_.baseParams.set_sparseBlockSize(sfaInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(sfaInfo_->sparseBlockCount);
    tilingData_.baseParams.set_attentionMode(sfaInfo_->attentionMode);
    tilingData_.baseParams.set_returnSoftmaxLse(sfaInfo_->returnSoftmaxLse);
    tilingData_.baseParams.set_isActualLenDimsNull(sfaInfo_->actualQSeqLenFlag ? 0U : 1U);
    tilingData_.baseParams.set_isActualLenDimsKVNull(sfaInfo_->actualSeqLenFlag ? 0U : 1U);
    tilingData_.baseParams.set_ropeHeadDim(sfaInfo_->ropeHeadDim);
}

// for flash decode
void HSFAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);

    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2 * sfaInfo_->n2Size * mBaseSize_ * headDimAlign_);   // 2:每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * sfaInfo_->n2Size * mBaseSize_ *  // 2:每个核可能有头规约和尾规约，一共两份规约信息;sum + max
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void HSFAMlaTiling::FillTilingSingleCoreParamsMla()
{
    tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void HSFAMlaTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void HSFAMlaTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t HSFAMlaTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum)  const
{
    return actCoreNum * 2 * sfaInfo_->n2Size * mBaseSize_; // 2:每个核可能有头规约和尾规约，一共两份规约信息
}

void HSFAMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum); //balanceModeFlag_ ? CalcBalanceFDParamNums(actCoreNum) : CalcUnbalanceFDParamNums();
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize = 2 * FDParamNums * (BYTE_BLOCK / sfaInfo_->blockTypeSize);  // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += (accumOutSize + logSumExpSize) * sfaInfo_->blockTypeSize;
    }
}

void HSFAMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    NormalCalcFDWorkSpace(actCoreNum);
}

void HSFAMlaTiling::GetWorkspaceSize()
{
    uint32_t actCoreNum = coreNum_;
    if (sfaInfo_->isA5) {
        workspaceSize_ = libapiSize_;
        constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
        constexpr uint32_t S2_BASE_SIZE = 128;            // S2轴基本块大小
        constexpr uint32_t D_SIZE = 576;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
        uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
        if (sfaInfo_->gSize > 64) { // N1大于64时切G，相邻两个cube核处理同一个s2Base
            aicNum = aicNum >> 1;
        }
        workspaceSize_ += (S2_BASE_SIZE * D_SIZE * GetTypeSize(sfaInfo_->inputQType) \
            * TRIPLE_BUFFER_NUM * aicNum);
    } else {
        uint32_t mmResElemSize = 4;         // 4:fp32
        uint32_t vec1ResElemSize = 2;       // 2:fp16/bf16
        uint32_t bmm2ResElemSize = 4;       // 4:fp32
        uint32_t qPreProcResElemSize = 0;   // 普通场景不涉及Q预处理
        uint32_t nUpdateElemSize = 4;   // 4:int32
        uint32_t softmaxSumElemSize = 4;   // 4:int32
        float kvDtypeRatio = 1.0;

        workspaceSize_ = libapiSize_;
        uint32_t preLoadNum = 1;
        preLoadNum = PRE_LOAD_NUM;

        workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
        workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(mmResUbSize_ * \
            actCoreNum * vec1ResElemSize) * kvDtypeRatio);
        workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
        workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(qPreSizeMla_ * \
            actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * nUpdateElemSize;
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
        // topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值
        // kvMergeGm_ is sized for the maximal rope=64 form so its stride
        // matches the kernel-side 512*576 layout. With rope=0 the rope segment
        // stays unused; the stride is unchanged so host and kernel agree.
        //              bufNum  s2Base   D   dRope  sizeOf(half)
        // 4:bufNum  512:s2Base  512:D  64:dRope  2:sizeOf(half)
        workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum;
        // 缓存有效mte2 size的长度 份数  512B对齐的长度  sizeof(int32_t)   aiv核数
        workspaceSize_ += 4 * 128 * 4 * (2 * actCoreNum); // 4:缓存有效mte2 size的长度 128:份数  4:512B对齐的长度  2:aiv核数
    }

    CalcFDWorkSpace(actCoreNum);
}

void HSFAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(sfaInfo_->opName, "HSFA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

ge::graphStatus HSFAMlaTiling::DoOpTiling(HSFATilingInfo *sfaInfo)
{
    sfaInfo_ = sfaInfo;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    Split();
    FillTiling();
    CalcBlockDim();
    GetWorkspaceSize();
    GenTilingKey();

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}


namespace {
constexpr int64_t LATENT_DIM = 512;
constexpr int64_t ROPE_DIM = 64;
constexpr int64_t PACKED_DIM = LATENT_DIM + ROPE_DIM;
constexpr int64_t MAX_QUERIES = 1048576;
constexpr int64_t MAX_SELECTIONS = 32768;

ge::graphStatus ParseHiSparseInputs(gert::TilingContext *context, HSFATilingInfo &info)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("HisparseFlashAttention", "Tiling context is null"),
                return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < 6; ++i) {
        OP_CHECK_IF(context->GetInputShape(i) == nullptr || context->GetInputDesc(i) == nullptr,
                    OP_LOGE(context->GetNodeName(), "Missing required input %u", i), return ge::GRAPH_FAILED);
        const auto expected = (i == 2 || i == 3 || i == 4) ? ge::DT_INT32 : ge::DT_BF16;
        OP_CHECK_IF(context->GetInputDesc(i)->GetDataType() != expected,
                    OP_LOGE(context->GetNodeName(), "Input %u has unsupported dtype", i), return ge::GRAPH_FAILED);
    }
    const auto &q = context->GetInputShape(0)->GetStorageShape();
    const auto &cache = context->GetInputShape(1)->GetStorageShape();
    const auto &indices = context->GetInputShape(2)->GetStorageShape();
    const auto &ends = context->GetInputShape(3)->GetStorageShape();
    const auto &valid = context->GetInputShape(4)->GetStorageShape();
    const auto &rope = context->GetInputShape(5)->GetStorageShape();
    OP_CHECK_IF(q.GetDimNum() != 3 || cache.GetDimNum() != 1 || indices.GetDimNum() != 3 ||
                    ends.GetDimNum() != 1 || valid.GetDimNum() != 1 || rope.GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(), "Expected Q/rope [Q,H,D], cache [span], indices [Q,1,K], lengths [Q]"),
                return ge::GRAPH_FAILED);
    const int64_t rows = q.GetDim(0);
    const int64_t heads = q.GetDim(1);
    const int64_t topk = indices.GetDim(2);
    OP_CHECK_IF(rows <= 0 || rows > MAX_QUERIES || heads < 8 || heads > 64 || heads % 8 != 0 ||
                    q.GetDim(2) != LATENT_DIM || rope.GetDim(0) != rows || rope.GetDim(1) != heads ||
                    rope.GetDim(2) != ROPE_DIM || indices.GetDim(0) != rows || indices.GetDim(1) != 1 ||
                    topk < 128 || topk > MAX_SELECTIONS || topk % 128 != 0 || ends.GetDim(0) != rows || valid.GetDim(0) != rows,
                OP_LOGE(context->GetNodeName(), "Unsupported shape for direct BF16 MLA attention"), return ge::GRAPH_FAILED);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "Missing attributes"), return ge::GRAPH_FAILED);
    const auto *scale = attrs->GetAttrPointer<float>(0);
    const auto *block = attrs->GetAttrPointer<int64_t>(1);
    const auto *stride = attrs->GetAttrPointer<int64_t>(2);
    const auto *capacity = attrs->GetAttrPointer<int64_t>(3);
    OP_CHECK_IF(scale == nullptr || block == nullptr || stride == nullptr || capacity == nullptr,
                OP_LOGE(context->GetNodeName(), "Missing direct-cache attributes"), return ge::GRAPH_FAILED);
    // The inherited shared scalar uses 12 bits for the block size. Physical IDs are int32.
    OP_CHECK_IF(!std::isfinite(*scale) || *scale <= 0 || *block <= 0 || *block > 4095 || *capacity <= 0 ||
                    *capacity > INT32_MAX || *capacity % *block != 0 || *stride < *block * PACKED_DIM,
                OP_LOGE(context->GetNodeName(), "Invalid cache geometry or scale"), return ge::GRAPH_FAILED);
    const uint64_t pages = static_cast<uint64_t>(*capacity / *block);
    const uint64_t pageElements = static_cast<uint64_t>(*block) * PACKED_DIM;
    OP_CHECK_IF(pages > 1 && static_cast<uint64_t>(*stride) > (INT64_MAX - pageElements) / (pages - 1),
                OP_LOGE(context->GetNodeName(), "Cache span exceeds signed 64-bit indexing"), return ge::GRAPH_FAILED);
    const uint64_t span = (pages - 1) * static_cast<uint64_t>(*stride) + pageElements;
    OP_CHECK_IF(cache.GetDim(0) < 0 || static_cast<uint64_t>(cache.GetDim(0)) < span,
                OP_LOGE(context->GetNodeName(), "Packed cache does not cover its declared page span"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetPlatformInfo() == nullptr,
                OP_LOGE(context->GetNodeName(), "Missing platform information"), return ge::GRAPH_FAILED);
    info.opName = context->GetNodeName();
    info.platformInfo = context->GetPlatformInfo();
    info.isA5 = true; // Only ascend950 is registered by this independent operator.
    info.bSize = rows;
    info.n1Size = heads;
    info.n2Size = 1;
    info.gSize = heads;
    info.s1Size = 1;
    info.s2Size = topk;
    info.qTSize = rows;
    info.qkHeadDim = LATENT_DIM;
    info.vHeadDim = LATENT_DIM;
    info.ropeHeadDim = ROPE_DIM;
    info.scaleValue = *scale;
    info.sparseBlockSize = 1;
    info.sparseBlockCount = topk;
    info.pageAttentionFlag = true;
    info.blockSize = *block;
    info.cacheBlockStride = *stride;
    info.cacheRowCapacity = *capacity;
    info.blockTypeSize = 4;
    info.maxBlockNumPerBatch = 0; // No block table: indices are physical cache rows.
    info.actualLenDimsQ = rows;
    info.actualLenDimsKV = rows;
    info.actualQSeqLenFlag = true;
    info.actualSeqLenFlag = true;
    info.sparseMode = 0; // Logical causal filtering has already happened before residency resolution.
    info.inputQType = ge::DT_BF16;
    info.inputKvType = ge::DT_BF16;
    info.outputType = ge::DT_BF16;
    info.inputQRopeType = ge::DT_BF16;
    info.inputKRopeType = ge::DT_BF16;
    info.kvStorageMode = HiSparseKvStorageMode::PAGE_ATTENTION;
    info.qLayout = HSFALayout::TND;
    info.topkLayout = HSFALayout::TND;
    info.outLayout = HSFALayout::TND;
    info.kvLayout = HSFALayout::PA_BSND;
    return ge::GRAPH_SUCCESS;
}
} // namespace

uint32_t HSFAMlaTiling::GetTypeSize(ge::DataType dtype) const
{
    (void)dtype;
    return NUM_BYTES_BF16;
}

ge::graphStatus TilingHisparseFlashAttention(gert::TilingContext *context)
{
    HSFATilingInfo info;
    if (ParseHiSparseInputs(context, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    HSFAMlaTiling tiling(context);
    return tiling.DoOpTiling(&info);
}

struct HisparseFlashAttentionCompileInfo { int64_t core_num; };
ge::graphStatus TilingPrepareForHisparseFlashAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HisparseFlashAttention)
    .Tiling(TilingHisparseFlashAttention)
    .TilingParse<HisparseFlashAttentionCompileInfo>(TilingPrepareForHisparseFlashAttention);
} // namespace optiling

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
 * \file hisparse_flash_attention_tiling.h
 * \brief
 */
#ifndef HISPARSE_FLASH_ATTENTION_TILING_H
#define HISPARSE_FLASH_ATTENTION_TILING_H

#include <sstream>
#include <vector>
#include <cstdint>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/soc_spec.h"

namespace optiling {
// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t SPARSE_INDICES_INPUT_INDEX = 3;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 4;
constexpr uint32_t ACT_SEQ_LEN_Q_INPUT_INDEX = 5;
constexpr uint32_t ACT_SEQ_LEN_KV_INPUT_INDEX = 6;
constexpr uint32_t QUERY_ROPE_INPUT_INDEX = 7;
constexpr uint32_t KEY_ROPE_INPUT_INDEX = 8;
// Outputs Index
constexpr uint32_t OUTPUT_INDEX = 0;
constexpr uint32_t SOFTMAXMAX_INDEX = 1;
constexpr uint32_t SOFTMAXSUM_INDEX = 2;

// Attributes Index
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 0;
constexpr uint32_t SPARSE_BLOCK_SIZE_ATTR_INDEX = 1;
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 2;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 3;
constexpr uint32_t SPARSE_MODE_ATTR_INDEX = 4;
constexpr uint32_t PRE_TOKENS_ATTR_INDEX = 5;
constexpr uint32_t NEXT_TOKENS_ATTR_INDEX = 6;
constexpr uint32_t ATTENTION_MODE_ATTR_INDEX = 7;
constexpr uint32_t RETURN_SOFTMAX_LSE_ATTR_INDEX = 8;
// Dim Num
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
// 常量
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;
const uint32_t HSFA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

// ------------------公共定义--------------------------
enum class HSFALayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2,
    BNSG = 3,
    NTG = 4
};

struct HSFATilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t T = 1;
    int64_t G = 1;
    // PA
    int64_t Bs = 1;
    int64_t Bn = 1;
};

enum class HiSparseKvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    PAGE_ATTENTION = 1
};

enum class HSFAPerfMode : uint32_t {
    C_TEMPLATE_MODE = 0,
    V_TEMPLATE_MODE
};

enum class HSFAAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3,  // sparse_indices的K和key的D枚举值相同，表达相同位置, 最后一维
    T = 5,
    Bn = 6, // block number
    Bs = 7, // block size
    G = 8,
};

struct HSFARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct HSFAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// -----------算子Tiling入参结构体定义---------------
struct HSFAParaInfo {
    HSFARequiredParaInfo query = {nullptr, nullptr};
    HSFARequiredParaInfo key = {nullptr, nullptr};
    HSFARequiredParaInfo value = {nullptr, nullptr};
    HSFARequiredParaInfo sparseIndices = {nullptr, nullptr};
    HSFAOptionalParaInfo blockTable = {nullptr, nullptr};
    HSFAOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    HSFAOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    HSFAOptionalParaInfo queryRope = {nullptr, nullptr};
    HSFAOptionalParaInfo keyRope = {nullptr, nullptr};
    HSFARequiredParaInfo attenOut = {nullptr, nullptr};
    HSFARequiredParaInfo softmaxMax = {nullptr, nullptr};
    HSFARequiredParaInfo softmaxSum = {nullptr, nullptr};

    const char *layoutQuery = nullptr;
    const char *layoutKV = nullptr;
    const int64_t *sparseBlockSize = nullptr;
    const float *scaleValue = nullptr;
    const int64_t *sparseMode = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
    const int64_t *attentionMode = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

struct HiSparseInnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(HisparseFlashAttentionBaseParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(int64_t, blockSize)
TILING_DATA_FIELD_DEF(uint64_t, cacheBlockStride)
TILING_DATA_FIELD_DEF(uint64_t, cacheRowCapacity)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsQ)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsKV)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
TILING_DATA_FIELD_DEF(uint32_t, attentionMode)
TILING_DATA_FIELD_DEF(uint32_t, returnSoftmaxLse)
TILING_DATA_FIELD_DEF(int64_t, sparseBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockCount)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsNull)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsKVNull)
TILING_DATA_FIELD_DEF(uint32_t, ropeHeadDim)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(HisparseFlashAttentionBaseParamsMlaOp, HisparseFlashAttentionBaseParamsMla)

BEGIN_TILING_DATA_DEF(HisparseFlashAttentionSingleCoreParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(HisparseFlashAttentionSingleCoreParamsMlaOp, HisparseFlashAttentionSingleCoreParamsMla)

BEGIN_TILING_DATA_DEF(HisparseFlashAttentionSingleCoreTensorSizeMla)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(HisparseFlashAttentionSingleCoreTensorSizeMlaOp, HisparseFlashAttentionSingleCoreTensorSizeMla)

BEGIN_TILING_DATA_DEF(HisparseFlashAttentionSplitKVParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, s2)             // S2切分份数
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)   // FD workspace
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)  // FD workspace
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(HisparseFlashAttentionSplitKVParamsMlaOp, HisparseFlashAttentionSplitKVParamsMla)

// 内切基本块参数
BEGIN_TILING_DATA_DEF(HisparseFlashAttentionInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(HisparseFlashAttentionInnerSplitParamsOp, HisparseFlashAttentionInnerSplitParams)

BEGIN_TILING_DATA_DEF(HisparseFlashAttentionTilingDataMla)
TILING_DATA_FIELD_DEF_STRUCT(HisparseFlashAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(HisparseFlashAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(HisparseFlashAttentionSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(HisparseFlashAttentionSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(HisparseFlashAttentionInnerSplitParams, innerSplitParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(HisparseFlashAttention, HisparseFlashAttentionTilingDataMla)

template <typename T> inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T>
std::string HSFAShape2String(const T &shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static std::string GetShapeStr(gert::Shape shape);
static std::string HSFADataTypeToSerialString(ge::DataType type);
std::string HSFATensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor);
std::string HSFADebugTilingContext(const gert::TilingContext *context);
std::string HSFALayoutToSerialString(HSFALayout layout);

// -----------算子Tiling入参信息类---------------
struct HSFATilingInfo {
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    HSFAParaInfo opParamInfo;

    // Base Param
    NpuArch npuArch = NpuArch::DAV_2201;
    bool isA5 = false;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t gSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t qTSize = 0; // 仅TND时生效
    uint32_t kvTSize = 0; // 仅TND时生效
    float scaleValue = 0;
    uint32_t innerPrecise = 0;
    uint32_t l2CacheOffFlag = 0;
    int64_t sparseBlockSize = 0;
    int64_t sparseBlockCount = 0;

    bool pageAttentionFlag = false;
    int64_t blockSize = 0;
    uint64_t cacheBlockStride = 0;
    uint64_t cacheRowCapacity = 0;
    uint32_t blockTypeSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    uint32_t actualLenDimsQ = 0;
    uint32_t maxActualseq = 0;

    bool actualQSeqLenFlag = false;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDimsKV = 0;
    std::vector<int64_t> kvListSeqLens {};

    uint32_t sparseMode = 0;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    uint32_t attentionMode = 2;
    bool returnSoftmaxLse = false;

    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    HiSparseKvStorageMode kvStorageMode = HiSparseKvStorageMode::BATCH_CONTINUOUS;

    HSFALayout qLayout = HSFALayout::BSND;
    HSFALayout topkLayout = HSFALayout::BSND;
    HSFALayout outLayout = HSFALayout::BSND;
    HSFALayout kvLayout = HSFALayout::BSND;
    HSFALayout softmaxMaxLayout = HSFALayout::BNSG;
    HSFALayout softmaxSumLayout = HSFALayout::BNSG;

    ge::DataType inputQRopeType = ge::DT_FLOAT16;
    ge::DataType inputKRopeType = ge::DT_FLOAT16;

    uint64_t l2CacheSize = 0;
};

// ---------------算子Tiling类---------------
class HSFAMlaTiling {
public:
    explicit HSFAMlaTiling(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus DoOpTiling(HSFATilingInfo *sfaInfo);

private:
    ge::graphStatus SetBlockDim(uint32_t blockDim) const;
    ge::graphStatus SetTilingKey(uint64_t tilingKey) const;
    ge::graphStatus SetWorkspaceSize(uint64_t workspaceSize) const;
    ge::graphStatus SetTilingData(TilingDef &tilingData) const;
    gert::TilingContext *context_ = nullptr;
    ge::graphStatus GetPlatformInfo();
    void GenTilingKey();
    bool DealSameSeqEachBatch();

    void ZeroTensorProcess() const;
    void InitParams();

    void Split();
    bool IsBalanceSplitCore();

    void SplitBalanced();
    void CalcInnerSize(uint32_t s2Size);

    bool IsFlashDecode(uint32_t coreNum);

    void FillTilingBaseParamsMla();
    void FillTilingSplitKVMla();

    void FillTilingSingleCoreParamsMla();
    void FillTilingSingleCoreTensorSizeMla();
    void FillTiling();

    void CalcUbBmm();
    void CheckUbSpace();
    void NormalCalcFDWorkSpace(const uint32_t actCoreNum);
    void CalcFDWorkSpace(const uint32_t actCoreNum);
    void GetWorkspaceSize();

    uint32_t CalcBalanceFDParamNums(const uint32_t actCoreNum) const;

    void CalcBlockDim();

    uint32_t GetTypeSize(ge::DataType dtype) const;

    bool balanceModeFlag_ = false;
    bool splitKVFlag_ = false;

    uint32_t coreNum_ = 0;
    HSFAPerfMode perfMode_ = HSFAPerfMode::V_TEMPLATE_MODE;
    uint32_t kvSplitPart_ = 1;
    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    size_t qPreSizeMla_= 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t kvSplit_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;

    HisparseFlashAttentionTilingDataMla tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 128;
    uint32_t mFdBaseSize_ = 8;

    HSFATilingInfo *sfaInfo_ = nullptr;
};

} // namespace optiling
#endif // HISPARSE_FLASH_ATTENTION_TILING_H

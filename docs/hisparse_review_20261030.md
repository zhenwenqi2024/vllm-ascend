# HiSparse 2026-10-30 集成代码审阅

这是一组供人工审阅的配套草稿，尚不是可发布版本。
vLLM 公共侧和 Ascend 插件必须按本次配套快照使用。
历史 10 月 23 日说明对应旧提交；不能用于证明当前代码已经通过全部检查。

## 本轮按 Ascend 后端收敛

本次新增实现位于 `vllm_ascend/distributed/kv_transfer/hisparse/runtime_adapter.py`。
公共 `runtime.py` 和 connector `worker.py` 恢复为合入 PR62592 后的原文，未新增公共侧源文件改动。
后端为每个 NPU runtime 绑定自身的 resolver/stage/invalidate 方法；Prefill plan 和 Host mirror 的两个调用点使用校验源码版本的 NPU 专用包装，其他设备走原函数。
这些是暂时的插件适配点，不应误称为上游已经提供的完整公共 hook；上游提供等价 hook 后移除。

本轮新适配器只通过 15 项隔离 CPU 合同检查，使用实际生产函数定义和真实 CPU torch；未证明 NPU 图或模型正确性，也未部署。
下面性能与 NPU 记录均来自先前 API12/API15 实测，并不自动资格化本轮重构后的 Git head。

## 公共侧来源与历史变化

- 官方基线：`43111fd1f0c3135ff671e25e2e85238aa05743d0`。
- 已测集成以 `17b5c9a97198275745961638a35e78aa1ad907ff` 为来源，仅合入 PR62592，不含 PR63801。
- PR62592 把 Host 注册、DMA、设备算子、stream/event、缓存视图和 Resident gather 放到平台 backend 后面。
- 历史 API15 曾在公共 `runtime.py` 移植 Prefill `index_select`、PR60103 padding 和 grouped invalidate；本轮已把这些实现移到 Ascend 适配层。
- 公共 connector 不再携带本轮批量失效改动；Ascend 的 NPU 包装负责该操作。Host 写回及 `finish_forward` 等待策略仍保留。
- 计数回拷和 MTP 提交时序未修改；没有重写调度器、LRU 或 KV 分配布局。

## Ascend 侧

| 模块 | 改动与用途 |
| --- | --- |
| `attention/sfa_hisparse.py`、`ops/hisparse_c8*.py` | attention/Indexer C8 opaque payload、scale、物理索引及图所有者适配。 |
| `distributed/kv_transfer/hisparse/backend*.py` | NPU backend、Host 注册及持有引用、完成事件与资源释放。 |
| `distributed/kv_transfer/hisparse/queued_dma.py` | 通过 PTA guard、stream 和任务队列提交实际 DMA；不使用被拒绝的 mapped-writeback 候选。 |
| `ops/hisparse*.py`、`ops/triton/hisparse*.py` | Resident/Hot 地址解析、gather、去重、失效和元数据维护；全 Resident 路径回搬数为零并跳过 Hot LRU。 |
| `ops/hisparse_ascendc.py`、`csrc/hisparse_runtime/` | A5 C8 request-union resolver、图实例参数所有权、PTA 队列桥接，以及可审阅的 native 源码。 |
| `distributed/kv_transfer/kv_p2p/hisparse_pd/` | SOURCE/INDEXER ticket、lease、Device URMA 暂存、D 导入、ACK 和关闭顺序。 |
| `patch/worker/patch_hisparse_*.py` | invalidate 的 NPU 替代、图 padding 所有权及公共 stream/event 的版本受限桥接。 |
| V2 attention/sampling/MTP 接口 | 适配冻结上游 API；其他模型路径未因此得到新的硬件支持声明。 |

普通 CANN 批量 DMA 曾出现卡住，因此当前 PTA 队列仍逐项提交 `aclrtMemcpyAsync`。
Host→Host URMA 问题未修复：P/D 仍使用有界 Device staging，不能宣称直接 Host→Host 已可用。
旧 BF16 自定义 attention 源码保留，但当前 C8 使用 C8 attention 路径；二者不能混为同一算子。

## 实际结果及范围

完整 GLM-5.2、78+1 层、W4A8C8、TP8、MTP3、V2，单 D 的 C8 / 16384 输入 / 8192 输出：

| 版本 | r1 pure-decode TPS | r2 pure-decode TPS | 均值 |
| --- | ---: | ---: | ---: |
| OFF | 570.1623 | 565.0282 | 567.5952 |
| API12 ON | 511.1050 | 505.6597 | 508.3823 |
| API15 / PR60103 移植 | 508.2738 | 512.4735 | 510.3737 |

API12 ON 比 OFF 低 10.43%；PR60103 移植版比 OFF 低 10.08%，相对 API12 仅高 0.39%，不构成已证实的有效提速。
primer 不计分，各臂正式 N2；正式计时关闭 profiler 并已移除观察器。
API12 的 16 个正式请求 token-ID/text 与 OFF 配对一致；API15 的两轮输出和完整 MTP 计数对应 OFF 两轮，但轮次顺序交换，不能暗中重标。
这些是单 D 结果，不是双机 P/D 性能，也不是数据集精度。
零 miss 是等待异步计数稳定后的报告结果，不是穷尽的设备 DMA 事件证明。

native primitive 已完成 3 个图 owner、每 owner 18 次变化回放；generic padding 也完成独立 NPU 检查。
完整模型已执行自然短请求和 N2；这些证据对应原冻结 runtime。
重新打包后的 build 入口、最终 Git head 的独立 native 重建、标准 CI、完整双机 P/D 对照和最终数据集精度仍待完成。
旧 CPU suite 的历史失败不能视为已解决；本次检查结果另见 PR 正文。

## 未采用及待验证

- mapped queued Host writeback：均值 474.5891 TPS，比 API12 ON 低 6.65%，部分输出分叉尚未定因，拒绝采用。
- `eager_host_mirror=False`：上游已有配置；模型实验被外部占用挡在首次门禁，尚无新结果。
- Hot 预留及共享块对齐带来的 Resident 容量损失已有分配与运行证据；尚未改变分配策略。
- profiling 仅部分 rank 有有效包，缺少同 shape 完整 OFF 对照，不能把累计 API/各流耗时相加解释 wall time。

## Native 源码与复现边界

`csrc/hisparse_runtime/build.py --cann-root CANN_ROOT --output NEW_BUILD_DIR`
提供独立的 A5 CANN 9.2 / torch 2.10 / torch-npu 2.10 构建入口。
它不初始化 NPU，不安装到服务环境，也不包含二进制、权重、凭据或机器启动配置。
产物需要人工审阅后放到对应 `vllm_ascend/ops/`；不得与旧 `torch_binding.so` 混装。
`benchmarks/hisparse/qualification/` 保留已使用的 primitive harness；调用方仍需提供完整 baseline/candidate 目录布局和独占设备门禁。
构建脚本本身仍待最终配套头实机重建验证。

AI assistance: OpenAI Codex assisted with implementation, investigation, tests and this review preparation.
Human line-by-line review and personally executed final validation are not asserted.

## 本次实际静态检查

- 49 个新增/修改 Python 文件通过语法、Ruff 和格式检查。
- `tests/ut/distributed/kv_transfer/test_hisparse_runtime_adapter.py`：15 passed。
  该套件通过 AST 隔离与设备边界替身执行实际函数定义和真实 CPU torch，不能替代标准全量收集或 NPU 验证。
- `bash format.sh ci` 已尝试，在安装 actionlint 的 pre-commit 环境时达到 45 秒限时；完整检查未通过。
- 新的 Ascend-owned adapter 尚未执行 NPU 图、完整模型、P/D 或性能资格检查。
- vLLM fork PR #1 未更新；配套依赖仍是冻结官方基线加 PR62592。

可复现的 CPU 合同检查（先准备对应 vLLM 源码/依赖）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON" -m pytest -q -p no:cacheprovider \
  --confcutdir=tests/ut/distributed/kv_transfer \
  tests/ut/distributed/kv_transfer/test_hisparse_runtime_adapter.py
```

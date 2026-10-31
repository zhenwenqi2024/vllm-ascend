# HiSparse 集成草稿更新（2026-10-31）

本次更新用于检查累计改动，不是最小合入补丁，也不是已达到性能目标的发布版本。
逐文件必要性、可拆分范围和入口定义见[逐文件审计](hisparse_file_necessity_20261031.md)。
其中兼容改动、独立量化修复、旧 BF16 分支和实验候选不能都算作 HiSparse 的必要成本。

## 本次新增或调整

| 范围 | 改了什么、为什么改 | 限制 |
| --- | --- | --- |
| Host 写回 | 对满足大小、数量和无重叠条件的描述符使用 mapped AIV 写入；描述符使用独立 pinned 快照 | 小批次、大页和重叠目标保留 native DMA；沿用原 stream/fence 和 owner 生命周期 |
| 页面描述符 | 批量计算跨层源地址、目标地址和字节数 | 保持 transfer-major/layer-major 顺序、重复 transfer 和 transfer ID；存储变化拒绝继续 |
| Resident 更新 | 从当轮请求顺序构造更新矩阵，减少索引与多次写表提交 | 不缓存页面值；不更改回收策略、页持久化或分配策略 |
| lazy MTP | 已认证 D-only、全 Resident 路径跳过无人消费的行映射和 staging | 不改计数回拷和 MTP 时序；P 端不得用此策略代替导出前完整镜像 |
| Indexer / invalidate | C8 索引写入和 recycled group invalidate 的 Ascend 实现 | 仍需保持原 slot、group 和事件语义；不把被拒绝的融合候选混入 |
| Host NUMA | 注册 Host 池后保留 CPU 绑定但不再次迁移其锁页内存 | 不改变其他服务，也不修改公共 vLLM 文件 |
| 前缀视图候选 | 复用持久 block-table 的 Tensor 视图对象 | CPU 已验证；NPU/model 未验证，不宣称端到端收益 |

修改都在 Ascend 仓库。公共 vLLM 仍为官方 `43111fd1f0c3135ff671e25e2e85238aa05743d0`
加 PR62592，冻结 merge 为 `17b5c9a97198275745961638a35e78aa1ad907ff`，不带 PR63801。
版本校验包装不是稳定公开扩展接口：不支持的公共源码版本会拒绝安装，后续应由公开 backend hook 替代。

## 最新已完成的模型对照

完整 GLM-5.2 W4A8C8 MXFP、78+1 层、A5 TP8/DP1、MTP3、V2、memory 0.8、Hot10240、
Host128 GiB/rank、FULL_DECODE_ONLY、prefix off。单 D，C8、输入 128 / 输出 8192，
EvalScope 1.13.0，一个不计分 primer，两臂各 N2。不是双机 P/D 结果。

| 指标 | OFF | ON |
| --- | ---: | ---: |
| r1 TPOT（ms） | 16.6911460 | 17.0165075 |
| r2 TPOT（ms） | 16.3903135 | 17.3317395 |
| N2 平均 TPOT（ms） | 16.5407298 | 17.1741235 |
| 合并稳态 decode TPS | 476.157815 | 459.245979 |
| r1 MTP 采信率 | 45.05484% | 47.43594% |
| r2 MTP 采信率 | 46.62714% | 45.73576% |

平均 TPOT 劣化 **3.8293%**，两轮分别为 1.9493% / 5.7438%；目标不超过 3% 尚未达到。
每臂 16/16 成功、实际返回 131072 个输出 token ID。ON settled Host 计数的 miss/H2D 增量为零，
这不是物理 DMA 全事件证明。ON/OFF 输出和 MTP 采信率有差异，均已保留；两轮不构成统计显著性或完整精度结论。
小请求对照、8 ranks × 79 层及 Target/first/later 图证据已取得，正式计时前移除了观察器。

已测源码 tree SHA256：
`b7bf04864e45cbc7255ba5b307455a91944e5a430759058f3f0e70beb6339063`。

## 同 shape profiling 的定位

8 ranks，每 rank 五个完整中间周期。ON 相对 OFF：Target +2.29314 ms，其中主计算流 wait +1.61305 ms，
Attention 计算本身仅 +0.07223 ms；query pack +0.23599 ms、scatter +0.22548 ms、map +0.01448 ms。
末次 draft 到下一 Target +1.05463 ms。应继续定位依赖等待和 CPU 提交间隙，不能据此把 Attention 称作主要根因。

自动 ProfilerStep 标记缺失，阶段由实际 graph/model identity 重建；原始数据和失败保留。
这些是 profiling 周期差异，不是正式 TPOT，不把并行 stream/rank 耗时相加。

## 当前 head 的验证边界

本次收集的运行时源码 tree SHA256：
`a6da14f89646863a22b384128119b07470a12444f94b1ca7a6b426c552754399`。
它在上述已测版本上增加前缀视图复用；该项只有 CPU、真实模块别名/值/回退及冷安装检查，
尚无完成的 NPU 功能或模型性能结果。此前尝试受新外部服务占用中断，未发送模型评测请求。

PR 代码还经过 Ruff 排版，因此本地源树 hash 与运行时 hash 不同。不能把运行时历史结果当作最终 PR head 通过。
未混入待验证的 follower query-prepack；被实测拒绝的 invalidator 融合及较早回退的 mapped 版本不算优化成果。
当前 pinned descriptor mapped 实现与较早失败版本不同，失败记录不被覆盖。

新增可移植 CPU 合同命令：

```bash
python -B -m pytest --noconftest -q \
  tests/ut/distributed/kv_transfer/test_hisparse_host_mirror_descriptors.py \
  tests/ut/distributed/kv_transfer/test_hisparse_submission_helpers.py
```

结果：31 passed。覆盖独立字节拷贝 oracle、相邻区域、映射边界与溢出、回退阈值、当前值与别名、
重新绑定回退、跨层页面描述符顺序/重复 transfer、源存储变化和 Resident 请求顺序。
纯函数用文件导入，Resident 纯 CPU 定义用 AST 隔离；没有执行 NPU kernel 或完整框架安装。
首次测试收集因未安装配套 vLLM 失败，改为纯模块导入后通过，没有将首次失败记作通过。

新增/修改 Python 的 Ruff、格式和语法检查单独记录。`bash format.sh ci` 已尝试，
55 秒后仍在安装 pre-commit actionlint 环境而超时，完整 CI 未通过。

合入前仍需：人工逐行审查、最终 head 独立 native 构建、NPU 正确性、同 shape 模型 N2、
最终双机 P/D 效果和优化完成后的数据集精度。保持 Draft。

AI assistance：OpenAI Codex 协助实现、诊断、测试和发布；不声称已完成提交人的人工逐行审查。

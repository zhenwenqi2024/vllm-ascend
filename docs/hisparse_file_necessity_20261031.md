# HiSparse 逐文件必要性审计（2026-10-31）

这份PR是累计集成草稿，不是最小合入补丁。不是每个文件都必须改。当前实现有依赖，不代表上游无法提供更小实现。

审计覆盖相对PR基线的每个变更文件，区分基础功能、P/D、可选优化、兼容和证据。分类依据是差异及模块职责；
尚未进行逐文件删除后的NPU回归，不能据此宣布任意单个文件可安全删除。文档自身不属于推理实现。

建议正式合入拆为：Core版本兼容 → Ascend基础后端/C8 → P/D传输 → 分项性能优化。旧BF16路径与独立量化修复单独审查。

待实测视图复用不应随正式功能合入；当前保留在Draft供检查。所有版本校验包装需在上游提供公开hook后移除。

| 类型 | 文件数 |
| --- | ---: |
| 证据/文档 | 12 |
| 可选BF16构建 | 2 |
| 可选BF16注意力 | 20 |
| HiSparse功能/桥接 | 29 |
| 测试 | 30 |
| 冻结上游兼容/可拆 | 39 |
| 功能与兼容混合 | 6 |
| 可独立性能优化 | 19 |
| 仅P/D需要 | 17 |
| 独立模型/量化修复 | 2 |
| 待实测候选 | 1 |

## 证据/文档

- `benchmarks/hisparse/qualification/generic_padding.py`（+149/−0）：不参与推理；用于复现、核对历史失败与验证边界。
  主要定义：`load`、`qualify`。
- `benchmarks/hisparse/qualification/resolver60103.py`（+193/−0）：不参与推理；用于复现、核对历史失败与验证边界。
  主要定义：`load`、`qualify`。
- `docs/hisparse_file_necessity_20261031.json`（+2488/−0）：不参与推理；用于复现、核对历史失败与验证边界。
- `docs/hisparse_file_necessity_20261031.md`（+367/−0）：不参与推理；用于复现、核对历史失败与验证边界。
- `docs/hisparse_integration_validation_20261020.json`（+209/−0）：不参与推理；用于复现、核对历史失败与验证边界。
- `docs/hisparse_review_20261023.json`（+457/−0）：不参与推理；用于复现、核对历史失败与验证边界。
- `docs/hisparse_review_20261023.md`（+53/−0）：不参与推理；用于复现、核对历史失败与验证边界。
- `docs/hisparse_review_20261030.json`（+191/−0）：不参与推理；用于复现、核对历史失败与验证边界。
- `docs/hisparse_review_20261030.md`（+116/−0）：不参与推理；用于复现、核对历史失败与验证边界。
- `docs/hisparse_review_20261031.md`（+85/−0）：不参与推理；用于复现、核对历史失败与验证边界。
- `docs/source/user_guide/feature_guide/hisparse.md`（+88/−0）：不参与推理；用于复现、核对历史失败与验证边界。
- `mkdocs.yml`（+1/−0）：不参与推理；用于复现、核对历史失败与验证边界。

## 可选BF16构建

- `csrc/aclnn_torch_adapter/op_api_common.h`（+1/−1）：构建注册或多翻译单元链接支持；随旧BF16原生注意力分支一起拆分。
- `csrc/build_aclnn.sh`（+1/−0）：构建注册或多翻译单元链接支持；随旧BF16原生注意力分支一起拆分。

## 可选BF16注意力

- `csrc/attention/hisparse_flash_attention/CMakeLists.txt`（+19/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/README.md`（+23/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/hisparse_flash_attention_torch_adpt.h`（+81/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_host/CMakeLists.txt`（+29/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_host/hisparse_flash_attention_def.cpp`（+48/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_host/hisparse_flash_attention_infershape.cpp`（+51/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_host/hisparse_flash_attention_tiling.cpp`（+469/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_host/hisparse_flash_attention_tiling.h`（+390/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_host/op_api/aclnn_hisparse_flash_attention.cpp`（+50/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_host/op_api/aclnn_hisparse_flash_attention.h`（+36/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_kernel/arch35/hisparse_flash_attention_common_arch35.h`（+112/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_kernel/arch35/hisparse_flash_attention_kernel_mla.h`（+745/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_kernel/arch35/hisparse_flash_attention_kvcache.h`（+303/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_kernel/arch35/hisparse_flash_attention_service_cube_mla.h`（+387/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_kernel/arch35/hisparse_flash_attention_service_vector_mla.h`（+831/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_kernel/arch35/util_regbase.h`（+267/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_kernel/hisparse_flash_attention.cpp`（+61/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_kernel/hisparse_flash_attention_common.h`（+202/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `csrc/attention/hisparse_flash_attention/op_kernel/hisparse_flash_attention_template_tiling_key.h`（+52/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
- `vllm_ascend/ops/hisparse_direct.py`（+186/−0）：旧BF16直调注意力分支；当前C8路径不以此为必要条件，应与C8支持拆分审查。
  主要定义：`load_direct_library`、`ensure_direct_operator`、`packed_cache_storage`、`compact_physical_indices`、`packed_sparse_flash_attention`。

## HiSparse功能/桥接

- `csrc/hisparse_direct_binding.cpp`（+25/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
- `csrc/hisparse_runtime/batch_copy.cpp`（+160/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
- `csrc/hisparse_runtime/build.py`（+96/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`main`。
- `csrc/hisparse_runtime/host_registration.cpp`（+54/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
- `csrc/hisparse_runtime/resolver.asc`（+450/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
- `csrc/hisparse_runtime/torch_binding.cpp`（+55/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
- `vllm_ascend/attention/hisparse_resident_state.py`（+76/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`eligible_resident_forward`、`ResidentForwardState`、`resident_eager_transport_allowed`。
- `vllm_ascend/attention/sfa_hisparse.py`（+666/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`_hisparse_uniform_mtp_group_size`、`AscendHiSparsePrefillMetadata`、`AscendHiSparseMetadata`、`AscendHiSparseMetadataBuilder`、`AscendHiSparseImpl`。
- `vllm_ascend/distributed/kv_transfer/__init__.py`（+6/−0）：注册后端/connector并支持组合生命周期；仅应在选择对应connector时生效。
  主要定义：`register_connector`。
- `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`（+25/−0）：注册后端/connector并支持组合生命周期；仅应在选择对应connector时生效。
  主要定义：`AscendMultiConnector`。
- `vllm_ascend/distributed/kv_transfer/hisparse/__init__.py`（+3/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
- `vllm_ascend/distributed/kv_transfer/hisparse/backend.py`（+176/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`AscendHiSparseBackend`。
- `vllm_ascend/distributed/kv_transfer/hisparse/backend_dma.py`（+162/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`_Generation`、`HostDMALifetime`。
- `vllm_ascend/distributed/kv_transfer/hisparse/binding.py`（+184/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`align_hisparse_indexer_specs`、`get_hisparse_indexer_specs`、`_validate_indexer_spec`、`_quantized_indexer_views`、`write_hisparse_indexer_c8`。
- `vllm_ascend/distributed/kv_transfer/hisparse/runtime_adapter.py`（+360/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`_ensure_gpu_sources`、`_resolve_residency`、`_resolve_and_stage_group`、`_enqueue_host_mirror`、`AscendHiSparseSlotInvalidator`。
- `vllm_ascend/ops/hisparse.py`（+746/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`MappedHostTensor`、`allocate_mapped_host`、`ResolverWorkspace`、`_cache_layout`、`_check_tensor`。
- `vllm_ascend/ops/hisparse_ascendc.py`（+163/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`BoundResolver`、`try_resolve`。
- `vllm_ascend/ops/hisparse_c8.py`（+192/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`pack_c8_components`、`c8_storage_view`、`map_c8_physical_indices`、`compact_c8_indices`、`c8_physical_attention`。
- `vllm_ascend/ops/hisparse_c8_staged.py`（+72/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`_staged_map_compact`、`prepare_staged_c8_indices`。
- `vllm_ascend/ops/hisparse_compact.py`（+36/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`compact_physical_indices_eager`。
- `vllm_ascend/ops/hisparse_resident.py`（+192/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`resident_index_kernel`、`metadata_signature`、`begin_resident_forward`、`group_mapping_contract`、`resident_topk_or_none`。
- `vllm_ascend/ops/triton/hisparse.py`（+786/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`_translate`、`_lookup_with_position`、`_lookup`、`prepare_residency_kernel`、`classify_residency_kernel`。
- `vllm_ascend/ops/triton/hisparse_compact.py`（+26/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`stable_compact_kernel`。
- `vllm_ascend/ops/triton/hisparse_gather_global.py`（+53/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`gather_compact_global_kernel`。
- `vllm_ascend/ops/triton/hisparse_invalidation.py`（+69/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`_invalidate_hot_slots`、`try_invalidate`。
- `vllm_ascend/ops/triton/hisparse_request_rows.py`（+58/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`_request_rows`、`try_build_request_rows`。
- `vllm_ascend/ops/triton/hisparse_resident_gather.py`（+62/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`_resident_rows_to_staging`、`try_gather_resident_rows`。
- `vllm_ascend/patch/worker/patch_hisparse_invalidation.py`（+31/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`install`。
- `vllm_ascend/patch/worker/patch_hisparse_padding_owner.py`（+149/−0）：参与Ascend后端、Host注册、KV映射或失效生命周期；当前实现依赖，但不等于唯一实现。
  主要定义：`PaddingMaskOwner`、`_prepare_group_owner`、`_padding_mask`、`prepare_group`、`install`。

## 测试

- `tests/e2e/pull_request/one_card/test_hisparse_attention.py`（+105/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`_require_ascend950_sfa`、`test_sparse_attention_reference`。
- `tests/e2e/pull_request/one_card/test_hisparse_direct_attention.py`（+228/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`normal_native_registration`、`test_direct_sfa_matches_bridge_and_cpu_reference`、`test_direct_sfa_addresses_beyond_eight_gib`、`test_direct_sfa_empty_and_tile_boundary_counts`。
- `tests/e2e/pull_request/one_card/test_hisparse_kv_preprocess.py`（+204/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`_setup`、`test_fused_preprocess_preserves_native_rounding_and_packed_writes`、`test_nope_preprocess_uses_native_layernorm`。
- `tests/e2e/pull_request/one_card/test_hisparse_ops.py`（+557/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`_require_mapped_host_device`、`reference`、`test_resolver_strides_padding_graph`、`test_topk_2048_speculative_union_graph`、`test_random_request_unions`。
- `tests/e2e/pull_request/one_card/test_hisparse_prefill_staging.py`（+84/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_resident_prefill_plan_shared_layers`。
- `tests/e2e/pull_request/one_card/test_hisparse_selection_order.py`（+85/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_valid_order_duplicates_and_interleaved_padding`。
- `tests/e2e/pull_request/one_card/test_shared_swiglu_a5.py`（+88/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_a5_shared_mlp_standard_swiglu`。
- `tests/e2e/pull_request/one_card/test_w4a8_a5.py`（+122/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`_cpu_gmm`、`_assert_close`、`test_a5_w4a8_modelslim_mlp`。
- `tests/ut/core/test_mla_storage_block_size.py`（+40/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_mla_storage_uses_num_states`、`test_sfa_indexer_storage_rows_match_page_size`、`test_sliding_window_mla_preserves_custom_storage_compression`。
- `tests/ut/distributed/kv_transfer/hisparse_pd/conftest.py`（+125/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`load_classes`、`modules`。
- `tests/ut/distributed/kv_transfer/hisparse_pd/test_cleanup.py`（+422/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`Owner`、`Engine`、`Device`、`environment`、`owners`。
- `tests/ut/distributed/kv_transfer/hisparse_pd/test_graph_mtp_contract.py`（+110/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`config`、`policy`、`test_mtp_graph_handoff_exports_only_completed_prefix`、`test_unimplemented_draft_policies_remain_rejected`、`test_loaded_layer_names_do_not_change_common_group_planning`。
- `tests/ut/distributed/kv_transfer/hisparse_pd/test_pd.py`（+555/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`port`、`layouts`、`test_real_strided_payload_partial_pages`、`test_64bit_addresses_and_reject_missing_geometry`、`test_lease_does_not_free_on_rank_mismatch_timeout_or_unfinished_read`。
- `tests/ut/distributed/kv_transfer/hisparse_pd/test_scheduler_lifecycle.py`（+118/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`actual_classes`、`test_cancel_waits_for_actual_receive_drain`。
- `tests/ut/distributed/kv_transfer/hisparse_pd/test_transport_config.py`（+368/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`configuration`、`test_canonical_bounded_mib`、`test_invalid_mib`、`test_pd_protocol_must_be_explicit`、`test_default_is_sixteen_mib_only_for_pd`。
- `tests/ut/distributed/kv_transfer/test_hisparse.py`（+382/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_explicit_dma_extension_load_keeps_global_op_policy`、`test_lru_init_avoids_unsupported_int16_range`、`test_backend_current_stream_tracks_context_changes`、`test_hisparse_uses_core_role_managers_and_host_pool`、`test_registration_retains_mapping_until_explicit_release`。
- `tests/ut/distributed/kv_transfer/test_hisparse_host_mirror_descriptors.py`（+106/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_mapped_snapshot_preserves_exact_bytes_and_unwritten_neighbors`、`test_ineligible_writes_keep_native_dma`、`test_bad_ownership_is_rejected_before_enqueue`、`test_small_batch_fallback_preserves_exact_descriptor_snapshot`、`test_small_batch_still_rejects_expired_or_invalid_ownership`。
- `tests/ut/distributed/kv_transfer/test_hisparse_runtime_adapter.py`（+285/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`load_definitions`、`test_prefill_resident_mapping_matches_common`、`test_group_invalidation_updates_all_leaders_and_reuses_pointer_storage`、`test_stage_keeps_dependency_and_follower_event_order`、`test_worker_mirror_preserves_dma_and_batches_only_active_group_leaders`。
- `tests/ut/distributed/kv_transfer/test_hisparse_submission_helpers.py`（+136/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`load_module`、`test_prefix_views_alias_current_values_without_recreating_views`、`test_rebound_or_custom_tables_do_not_reuse_old_allocation_views`、`test_page_descriptors_match_scalar_oracle_and_preserve_duplicate_transfer_order`、`resident_functions`。
- `tests/ut/ops/test_hisparse_direct.py`（+125/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_raw_alias_preserves_layer_offset_and_page_gaps`、`test_expanded_attention_view_alias_stays_within_tail_storage`、`test_reject_unsupported_cache_layouts`、`test_compaction_preserves_order_duplicates_and_high_physical_ids`、`test_full_prefill_issues_one_launch_without_materializing_kv`。
- `tests/ut/ops/test_shared_swiglu_a5.py`（+63/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_standard_swiglu_keeps_other_contracts`。
- `tests/ut/patch/platform/test_kv_cache_planning_compat.py`（+53/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`_config_and_groups`、`test_default_builder_materializes_exact_requested_block_count`、`test_memory_accounting_preserves_selected_builder`。
- `tests/ut/quantization/methods/w4a8/test_w4a8_a5.py`（+35/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_w4a8_fused_activation_hardware_scope`、`test_a5_unverified_weight_layouts_fail_before_loading`。
- `tests/ut/test_platform.py`（+11/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_validate_v4_index_cache_pp_partition`、`test_v4_index_cache_schedule_uses_indexer_ordinals`、`test_validate_v4_index_cache_pp_pattern`、`test_v4_index_cache_recomputes_first_local_c4`、`test_ascend_sequence_parallel_moe_supports_dp1`。
- `tests/ut/worker/test_model_runner_v2_kv_cache.py`（+12/−2）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_initialize_preserves_connector_containers_and_flattens_runner_cache`、`test_mrv2_block_copy_preserves_segmented_mamba_storage`。
- `tests/ut/worker/test_worker_v1.py`（+8/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`TestNPUWorker`、`TestNPUWorkerWeightUpdate`、`TestKVPPWorkerBudget`。
- `tests/ut/worker/v2/test_block_table.py`（+32/−2）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`_allow_cpu_buffers`、`test_encoder_only_zero_cache_groups_initialize_and_compute`、`test_current_core_keeps_manager_ids_with_smaller_kernel_pages`。
- `tests/ut/worker/v2/test_spec_decode_sampling_compat.py`（+88/−0）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_rejection_statistics_cover_all_non_bonus_logits`、`test_draft_sampling_forwards_step_and_watermark_width`。
- `tests/ut/worker/v2/test_speculator.py`（+38/−3）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`test_calc_next_seq_lens_cpu`、`test_get_seq_lens_cpu`、`test_gather_without_out`、`test_gather_with_out`、`test_torch_gather_wrapper`。
- `tests/ut/worker/v2/test_v2_block_table.py`（+5/−5）：不参与推理；覆盖对应行为，不能因运行时不引用而删除。
  主要定义：`_parent_init`、`_init_tables`、`test_init_defaults_kernel_sizes_and_rebuilds_int32_slots`、`test_init_keeps_explicit_kernel_block_sizes`、`test_init_forwards_slot_mapping_enabled_on_newer_vllm`。

## 冻结上游兼容/可拆

- `vllm_ascend/_310p/worker/v2/spec_decode/mtp_speculator.py`（+3/−3）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`AscendMTPSpeculator310`。
- `vllm_ascend/_310p/worker_310p.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`NPUWorker310`。
- `vllm_ascend/ascend_config.py`（+7/−2）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`is_mega_moe_supported`、`validate_additional_config_bool`、`KVPPConfig`、`AscendCompilationConfig`、`AscendFusionConfig`。
- `vllm_ascend/attention/mla_v1.py`（+3/−5）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`AscendMLABackend`、`AscendMLAAttention`、`ChunkedContextMetadata`、`AscendMLAPrefillMetadata`、`AscendMLADecodeMetadata`。
- `vllm_ascend/compilation/graph_fusion_pass_manager.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`GraphFusionPassManager`。
- `vllm_ascend/core/kv_cache_interface.py`（+6/−6）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`get_kv_cache_compression_ratio`、`get_storage_block_size`、`is_circular_kv_cache_spec`、`is_prefix_cacheable`、`declared_kwarg`。
- `vllm_ascend/kv_cache_dtype.py`（+68/−0）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`native_c8_cache_dtype`、`prepare_native_hisparse_view_dtypes`。
- `vllm_ascend/models/deepseek_v4/compressor.py`（+1/−2）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`AscendCompressorStateCache`、`AscendCompressorMetadata`、`Compressor`。
- `vllm_ascend/models/deepseek_v4/indexer.py`（+1/−2）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`hadamard_linear`、`hadamard_scale`、`rotate_activation`、`_is_w8a8_dynamic`、`_is_mxfp8_dynamic`。
- `vllm_ascend/models/deepseek_v4/mm_preprocess.py`（+1/−2）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_get_thread_local_tokenizer`、`image_sentinel_mask`、`validate_image_sentinel_ids`、`grid_tokens`、`solve_resize_ratio`。
- `vllm_ascend/models/deepseek_v4/model.py`（+1/−2）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`AscendDeepseekV4SWACache`、`precompute_freqs_cis_cpu`、`apply_rotary_emb`、`get_spec_layer_idx_from_weight_name`、`DeepseekV2MLP`。
- `vllm_ascend/models/deepseek_v41/cache_config.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`uses_a5_packed_cache`、`make_swa_cache_spec`、`make_mla_cache_spec`、`make_index_cache_spec`、`make_folded_index_cache_spec`。
- `vllm_ascend/models/deepseek_v41/engram/parallel.py`（+2/−2）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`resolve_dp_shared_memory`、`engram_gathered_num_tokens`、`gather_engram_hashes`、`exchange_engram_rows`。
- `vllm_ascend/models/glm5next/cache_config.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_Glm5NextCacheLayout`、`_is_glm5_next_spec`、`_unpadded_page_size`、`_sorted_layer_names`、`_layer_indices`。
- `vllm_ascend/ops/gdn.py`（+0/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_get_fla_gdn_prefill_op`、`_get_base_conv1d`、`_pack_conv_weights`、`initialize_packed_conv_weight`、`_get_packed_conv_weights`。
- `vllm_ascend/ops/kimi_kda.py`（+0/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_kda_bfg_stream`、`_KDAFusedBFGLinear`、`_normalize_causal_cache_indices`、`_prepare_beta`、`AscendKimiK3DeltaAttention`。
- `vllm_ascend/ops/linear.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`unquantized_gemm`、`unquantized_gemm_fake`、`_should_keep_nd_for_compatibility_weight`、`_should_reshape_wo_a_to_3d`、`AscendUnquantizedLinearMethod`。
- `vllm_ascend/ops/mla.py`（+2/−0）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`IndexerWrapper`、`AscendMultiHeadLatentAttention`、`mla_forward`、`mla_forward_fake`。
- `vllm_ascend/ops/rotary_embedding.py`（+2/−2）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`set_cos_and_sin`、`get_cos_and_sin_mla`、`_record_cos_sin_cache`、`_record_cos_and_sin_cache`、`_record_cos_and_sin_cache_interleaved`。
- `vllm_ascend/patch/__init__.py`（+25/−0）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
- `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`（+29/−0）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_skips_eagle_block_drop`、`_select_kv_token_budget`、`_is_deepseek_v4_kv_cache_spec`、`_is_deepseek_v4_kv_cache_config`、`_manager_spec`。
- `vllm_ascend/patch/platform/patch_kv_cache_utils.py`（+20/−26）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_page_sizes`、`_ascend_resolve_kv_cache_block_sizes`、`_get_kimi_k3_dspark_mixed_kv_cache_groups`、`_get_kv_cache_groups_uniform_page_size`、`_kv_cache_config_has_mamba_layers`。
- `vllm_ascend/patch/worker/__init__.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
- `vllm_ascend/patch/worker/patch_deepseek_v2.py`（+1/−2）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_should_skip_indexer_init`、`_deepseek_v2_mla_attention_init`、`_find_topk_indices_buffer`、`_patched_deepseek_v2_model_init`、`_set_aux_hidden_state_layers`。
- `vllm_ascend/patch/worker/patch_v2/patch_adaptive_verification.py`（+20/−9）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_index_fill`、`_IndexFillMode`、`_assign_draft_token_budget_ascend`。
- `vllm_ascend/patch/worker/patch_v2/patch_eagle_speculator.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
- `vllm_ascend/patch/worker/patch_v2/patch_model_runner.py`（+4/−9）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`dispatch_cg_and_sync_dp`、`initialize_kv_cache`。
- `vllm_ascend/patch/worker/patch_v2/patch_spec_decode_310.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_embed_input_ids`、`_init_speculator_310p`。
- `vllm_ascend/patch/worker/patch_v2/patch_uva.py`（+15/−0）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`check_triton_ascend_version_valid`、`is_uva_available`、`get_row_indices_from_key`、`MonitoredNumPyArray`、`MonitoredTorchTensor`。
- `vllm_ascend/quantization/methods/kv_cache/turboquant/cache.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`uses_turboquant_groups`、`group_specs`、`_alignment`、`packed_layout`、`pool_bytes_per_block`。
- `vllm_ascend/spec_decode/__init__.py`（+4/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`get_spec_decode_method`。
- `vllm_ascend/worker/v2/aclgraph_utils.py`（+29/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_prepare_pcp_inputs_to_capture`、`collect_sorted_captured_token_sizes`、`_get_graph_update_backend`、`ModelAclGraphManager`、`_suspend_hisparse_mirrors_for_capture`。
- `vllm_ascend/worker/v2/attn_utils.py`（+43/−5）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`skip_ring_state_update`、`ring_state_update_skipped`、`unwrap_mamba_kv_cache_groups`、`_align_hybrid_attention_page_sizes`、`get_kv_cache_spec`。
- `vllm_ascend/worker/v2/block_table.py`（+11/−4）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`AscendBlockTables`。
- `vllm_ascend/worker/v2/sample/apply_top_k_top_p.py`（+9/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`apply_top_k_top_p_npu`。
- `vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py`（+1/−1）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`AutoRegressiveAclGraphManager`。
- `vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py`（+29/−4）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`AscendAutoRegressiveSpeculator`、`gather`、`torch_gather_wrapper`。
- `vllm_ascend/worker/v2/spec_decode/lmhead_tp_utils.py`（+21/−3）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`LmheadTPDraftSamplingMixin`。
- `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`（+25/−4）：适配当前冻结Core接口或独立模型路径；不是HiSparse算法必需，换基线后应重审并独立提交。
  主要定义：`_probabilistic_rejection_kernel`、`rejection_sample`。

## 功能与兼容混合

- `vllm_ascend/attention/indexer.py`（+63/−2）：包含HiSparse/C8选择或硬件能力接入；必须按hunk拆分基础功能、可选优化与兼容改动。
  主要定义：`AscendSFAIndexerMetadata`、`AscendSFAIndexerBackend`、`AscendSFAIndexerMetadataBuilder`、`AscendHiSparseIndexerMetadataBuilder`。
- `vllm_ascend/attention/sfa_v1.py`（+21/−12）：包含HiSparse/C8选择或硬件能力接入；必须按hunk拆分基础功能、可选优化与兼容改动。
  主要定义：`generate_smla_plan`、`build_smla_metadata`、`_view_cache_as_operator_pages`、`sparse_mla`、`SparseMLAMetadataState`。
- `vllm_ascend/device/hardware_profile.py`（+4/−0）：包含HiSparse/C8选择或硬件能力接入；必须按hunk拆分基础功能、可选优化与兼容改动。
  主要定义：`HardwareCapability`、`AttentionBackendFamily`、`CPUBindingMode`、`DeviceAdaptorFamily`、`DeviceAddressingMode`。
- `vllm_ascend/platform.py`（+9/−4）：包含HiSparse/C8选择或硬件能力接入；必须按hunk拆分基础功能、可选优化与兼容改动。
  主要定义：`NPUPlatform`、`_configure_minimax_m3_a5_mixed_kv_cache`、`_fix_incompatible_config`、`_validate_model_runner_config`、`_validate_eplb_config`。
- `vllm_ascend/worker/v2/model_runner.py`（+8/−1）：包含P/D owner绑定、profiling签名和dummy preemption保护；可分别审查，不能归为计数/MTP时序修复。
  主要定义：`NPUModelRunner`、`graph_manager_wrapper`。
- `vllm_ascend/worker/worker.py`（+14/−5）：同时包含新Core导入、shared-slab计费绕过及已注册Host页的NUMA保护；不能按整文件判断必要。
  主要定义：`NPUWorker`、`parse_text_output`。

## 可独立性能优化

- `vllm_ascend/distributed/kv_transfer/hisparse/host_mirror_descriptors.py`（+65/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`prepare_snapshot`。
- `vllm_ascend/distributed/kv_transfer/hisparse/page_dma_descriptors.py`（+187/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`_PageGeometry`、`_geometry`、`enqueue_transfers`、`install`。
- `vllm_ascend/distributed/kv_transfer/hisparse/queued_dma.py`（+198/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`QueuedHostDMA`。
- `vllm_ascend/distributed/kv_transfer/hisparse/resident_update_installer.py`（+37/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`install`。
- `vllm_ascend/distributed/kv_transfer/hisparse/resident_updates.py`（+136/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`pack_current_updates`、`fresh_host_updates`、`unique_destinations`、`eligible_update_matrix`、`_apply_updates`。
- `vllm_ascend/ops/hisparse_c8_mapping.py`（+87/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`_map_compact_prefix1417`、`map_compact_c8_indices`。
- `vllm_ascend/ops/hisparse_c8_scope.py`（+130/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`C8MappingScope`、`begin_c8_mapping_scope`、`current_c8_mapping_scope`。
- `vllm_ascend/ops/triton/hisparse_c8_fused.py`（+214/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`_scatter_components`、`scatter_c8_components`、`_pack_query_lengths`、`pack_c8_query_lengths`、`_zero_empty_rows`。
- `vllm_ascend/ops/triton/hisparse_c8_map_pack.py`（+140/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`_map32`、`_map_pack`、`geometry`、`map32`、`map_pack`。
- `vllm_ascend/ops/triton/hisparse_gather_persistent.py`（+59/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`gather_rows_persistent_kernel`。
- `vllm_ascend/ops/triton/hisparse_host_mirror.py`（+16/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`store_opaque_extents`。
- `vllm_ascend/ops/triton/hisparse_indexer_c8_write.py`（+93/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`_write_indexer_c8`、`try_write`。
- `vllm_ascend/ops/triton/hisparse_invalidate_groups.py`（+76/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`invalidate_groups_kernel`、`invalidate_written_groups`。
- `vllm_ascend/ops/triton/hisparse_recycled_invalidate.py`（+132/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`normalized_blocks`、`_invalidate_recycled_groups`、`RecycledGroupInvalidator`。
- `vllm_ascend/patch/worker/patch_hisparse_lazy_mtp_mirror.py`（+207/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`_eligible`、`_verify_contract`、`_validate_consumer`、`install`。
- `vllm_ascend/patch/worker/patch_hisparse_lazy_row_map.py`（+86/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`can_discard`、`resolve`、`install`。
- `vllm_ascend/patch/worker/patch_hisparse_recycled_groups.py`（+45/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`install`。
- `vllm_ascend/patch/worker/patch_hisparse_streams.py`（+146/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`_is_npu`、`_native_stream`、`HiSparseStream`、`HiSparseEvent`、`native_current_stream`。
- `vllm_ascend/patch/worker/patch_hisparse_unused_staging.py`（+140/−0）：替代原有准备/搬运/映射路径；应独立证明正确性和收益，不能视为功能必需。
  主要定义：`scope_for`、`make_bindings`、`install`。

## 仅P/D需要

- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/__init__.py`（+1/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/acl_stage.py`（+396/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`PTAStageBuffer`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/client.py`（+100/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`SnapshotClient`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/configuration.py`（+75/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`parse_urma_swap_mib`、`validate_transport_options`、`configure_urma_staging`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/connector.py`（+243/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`HiSparsePDMetadata`、`HiSparsePDConnector`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/execution.py`（+64/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`draft_transfer_enabled`、`execution_contract`、`draft_state`、`validate_draft_state`、`export_mirror_ready`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/import_session.py`（+138/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`ImportSession`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/layout.py`（+190/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`_fp8_indexer_layouts`、`build_layouts`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/owners.py`（+23/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`bind_loaded_owners`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/pool.py`（+232/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`PoolKey`、`SharedSnapshotPool`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/protocol.py`（+417/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`validate_wire`、`digest`、`LayerLayout`、`validate_layouts`、`plan_reads`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/stage.py`（+175/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`StageSlot`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/stage_reorder2581.py`（+23/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`_reorder`、`reorder`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/topology.py`（+104/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`_integer`、`Replica`、`ProducerPeer`、`ProducerDirectory`、`contiguous_peers`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/transfer.py`（+76/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`StagedTransfer`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/transport.py`（+414/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`TransferCancelled`、`LeaseServer`、`RpcClient`、`rpc`、`acquire`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd/worker.py`（+587/−0）：属于ticket、导入导出或传输生命周期；单机HiSparse不需要，本轮单D结果不能验证它。
  主要定义：`HiSparsePDWorker`。

## 独立模型/量化修复

- `vllm_ascend/ops/fused_moe/shared_experts.py`（+23/−1）：A5 SwiGLU或INT4权重ABI修复；不由HiSparse算法要求，单列，不计为HiSparse性能优化。
  主要定义：`SharedExpertParallelMode`、`SharedExpertMLPPath`、`PreparedSharedExpertExecution`、`SubmittedSharedExpertGateUp`、`AscendSharedExperts`。
- `vllm_ascend/quantization/methods/w4a8/w4a8.py`（+29/−2）：A5 SwiGLU或INT4权重ABI修复；不由HiSparse算法要求，单列，不计为HiSparse性能优化。
  主要定义：`AscendW4A8DynamicFusedMoEMethod`。

## 待实测候选

- `vllm_ascend/patch/worker/patch_hisparse_block_table_views.py`（+87/−0）：复用前缀视图对象；仅CPU验证，尚无模型收益证据，不是基础功能必需。
  主要定义：`current_table_views`、`bind`、`install`。

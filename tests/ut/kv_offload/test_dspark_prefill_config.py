# SPDX-License-Identifier: Apache-2.0
"""P-side DSpark configuration belongs to the context backend, not the runner."""

from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import get_pd_dspark_aux_layer_ids


def _config(
    ids=(2, 22, 38, 58, 74), *, producer=True, consumer=False, speculative=None, pcp=1, dcp=1, eager=True, prefix=False
):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"dspark_aux_hidden_state_layer_ids": ids},
            is_kv_producer=producer,
            is_kv_consumer=consumer,
        ),
        speculative_config=speculative,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp),
        model_config=SimpleNamespace(enforce_eager=eager, hf_text_config=SimpleNamespace(num_hidden_layers=78)),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix),
    )


@pytest.mark.parametrize("ids", [[2, 22, 38, 58, 74], (2, 22, 38, 58, 74), [0, 78], (0, 78)])
def test_backend_returns_immutable_boundaries_without_mutating_config(ids):
    config = _config(ids)
    assert get_pd_dspark_aux_layer_ids(config) == tuple(ids)
    assert config.kv_transfer_config.kv_connector_extra_config["dspark_aux_hidden_state_layer_ids"] is ids


@pytest.mark.parametrize("ids", [[], [38, 22], [2, 2], [79], [-1], [True], [1.0], "2,22,38", {2: 22}, [[2]]])
def test_backend_rejects_invalid_auxiliary_schema(ids):
    with pytest.raises(ValueError, match="ordered unique target-layer boundaries"):
        get_pd_dspark_aux_layer_ids(_config(ids))


@pytest.mark.parametrize(
    "options,message",
    [
        ({"producer": False}, "P-only producer"),
        ({"consumer": True}, "P-only producer"),
        ({"speculative": object()}, "P-only producer"),
        ({"pcp": 2}, "context parallelism"),
        ({"dcp": 2}, "context parallelism"),
        ({"eager": False}, "eager prefill"),
        ({"prefix": True}, "prefix caching disabled"),
    ],
)
def test_backend_preserves_capture_configuration_constraints(options, message):
    with pytest.raises(ValueError, match=message):
        get_pd_dspark_aux_layer_ids(_config(**options))


@pytest.mark.parametrize("extra", [None, {}, {"dspark_aux_hidden_state_layer_ids": None}])
def test_unconfigured_backend_does_not_apply_dspark_constraints(extra):
    # Other fields deliberately omitted: opting out must not inspect them.
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra))
    assert get_pd_dspark_aux_layer_ids(config) == ()


@pytest.mark.parametrize("config", [SimpleNamespace(), SimpleNamespace(kv_transfer_config=None)])
def test_without_kv_transfer_does_not_enable_capture(config):
    assert get_pd_dspark_aux_layer_ids(config) == ()


def test_d_checkpoint_without_p_option_does_not_enable_prefill_capture():
    config = _config(None, producer=False, consumer=True, speculative=SimpleNamespace(method="dspark"), eager=False)
    assert get_pd_dspark_aux_layer_ids(config) == ()

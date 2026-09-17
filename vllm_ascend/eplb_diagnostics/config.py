# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm_ascend.config_utils import config


@config
class EplbDiagnosticsConfig:
    """Independent of EPLB: observe the current execution without changing placement."""

    mode: Literal["off", "observe"] = "off"
    run_id: str = Field(default="", pattern=r"^[a-zA-Z0-9_.-]*$", max_length=128)
    output_dir: str = "eplb_diagnostics"
    sample_interval: int = Field(default=32, ge=1)
    burst_size: int = Field(default=1, ge=1, le=64)
    warmup_steps: int = Field(default=32, ge=0)
    max_samples: int = Field(default=128, ge=1)
    max_pending: int = Field(default=2, ge=1, le=16)
    max_snapshot_mb: int = Field(default=16, ge=1, le=256)

    @model_validator(mode="after")
    def validate_enabled(self) -> Self:
        if self.burst_size > self.sample_interval:
            raise ValueError("burst_size must not exceed sample_interval")
        if self.mode != "off" and (not self.run_id or not self.output_dir):
            raise ValueError("EPLB diagnostics requires a shared, unique run_id and a nonempty output_dir.")
        return self

# ----------------------------------------------------------------------------
# ACT checkpoint compatibility layer for RoboSynChallenge.
#
# Released ACT checkpoints carry their normalization statistics *inside*
# `model.safetensors` (keys such as `normalize_inputs.buffer_observation_state.mean`).
# Whether those statistics actually reach the actuators depends on the runtime
# that loads them:
#
#   * A runtime whose ACTPolicy already builds `normalize_inputs` /
#     `unnormalize_outputs` populates them from the state dict. Nothing to do.
#   * A runtime that stopped building them, or that loads with `strict=False`,
#     will happily accept the state dict while dropping the statistics. The
#     policy then emits raw model-space numbers straight to the actuators, and
#     the rollout still "runs" - it is just silently wrong, with no error raised
#     anywhere.
#
# `CheckpointACTPolicy` closes that gap: it restores the transformations when the
# runtime is missing them, and it refuses lax loading so the statistics cannot be
# skipped without notice. The inverse transform multiplies by the standard
# deviation rather than dividing, so a degenerate (constant) actuator statistic
# maps every model-space value back to its mean exactly instead of producing
# non-finite actions.
# ----------------------------------------------------------------------------
from __future__ import annotations

import torch
from torch import nn

from lerobot.configs.types import NormalizationMode
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.normalize import create_stats_buffers

ACTION = "action"

# Matches LeRobot's own Normalize; kept identical so a checkpoint trained by
# either path produces the same numbers.
_FORWARD_EPSILON = 1e-8


class _CheckpointNormalization(nn.Module):
    """Apply checkpoint-embedded statistics, in either direction.

    Buffer naming mirrors LeRobot (`buffer_<feature_key>` with dots replaced by
    underscores) so a state dict saved by the trainer loads without remapping.
    Features absent from the batch are skipped: released checkpoints list unused
    `observation.qvel` / `observation.qf` metadata that the policy never consumes.
    """

    def __init__(self, features, norm_map, inverse: bool = False):
        super().__init__()
        self.features = dict(features)
        self.norm_map = dict(norm_map)
        self.inverse = bool(inverse)
        for key, buffer in create_stats_buffers(self.features, self.norm_map).items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    def forward(self, batch: dict) -> dict:
        # Shallow copy: the caller's tensors are never mutated in place.
        batch = dict(batch)
        for key, feature in self.features.items():
            if key not in batch:
                continue

            mode = self.norm_map.get(feature.type, NormalizationMode.IDENTITY)
            if mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if mode is NormalizationMode.MEAN_STD:
                mean, std = buffer["mean"], buffer["std"]
                if self.inverse:
                    # No epsilon here: multiplying by a zero standard deviation
                    # must land exactly on the mean, not near it.
                    batch[key] = batch[key] * std + mean
                else:
                    batch[key] = (batch[key] - mean) / (std + _FORWARD_EPSILON)
            elif mode is NormalizationMode.MIN_MAX:
                lower, upper = buffer["min"], buffer["max"]
                if self.inverse:
                    batch[key] = (batch[key] + 1) / 2 * (upper - lower) + lower
                else:
                    span = upper - lower
                    batch[key] = (batch[key] - lower) / (span + _FORWARD_EPSILON) * 2 - 1
            else:
                raise ValueError(f"unsupported normalization mode: {mode}")

        return batch


class CheckpointACTPolicy(ACTPolicy):
    """ACTPolicy that guarantees embedded statistics reach the actuators."""

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, strict: bool = True, **kwargs):
        if not strict:
            raise ValueError(
                "CheckpointACTPolicy requires a strict checkpoint load: loading with "
                "strict=False would silently accept a state dict whose embedded "
                "normalization statistics are missing or mismatched"
            )
        return super().from_pretrained(pretrained_name_or_path, *args, strict=True, **kwargs)

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # A runtime that already builds these has nothing to restore; one that
        # does not is exactly the case this class exists for.
        self._restore_inline_normalization = not hasattr(self, "normalize_inputs")
        if self._restore_inline_normalization:
            inputs = getattr(config, "input_features", None) or {}
            outputs = getattr(config, "output_features", None) or {}
            mapping = getattr(config, "normalization_mapping", None) or {}
            self.normalize_inputs = _CheckpointNormalization(inputs, mapping)
            self.normalize_targets = _CheckpointNormalization(outputs, mapping)
            self.unnormalize_outputs = _CheckpointNormalization(outputs, mapping, inverse=True)

    def predict_action_chunk(self, batch: dict, **kwargs):
        if not self._restore_inline_normalization:
            # The runtime normalizes on its own; adding ours would apply the
            # statistics twice and corrupt every action.
            return super().predict_action_chunk(batch, **kwargs)

        normalized = self.normalize_inputs(batch)
        actions = super().predict_action_chunk(normalized, **kwargs)
        return self.unnormalize_outputs({ACTION: actions})[ACTION]

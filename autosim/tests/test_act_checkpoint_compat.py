"""ACT adapter regression: model-space numbers must never reach actuators."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

from autosim.research.common import find_benchmark


# The benchmark is a separate checkout whose location is the operator's choice, so it is
# found by what it contains rather than by counting ancestors. A fixed ancestor count made
# moving the checkout a test failure; this only fails if the file is absent inside a repo
# that is otherwise there.
_repo = find_benchmark("RoboSynChallenge", Path(__file__).resolve().parents[2],
                       marker="scripts/eval_policy.py")
if _repo is None:
    pytest.skip("no RoboSynChallenge checkout on disk", allow_module_level=True)

_path = _repo / "policy/act/checkpoint_compat.py"
if not _path.is_file():
    pytest.skip(f"checkout has no {_path.name}; deploy it from patches/",
                allow_module_level=True)

_spec = importlib.util.spec_from_file_location("act_checkpoint_compat_test", _path)
compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compat)


def transform(mode=NormalizationMode.MEAN_STD, inverse=False):
    module = compat._CheckpointNormalization(
        {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        {FeatureType.ACTION: mode}, inverse=inverse)
    values = {"mean": [2., 4.], "std": [.5, 0.]} if mode == NormalizationMode.MEAN_STD else {"min": [2., 4.], "max": [6., 4.]}
    for name, value in values.items():
        module.buffer_action[name].data.copy_(torch.tensor(value))
    return module


def test_inverse_preserves_constant_actuator_and_broadcasts_chunk():
    normalized = torch.tensor([[[3., -900.], [-5., 900.]]])
    result = transform(inverse=True)({"action": normalized})["action"]
    torch.testing.assert_close(result, torch.tensor([[[3.5, 4.], [-.5, 4.]]]), rtol=0, atol=0)
    assert torch.equal(normalized, torch.tensor([[[3., -900.], [-5., 900.]]]))


@pytest.mark.parametrize("mode", [NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX])
def test_transform_roundtrip_and_legacy_optional_input(mode):
    data = {"action": torch.tensor([[3., 4.]])}
    result = transform(mode, inverse=True)(transform(mode)(data))
    torch.testing.assert_close(result["action"], data["action"])
    # Match legacy checkpoints with unused qvel/qf feature metadata.
    assert transform(mode)({}) == {}


@pytest.mark.parametrize("legacy_runtime", [False, True])
def test_normalization_applied_exactly_once(monkeypatch, legacy_runtime):
    config = SimpleNamespace(input_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        normalization_mapping={FeatureType.ACTION: NormalizationMode.MEAN_STD})
    seen = []

    def init(self, cfg, **kwargs):
        nn.Module.__init__(self)
        if legacy_runtime:
            self.normalize_inputs = transform()
            self.normalize_targets = transform()
            self.unnormalize_outputs = transform(inverse=True)

    def predict(self, batch):
        if legacy_runtime:
            batch = self.normalize_inputs(batch)
        seen.append(batch["action"].clone())
        result = torch.tensor([[[3., -900.]]])
        return self.unnormalize_outputs({"action": result})["action"] if legacy_runtime else result

    monkeypatch.setattr(compat.ACTPolicy, "__init__", init)
    monkeypatch.setattr(compat.ACTPolicy, "predict_action_chunk", predict)
    policy = compat.CheckpointACTPolicy(config)
    if not legacy_runtime:
        policy.normalize_inputs, policy.normalize_targets = transform(), transform()
        policy.unnormalize_outputs = transform(inverse=True)
    output = policy.predict_action_chunk({"action": torch.tensor([[3., 4.]])})
    torch.testing.assert_close(seen[0], torch.tensor([[2., 0.]]))
    torch.testing.assert_close(output, torch.tensor([[[3.5, 4.]]]))


def test_deployment_refuses_non_strict_loading():
    with pytest.raises(ValueError, match="strict checkpoint"):
        compat.CheckpointACTPolicy.from_pretrained("unused", strict=False)

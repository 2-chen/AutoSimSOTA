"""What a checkpoint is, tested against the thing the old answer could not tell apart.

The system identified a checkpoint by `sha256(model.safetensors)` -- forty-eight call sites
of that one expression. Two runs stopped at different steps, with different optimizers and
different random streams, had the same identity, and the identity is what the ledger, the
evaluation request and the round comparison all key on.
"""

from pathlib import Path

from autosim.research.checkpoint import Checkpoint, identity_of, role_of
from autosim.research.common import digest


def a_checkpoint(root: Path) -> Path:
    """The shape a LeRobot/ACT trainer writes."""
    checkpoint = root / "train/checkpoints/020000/pretrained_model"
    (checkpoint / "training_state").mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"WEIGHTS")
    (checkpoint / "config.json").write_text('{"policy": "act"}')
    (checkpoint / "train_config.json").write_text('{"steps": 20000}')
    state = checkpoint / "training_state"
    (state / "optimizer_state.safetensors").write_bytes(b"OPTIMIZER-1")
    (state / "optimizer_param_groups.json").write_text("[]")
    (state / "rng_state.safetensors").write_bytes(b"RNG")
    (state / "training_step.json").write_text('{"step": 20000}')
    return checkpoint


def test_two_checkpoints_that_differ_only_in_optimizer_state_are_two_checkpoints(tmp_path):
    """The whole point, and what the single-file hash could not say."""
    checkpoint = a_checkpoint(tmp_path)
    before = identity_of(checkpoint)
    weights_before = digest(checkpoint / "model.safetensors")

    (checkpoint / "training_state/optimizer_state.safetensors").write_bytes(b"OPTIMIZER-2")

    assert identity_of(checkpoint) != before, "the optimizer is part of the checkpoint"
    assert digest(checkpoint / "model.safetensors") == weights_before, \
        "and the weights alone would not have noticed"


def test_a_checkpoint_that_moved_is_the_same_checkpoint(tmp_path):
    """Identity is over the files, not over where they sit: a rename is not a new policy."""
    checkpoint = a_checkpoint(tmp_path)
    before = identity_of(checkpoint)
    assert identity_of(checkpoint.parent) != before, "the root is not the checkpoint"
    assert identity_of(checkpoint) == before


def test_the_roles_are_read_from_the_names(tmp_path):
    found = Checkpoint.at(a_checkpoint(tmp_path))
    assert found is not None
    assert found.has("weights", "config", "optimizer", "rng", "step")
    assert found.missing("nonexistent") == ["nonexistent"]
    assert found.role("weights").name == "model.safetensors"
    # `optimizer_state.safetensors` is the optimizer and not a second copy of the weights.
    assert "training_state/optimizer_state.safetensors" in found.roles["optimizer"]
    assert "training_state/optimizer_state.safetensors" not in found.roles["weights"]


def test_a_name_that_says_nothing_is_still_part_of_the_checkpoint(tmp_path):
    """A file nobody named is a file that exists, and dropping it would make two different
    checkpoints equal -- which is the failure this module was written to remove."""
    checkpoint = a_checkpoint(tmp_path)
    before = identity_of(checkpoint)
    (checkpoint / "notes.txt").write_text("a human wrote this")
    assert identity_of(checkpoint) != before
    assert Checkpoint.at(checkpoint).other == ("notes.txt",)


def test_a_pointer_is_not_content(tmp_path):
    """`last` points at the newest step. Hashing it would make two checkpoints equal whenever
    they point at the same place, which is the opposite of what an identity is for."""
    checkpoint = a_checkpoint(tmp_path)
    before = identity_of(checkpoint)
    (checkpoint.parents[1] / "last").symlink_to(checkpoint.parent)
    assert identity_of(checkpoint) == before


def test_a_path_with_no_checkpoint_has_no_identity(tmp_path):
    assert identity_of(tmp_path / "nothing") is None
    assert Checkpoint.at(tmp_path / "nothing") is None


def test_a_narrower_identity_can_be_asked_for_and_says_what_it_covers(tmp_path):
    """A caller that only wants the weights can have them, but has to say so."""
    checkpoint = a_checkpoint(tmp_path)
    weights_only = identity_of(checkpoint, roles=("weights",))
    everything = identity_of(checkpoint)
    (checkpoint / "training_state/optimizer_state.safetensors").write_bytes(b"OPTIMIZER-3")
    # The narrow one covers what it says it covers, and nothing else moved.
    assert identity_of(checkpoint, roles=("weights",)) == weights_only
    # And the full one is a different question, so it answers differently.
    assert identity_of(checkpoint) != everything


def test_role_names_are_matched_by_what_they_mean(tmp_path):
    assert role_of("model.safetensors") == "weights"
    assert role_of("ckpt-0005.pt") == "weights"
    assert role_of("optimizer_state.safetensors") == "optimizer"
    assert role_of("rng_state.safetensors") == "rng"
    assert role_of("training_step.json") == "step"
    assert role_of("train_config.yaml") == "config"
    assert role_of("something_else.bin") == "weights"          # a bin is still weights
    assert role_of("README.md") == "other"

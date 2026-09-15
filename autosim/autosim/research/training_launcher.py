"""Explicit distributed-training contracts and bounded launch commands."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingContract:
    global_batch: int
    world_size: int = 1
    gradient_accumulation: int = 1
    optimizer_updates: int = 100
    precision: str = "fp32"
    trainer_digest: str = ""
    verified_contract_digest: str = ""

    def __post_init__(self):
        if min(self.global_batch, self.world_size, self.gradient_accumulation, self.optimizer_updates) < 1:
            raise ValueError("training counts must be positive")
        if self.global_batch % (self.world_size * self.gradient_accumulation):
            raise ValueError("global batch cannot be preserved")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("unknown precision")

    @property
    def micro_batch(self):
        return self.global_batch // (self.world_size * self.gradient_accumulation)

    def as_dict(self):
        return {**asdict(self), "micro_batch": self.micro_batch}


def launch_command(python: Path, script: Path, args: list[str], contract: TrainingContract) -> list[str]:
    if contract.world_size == 1:
        return [str(python), str(script), *args]
    if not contract.verified_contract_digest:
        raise ValueError("distributed ACT semantics must be verified before automatic activation")
    return [str(python), "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={contract.world_size}", str(script), "--distributed", *args]

"""Benchmark registry and candidate generation for AutoSim."""

from autosim.benchmark.spec import BenchmarkSpec
from autosim.benchmark.registry import load_benchmark_spec, default_robotwin_spec
from autosim.benchmark.generator import BenchmarkCandidateGenerator
from autosim.benchmark.llm_generator import LLMBenchmarkCandidateGenerator

__all__ = [
    "BenchmarkSpec",
    "BenchmarkCandidateGenerator",
    "LLMBenchmarkCandidateGenerator",
    "default_robotwin_spec",
    "load_benchmark_spec",
]

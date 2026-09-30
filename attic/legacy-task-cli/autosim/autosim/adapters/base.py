"""
TaskAdapter — 通用任务适配器基类

任何仿真/训练项目只需实现这个接口，AutoSim 就能自动优化。
对标 AutoSOTA 的 config.yaml → eval_command 抽象。
"""

import os
import textwrap
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ParamDef:
    """超参定义"""
    name: str
    default: float
    range: Tuple[float, float]
    description: str = ""
    dtype: str = "float"  # float, int, bool, categorical


@dataclass
class EvalResult:
    """评估结果"""
    score: float             # 主指标 (越低/越高越好)
    success: bool = True     # 评估是否成功
    metrics: Dict = field(default_factory=dict)  # 辅助指标
    info: str = ""           # 诊断信息


@dataclass
class Candidate:
    """系统级仿真候选。

    Candidate 是 AutoSim 系统调度的基本对象，可以代表 checkpoint、
    参数配置、训练配方、控制器、策略文件或外部服务端点。adapter 负责解释 payload。

    常见 kind:
    - checkpoint: payload.ckpt_dir 指向已有 checkpoint，直接仿真评测
    - params: payload 是一组参数，adapter.evaluate(params)
    - train_recipe: payload.params 定义训练参数，adapter 先训练再评测
    - patch/architecture: payload 可包含源码 patch 或架构描述
    """
    name: str
    kind: str = "checkpoint"
    payload: Dict = field(default_factory=dict)
    description: str = ""


class TaskAdapter(ABC):
    """
    通用任务适配器 — AutoSOTA 三层优化接口。

    PARAM 层: evaluate(params) 传入数字参数
    CODE/ALGO 层: apply_change(old, new) 传入精确代码 diff
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._backup_files = {}  # 代码变更备份，用于 revert

    # ── PARAM 层（超参优化）────────────────────────────────

    @abstractmethod
    def get_param_space(self) -> Dict[str, ParamDef]:
        """返回可优化超参空间"""
        ...

    @abstractmethod
    def evaluate(self, params: Dict = None) -> EvalResult:
        """
        评估当前代码 + 参数。

        params: 超参覆盖（None = 用默认值）
        返回: EvalResult(score, metrics, info)
        """
        ...

    @abstractmethod
    def get_source_files(self) -> Dict[str, str]:
        """返回 {文件路径: 文件内容}，供 LLM 分析"""
        ...

    # ── CODE/ALGO 层（代码优化）────────────────────────────

    def apply_change(self, file_path: str, old_code: str, new_code: str) -> bool:
        """
        应用代码变更。精确字符串匹配替换。
        自动备份原文件，可通过 revert() 恢复。
        """
        full_path = self._resolve_change_path(file_path)
        if full_path is None:
            return False

        source = full_path.read_text()
        new_source = self._replace_source_block(source, old_code, new_code)
        if new_source is None:
            return False

        # 备份
        backup_key = str(full_path)
        if backup_key not in self._backup_files:
            self._backup_files[backup_key] = source

        # 应用
        full_path.write_text(new_source)
        return True

    def _replace_source_block(
        self,
        source: str,
        old_code: str,
        new_code: str,
    ) -> Optional[str]:
        """Replace exact or indentation-normalized code blocks."""
        if old_code in source:
            return source.replace(old_code, new_code, 1)

        old_block = textwrap.dedent(old_code).strip("\n")
        new_block = textwrap.dedent(new_code).strip("\n")
        if not old_block or "..." in old_block:
            return None

        if old_block in source:
            return source.replace(old_block, new_block, 1)

        source_lines = source.splitlines(keepends=True)
        old_lines = old_block.splitlines()
        if not old_lines:
            return None

        old_norm = [line.strip() for line in old_lines if line.strip()]
        if not old_norm:
            return None

        offsets = []
        pos = 0
        for line in source_lines:
            offsets.append(pos)
            pos += len(line)

        window = len(old_lines)
        for start_line in range(0, len(source_lines) - window + 1):
            end_line = start_line + window
            chunk = source_lines[start_line:end_line]
            chunk_norm = [line.strip() for line in chunk if line.strip()]
            if chunk_norm != old_norm:
                continue

            indent = self._line_indent(chunk[0])
            replacement = self._indent_block(new_block, indent)
            if chunk and chunk[-1].endswith(("\n", "\r")):
                replacement += "\n"
            start = offsets[start_line]
            end = offsets[end_line] if end_line < len(offsets) else len(source)
            return source[:start] + replacement + source[end:]

        return None

    @staticmethod
    def _line_indent(line: str) -> str:
        return line[:len(line) - len(line.lstrip())]

    @staticmethod
    def _indent_block(block: str, indent: str) -> str:
        return "\n".join(
            f"{indent}{line}" if line.strip() else line
            for line in block.splitlines()
        )

    def _resolve_change_path(self, file_path: str) -> Optional[Path]:
        """Resolve LLM-reported source paths against common project roots."""
        raw = Path(file_path)
        if raw.is_absolute() and raw.exists():
            return raw

        repo = Path(self.repo_path) if self.repo_path else Path.cwd()
        package_root = Path(__file__).resolve().parents[1]
        project_root = package_root.parent

        candidates = [
            repo / file_path,
            repo / "autosim" / file_path,
            repo / "autosim" / "adapters" / file_path,
            repo / "autosim" / "tasks" / file_path,
            project_root / file_path,
            project_root / "autosim" / file_path,
            project_root / "autosim" / "adapters" / file_path,
            project_root / "autosim" / "tasks" / file_path,
            package_root / file_path,
            package_root / "adapters" / file_path,
            package_root / "tasks" / file_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def revert_all(self):
        """恢复所有被修改的文件到原始状态"""
        for path, original in self._backup_files.items():
            Path(path).write_text(original)
        self._backup_files.clear()

    def accept_all_changes(self):
        """接受当前代码变更，后续 revert 只回滚新的试验改动。"""
        self._backup_files.clear()

    def get_modified_files(self) -> list:
        """返回被修改过的文件列表"""
        return list(self._backup_files.keys())

    # ── 元信息 ─────────────────────────────────────────────

    def get_metric_direction(self) -> str:
        return "lower"

    def get_primary_metric_name(self) -> str:
        return "score"

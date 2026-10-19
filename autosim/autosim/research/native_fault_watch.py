"""Bounded engineering-fault watch, independent of scores or benchmark names."""
import re
from pathlib import Path


class NativeFaultWatch:
    def __init__(self, log: Path):
        self.log = log
        self.detected = None

    def __call__(self, control: dict | None = None) -> dict:
        value = dict(control or {})
        if self.detected:
            return {**value, 'cancelled': True}
        try:
            if self.log.is_symlink():
                return value
            with self.log.open('rb') as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - 16384))
                text = stream.read(16384).decode(errors='replace')
            text = re.sub(r'\x1b\[[0-9;]*m', '', text)
            header = re.search(r'Assertion at\s*:\s*([^\n]+?):(\d+)\s*:', text)
            frame = re.search(r'#\d+\s*:\s*([A-Za-z_][A-Za-z0-9_:]*)\(', text)
            if header and frame and 'assert failed!' in text:
                self.detected = {'kind': 'native_assertion_with_stack',
                    'source': Path(header[1]).name, 'line': int(header[2]), 'function': frame[1],
                    'scope': 'engineering diagnostic; exposure and repair require independent review'}
                return {**value, 'cancelled': True}
        except OSError:
            pass  # Missing transient log is not evidence of failure.
        return value

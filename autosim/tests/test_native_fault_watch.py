import subprocess
import sys
import time

from autosim.research.native_fault_watch import NativeFaultWatch
from autosim.research.process_executor import run_process


def test_stuck_native_assertion_is_cancelled_without_waiting_for_total_budget(tmp_path):
    log = tmp_path / 'native.log'
    watch = NativeFaultWatch(log)
    code = 'import time; print("Assertion at : /sdk/CoreString.h:94 :\\nassert failed!\\n #0 :sdk::Material::Create(char const*) at 0x123",flush=True); time.sleep(20)'
    started = time.monotonic()
    deadline = time.time() + 15
    with log.open('w') as stream:
        result = run_process([sys.executable, '-c', code], cwd=tmp_path, env={}, timeout=15,
                             stdout=stream, stderr=subprocess.STDOUT,
                             live_control=lambda: watch({'deadline_epoch': deadline}))
    assert result.cancelled and time.monotonic() - started < 4
    assert watch.detected['source'] == 'CoreString.h'
    assert watch.detected['scope'].startswith('engineering diagnostic')


def test_warning_or_standalone_assert_text_does_not_cancel(tmp_path):
    log = tmp_path / 'native.log'
    log.write_text('warning: assert failed! is a string in the tutorial\n')
    watch = NativeFaultWatch(log)
    assert watch({'deadline_epoch': 123}) == {'deadline_epoch': 123}
    assert watch.detected is None

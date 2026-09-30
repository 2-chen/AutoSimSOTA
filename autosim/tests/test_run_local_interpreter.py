"""A provided interpreter must not turn provisioning into a shared-environment write."""

import subprocess
import sys

from autosim.research.derive_and_run import _run_local_interpreter


def test_parent_venv_packages_remain_visible_from_run_local_child(tmp_path):
    parent = tmp_path / "parent"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(parent)],
                   check=True, timeout=60)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    parent_site = parent / "lib" / version / "site-packages"
    (parent_site / "autosim_parent_marker.py").write_text("MARKER = 73\n", encoding="utf-8")
    output = tmp_path / "run"
    child = _run_local_interpreter(parent / "bin" / "python", output)
    done = subprocess.check_output(
        [str(child), "-c", "import autosim_parent_marker,sys; "
         "print(sys.prefix); print(autosim_parent_marker.MARKER)"], text=True, timeout=15)
    assert done.splitlines() == [str(output / "env"), "73"]
    assert str(parent_site) in (output / "env" / "lib" / version / "site-packages" /
                                "autosim_parent_site.pth").read_text(encoding="utf-8")

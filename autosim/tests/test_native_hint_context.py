"""Prepared dependencies must be readable without pretending the simulator passed."""
import subprocess
import sys
from pathlib import Path

from autosim.research.common import atomic_json
from autosim.research.native_context import publish_interpreter_hint, load_context
from autosim.research.prepare import Preparation
from tests.test_prepare import _Client


def test_run_local_hint_publishes_cpu_context_without_environment_readiness(tmp_path):
    output=tmp_path/'run';repo=output/'checkout';repo.mkdir(parents=True)
    prefix=output/'env'
    subprocess.run([sys.executable,'-m','venv','--copies','--without-pip',str(prefix)],check=True)
    atomic_json(output/'workspace_snapshot.json',{'destination':str(repo)})
    real=Preparation(repo=repo,output=output,client=_Client([]),scouting=output/'scouting',
                     interpreter_hint=prefix/'bin/python')
    assert real.native_hint_publication['status']=='diagnostic_hint_published'
    assert load_context(output,repo)['interpreter']==str(prefix/'bin/python')
    assert real.interpreter is None and not (output/'environment.json').exists()
    assert not real.stages and not real.declaration
    assert real.native_hint_publication['gpu_access'] is False


def test_external_hint_is_not_a_host_prefix_grant_or_success(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();output=tmp_path/'run';output.mkdir()
    result=publish_interpreter_hint(output,repo,Path(sys.executable))
    assert result['status']=='hint_not_run_owned'
    assert not (output/'native_context.json').exists()


def test_hint_does_not_overwrite_an_existing_context(tmp_path):
    atomic_json(tmp_path/'native_context.json',{'preserve':'existing context'})
    result=publish_interpreter_hint(tmp_path,tmp_path,None)
    assert result['status']=='existing_context_preserved'

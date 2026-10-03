import pytest

from autosim.research.operation_contracts import schema, validate
from tests.test_prepare import preparation


def test_public_schema_is_generated_from_handler_and_rejects_type_errors(tmp_path):
    real=preparation(tmp_path)
    held=schema(real._step_submit_native_job)
    assert 'idea_label' in held['properties']
    with pytest.raises(ValueError):validate(held,dict(stage='train',window_seconds=True,reason='test'))
    with pytest.raises(ValueError):validate(held,dict(stage='train',window_seconds=4,reason='x'*1001))
    validate(held,dict(stage='train',window_seconds=4,reason='test',idea_label='audited'))


def test_invalid_job_arguments_do_not_enter_native_handler(tmp_path,monkeypatch):
    real=preparation(tmp_path)
    real.client.supports_main_agent=True
    result=real._do_operation('submit_native_job',stage='train',window_seconds=True,reason='bad')
    assert result['failure_category']=='operation_arguments_invalid'
    assert not (real.output/'native_jobs').exists()

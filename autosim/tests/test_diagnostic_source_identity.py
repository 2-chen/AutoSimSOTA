from autosim.research.diagnostic_source_identity import source_identity


def test_source_and_config_changes_invalidate_but_resource_content_is_not_read(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    external=tmp_path/'resource';external.mkdir()
    (external/'data.json').write_text('1')
    (repo/'data').symlink_to(external, target_is_directory=True)
    code=repo/'policy.py';code.write_text('chunk = 8')
    config=repo/'train.yaml';config.write_text('steps: 100')
    original=source_identity(repo)
    (external/'data.json').write_text('2')
    assert source_identity(repo)==original
    code.write_text('chunk = 32')
    changed=source_identity(repo)
    assert changed!=original
    config.write_text('steps: 200')
    assert source_identity(repo)!=changed
    (repo/'data').unlink();(repo/'data').symlink_to(tmp_path/'missing')
    assert source_identity(repo)!=changed


def test_bounded_unknown_or_symlinked_repository_is_not_certified(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();(repo/'policy.py').write_text('x=1')
    assert source_identity(repo,max_entries=0) is None
    assert source_identity(repo,max_bytes=2) is None
    alias=tmp_path/'alias';alias.symlink_to(repo, target_is_directory=True)
    assert source_identity(alias) is None
    assert source_identity(tmp_path/'absent') is None


def test_cache_changes_do_not_invalidate_source(tmp_path):
    (tmp_path/'policy.py').write_text('x=1')
    original=source_identity(tmp_path)
    cache=tmp_path/'__pycache__';cache.mkdir();(cache/'ignored.py').write_text('cache')
    assert source_identity(tmp_path)==original

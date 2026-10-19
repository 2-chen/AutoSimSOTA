"""Recheck a declaration against the SAME sealed evaluation, never cache a score.

Scope is one make_runnable call. Changed commands, code, policy, environment or
inputs require a new execution. Training and failed/partial executions never reuse.
"""
import re
from pathlib import Path

from .common import digest, object_digest, read_json


def file_manifest(root, *, limit_bytes=64*1024**2):
    root = Path(root)
    if root.is_symlink() or not root.exists():
        raise ValueError('missing or linked recheck input')
    paths = [root] if root.is_file() else root.rglob('*')
    files, total = [], 0
    for index, path in enumerate(paths):
        if index >= 512 or path.is_symlink():
            raise ValueError('unbounded or linked recheck input')
        if path.is_file():
            total += path.stat().st_size
            if total > limit_bytes:
                raise ValueError('oversized recheck input')
            files.append(path)
    if not files:
        raise ValueError('empty or oversized recheck input')
    return [(p.name if root.is_file() else str(p.relative_to(root)), digest(p)) for p in sorted(files)]


class ArtifactRecheck:
    def __init__(self, stage, repo, run_root):
        self.stage, self.repo, self.root = stage, Path(repo).resolve(), Path(run_root).resolve()
        self.saved = None

    def identity(self, argv, row, inputs, environment, directory, stdin):
        from .source_tracking import materialized_files
        source = [(str(p),digest(self.repo/p)) for p in materialized_files(self.repo)]
        policy = None
        if inputs.get('checkpoint'):
            checkpoint = Path(inputs['checkpoint'])
            if not checkpoint.resolve().is_relative_to(self.root):
                raise ValueError('recheck requires a run-owned policy input')
            policy = file_manifest(checkpoint,limit_bytes=512*1024**2)
        context = None
        if (self.root/'native_context.json').exists():
            from .native_context import load_context
            native = load_context(self.root,self.repo)
            context = native['identity']
            overlay = Path(native['interpreter']).parent.parent/'overlay.json'
            if overlay.is_file():
                # Explicit external SDK bindings are code too. A source edit with
                # unchanged distribution metadata must not reuse an old rollout.
                bound = read_json(overlay).get('bindings') or []
                external, count, total = [], 0, 0
                for binding in bound:
                    if binding['origin']=='checkout':
                        continue
                    target=Path(binding['path'])
                    paths=[target] if target.is_file() else target.rglob('*')
                    for path in paths:
                        count += 1
                        if count>10000 or path.is_symlink():
                            raise ValueError('external source identity incomplete')
                        if path.is_file() and path.suffix in {'.py','.so','.json','.yaml','.yml','.toml','.cfg'}:
                            total += path.stat().st_size
                            if total>256*1024**2:
                                raise ValueError('external source exceeds recheck identity budget')
                            external.append((binding['module'],str(path.relative_to(target))
                                if target.is_dir() else path.name,digest(path)))
                source += [('external-source',object_digest(sorted(external)))]
        output = str(inputs.get('output') or '')
        # Only normalize this invocation's declared output namespace. All other
        # argv/input/environment values remain exact, including policy and seed.
        normalize = lambda s: re.sub(re.escape(output)+r'(?=/|$)', '<OUTPUT>', str(s)) if output else str(s)
        return object_digest({'stage':self.stage,'argv':[normalize(a) for a in argv],
            'row':{k:v for k,v in row.items() if k!='artifact'},
            'inputs':{k:v for k,v in inputs.items() if k!='output'},
            'environment':environment,'directory':str(directory),'stdin':stdin,
            'source':source,'policy':policy,'native_context':context})

    def capture(self, done, started, argv, row, inputs, environment, directory, stdin, *, before):
        if self.stage != 'evaluate' or done.returncode != 0:
            return
        try:
            ref = str(getattr(done,'evidence_ref',''))
            if not re.fullmatch(r'evidence/[0-9a-f]{32}\.json',ref):
                return
            from .evidence_store import read_attempt_evidence
            identity = Path(ref).stem
            evidence = read_attempt_evidence(self.root,identity,limit=1)
            receipt = self.root/'derivation_native'/identity/'receipt.json'
            record = read_json(receipt)
            if (receipt.is_symlink() or record.get('id') != identity or
                    record.get('status')!='completed' or record.get('argv')!=list(argv) or
                    record.get('cwd')!=str(directory) or evidence.get('returncode')!=0):
                return
            output = Path(inputs['output'])
            if (not output.resolve().is_relative_to(self.root/'verification_outputs') or
                    record.get('verification_output') != str(output)):
                return
            after = self.identity(argv,row,inputs,environment,directory,stdin)
            if before is None or before != after:
                return  # Native/staging edits cannot be certified retroactively.
            self.saved = {'done':done,'started':started,'output':str(output),
                'identity':after,
                'artifact':row.get('artifact'),
                'files':file_manifest(output),'receipt':receipt,'receipt_sha256':digest(receipt)}
        except (OSError,ValueError,KeyError,TypeError):
            self.saved = None  # Failure to prove reuse never relaxes native execution.

    def lookup(self, argv, row, inputs, environment, directory, stdin):
        saved = self.saved
        if not saved:
            return None
        if row.get('artifact') == saved['artifact']:
            return None  # Not a declaration correction; no implicit score-repeat cache.
        try:
            if (self.identity(argv,row,inputs,environment,directory,stdin)!=saved['identity'] or
                    digest(saved['receipt'])!=saved['receipt_sha256'] or
                    file_manifest(saved['output'])!=saved['files']):
                return None
            from .evidence_store import read_attempt_evidence
            if read_attempt_evidence(self.root,Path(saved['done'].evidence_ref).stem,limit=1)['returncode']!=0:
                return None
            return saved
        except (OSError,ValueError,KeyError,TypeError):
            return None

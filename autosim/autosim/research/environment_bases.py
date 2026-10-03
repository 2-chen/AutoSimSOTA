"""Prevalidated runtime families, not benchmark-specific installation rules.

Checks run in a disposable overlay through the normal native sandbox/lease boundary.
No package is downloaded, no source environment is modified, and a base certificate
never replaces the current repository's consumer probes.
"""
from __future__ import annotations

import json
import math
import shlex
import subprocess
from pathlib import Path

from .common import atomic_json, atomic_text, digest, now, object_digest, read_json

RESULT = 'AUTOSIM_BASE_RESULT='
CAPABILITIES = {
    'python-toolchain': ['python_subprocess', 'local_io'],
    'torch-cuda': ['cuda_compute', 'backward_optimizer', 'weights_roundtrip'],
    'mujoco-headless': ['reset', 'physics_step', 'offscreen_frame'],
    'sapien-headless': ['scene_initialize', 'physics_step', 'offscreen_frame'],
}

PYTHON = '''import sys, json, subprocess, tempfile, pathlib
assert sys.version_info >= (3, 8)
with tempfile.TemporaryDirectory() as directory:
    target = pathlib.Path(directory) / 'check'
    target.write_text('ok')
    assert target.read_text() == 'ok'
assert subprocess.check_output([sys.executable, '-I', '-S', '-c', 'print(3)'], text=True).strip() == '3'
result = {'python': sys.version.split()[0], 'capabilities': ['python_subprocess', 'local_io']}
'''

TORCH = '''import torch, io
assert torch.cuda.is_available(), 'CUDA unavailable: import success is not GPU readiness'
torch.manual_seed(0)
model = torch.nn.Linear(2, 1).cuda()
x, y = torch.ones(8, 2, device='cuda'), torch.ones(8, 1, device='cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
before = ((model(x) - y) ** 2).mean().item()
for _ in range(4):
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
assert ((model(x) - y) ** 2).mean().item() < before
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
buffer.seek(0)
restored = torch.nn.Linear(2, 1).cuda()
restored.load_state_dict(torch.load(buffer, weights_only=True))
assert torch.allclose(model(x), restored(x))
result = {'torch': torch.__version__, 'cuda': torch.version.cuda,
          'device': torch.cuda.get_device_name(),
          'capabilities': ['cuda_compute', 'backward_optimizer', 'weights_roundtrip']}
'''

MUJOCO = '''import mujoco, numpy as np
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1"/><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" rgba="1 .2 .2 1"/></body></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
initial = data.qpos.copy()
for _ in range(5):
    mujoco.mj_step(model, data)
assert data.time > 0 and np.isfinite(data.qpos).all() and not np.array_equal(initial, data.qpos)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    frame = renderer.render().copy()
assert frame.shape == (64, 64, 3) and np.isfinite(frame).all() and frame.max() > frame.min()
save_frame(frame)
result = {'mujoco': mujoco.__version__, 'capabilities': ['reset', 'physics_step', 'offscreen_frame']}
'''

SAPIEN2 = '''import sapien.core as sapien, numpy as np
engine = sapien.Engine()
engine.set_renderer(sapien.SapienRenderer(offscreen_only=True))
scene = engine.create_scene()
scene.set_timestep(0.01)
scene.add_ground(0)
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[.1, .1, .1])
builder.add_box_visual(half_size=[.1, .1, .1], color=[1, .2, .2])
actor = builder.build()
actor.set_pose(sapien.Pose(p=[0, 0, 1]))
camera = scene.add_camera('check', 64, 64, 1.0, .01, 10)
camera.set_pose(sapien.Pose(p=[-2, 0, 1]))
'''

SAPIEN3 = '''import sapien, numpy as np
import os
ordinal = os.environ.get('AUTOSIM_DEFAULT_CUDA_ORDINAL', '0')
renderer = sapien.render.RenderSystem(device='cuda:' + ordinal)
scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), renderer])
scene.set_timestep(0.01)
scene.add_ground(0)
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[.1, .1, .1])
builder.add_box_visual(half_size=[.1, .1, .1], material=[1, .2, .2, 1])
builder.set_initial_pose(sapien.Pose(p=[0, 0, 1]))
actor = builder.build()
camera = scene.add_camera('check', 64, 64, 1.0, .01, 10)
camera.entity.set_pose(sapien.Pose(p=[-2, 0, 1]))
'''

SAPIEN_END = '''scene.set_ambient_light([.5, .5, .5])
initial = actor.get_pose().p.copy()
for _ in range(5):
    scene.step()
assert np.isfinite(actor.get_pose().p).all() and not np.array_equal(initial, actor.get_pose().p)
scene.update_render()
camera.take_picture()
frame = FRAME_EXPRESSION
assert frame.shape == (64, 64, 4) and np.isfinite(frame).all()
assert frame[:, :, :3].max() > frame[:, :, :3].min()
save_frame((frame[:, :, :3].clip(0, 1) * 255).astype('uint8'))
result = {'sapien': sapien.__version__, 'capabilities': ['scene_initialize', 'physics_step', 'offscreen_frame']}
'''

# PNG output needs no Pillow/matplotlib dependency and is preserved as real evidence.
FRAME_WRITER = '''import pathlib, struct, zlib
def save_frame(frame):
    height, width, channels = frame.shape
    assert channels == 3 and str(frame.dtype) == 'uint8'
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    scanlines = b''.join(b'\\0' + frame[row].tobytes() for row in range(height))
    pathlib.Path('frame.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))
'''


def probe_spec(profile: str, row: dict) -> dict:
    """Version variants describe public engine APIs, never repository names."""
    if profile == 'python-toolchain':
        code, resource, variant = PYTHON, 'cpu', 'python3'
    elif profile == 'torch-cuda':
        code, resource, variant = TORCH, 'gpu', 'torch2'
    elif profile == 'mujoco-headless':
        code, resource, variant = MUJOCO, 'gpu', 'mujoco-python'
    elif profile == 'sapien-headless':
        version = str(row['packages'].get('sapien', ''))
        if version.startswith('2.'):
            code, variant = SAPIEN2 + SAPIEN_END.replace('FRAME_EXPRESSION', "camera.get_float_texture('Color')"), 'sapien2'
        elif version.startswith('3.'):
            code, variant = SAPIEN3 + SAPIEN_END.replace('FRAME_EXPRESSION', "camera.get_picture('Color')"), 'sapien3'
        else:
            raise ValueError('no supported SAPIEN 2/3 version in this base; select another base or add a reviewed engine probe')
        resource = 'gpu'
    else:
        raise ValueError('unknown runtime profile')
    code = 'import json\n' + FRAME_WRITER + code + f'print({RESULT!r} + json.dumps(result))\n'
    return {'profile': profile, 'variant': variant, 'code': code, 'resource': resource,
            'capabilities': CAPABILITIES[profile],
            'identity': object_digest([profile, variant, code, resource])}


def base_identity(prefix: Path) -> dict:
    from .environment_pool import describe
    from .environment_overlay import inspect
    row = describe(prefix)
    return {'fingerprint': row['fingerprint'],
            'dependency_fingerprint': inspect(prefix, row['python'])['fingerprint'],
            'interpreter_sha256': digest(prefix / 'bin/python')}


def machine_identity(machine: dict, *, gpu: bool = True) -> str:
    import platform
    keys = ('os', 'release', 'gpus', 'cuda_toolkit') if gpu else ('os', 'release')
    return object_digest({'architecture': platform.machine(), **{k: machine.get(k) for k in keys}})


def verification_view(prefix: Path, proofs: list, machine: dict) -> dict:
    """Recheck identities, contract and sealed evidence; stale proof is only history."""
    from .environment_pool import describe
    from .evidence_store import read_attempt_evidence
    status, capabilities, summaries, invalid = 'unverified', set(), [], []
    try:
        identity = base_identity(prefix)
        row = describe(prefix)
    except (OSError, ValueError, KeyError, TypeError):
        return {'status': 'stale', 'capabilities': [], 'checks': []}
    if not isinstance(proofs, list):
        return {'status': 'stale', 'capabilities': [], 'checks': []}
    checked_profiles = set()
    for reference in reversed(proofs[-8:]):
        try:
            path = Path(reference)
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
                raise ValueError('unsafe base verification')
            proof = read_json(path)
            payload = proof['payload']
            if object_digest(payload) != proof['identity']:
                raise ValueError('base verification changed')
            if payload['profile'] in checked_profiles:
                continue  # Older successes stay on disk, not in the current ability list.
            checked_profiles.add(payload['profile'])
            spec = probe_spec(payload['profile'], row)
            if (payload['base_identity'] != identity
                    or payload['probe_identity'] != spec['identity'] or payload['prefix'] != str(prefix.absolute())):
                raise ValueError('base or probe contract changed')
            if sorted(payload['result']['capabilities']) != sorted(spec['capabilities']):
                raise ValueError('capability result does not match probe contract')
            receipt_path = path.parent / payload['receipt_ref']
            if receipt_path.is_symlink() or not receipt_path.resolve().is_relative_to(path.parent.resolve()):
                raise ValueError('unsafe base receipt')
            if digest(receipt_path) != payload['receipt_sha256']:
                raise ValueError('base receipt changed')
            receipt = read_json(receipt_path)
            sealed = read_attempt_evidence(path.parent, payload['evidence_id'], limit=1)
            expected_command = shlex.quote(str(path.parent / 'env/bin/python')) + ' ' + shlex.quote(str(path.parent / 'checkout/probe.py'))
            script = path.parent / 'checkout/probe.py'
            if (receipt.get('ok') is not True or receipt.get('returncode') != 0
                    or receipt.get('evidence_id') != payload['evidence_id']
                    or receipt.get('command') != expected_command
                    or script.is_symlink() or script.read_text() != spec['code']
                    or sealed['receipt_ref'] != payload['receipt_ref']
                    or sealed['returncode'] != 0
                    or (spec['resource'] == 'gpu' and (receipt.get('gpu_access') is not True or not receipt.get('gpu_lease_id')))):
                raise ValueError('probe did not succeed with required resources')
            frame = payload.get('frame')
            if 'offscreen_frame' in spec['capabilities'] and not frame:
                raise ValueError('renderer check has no recorded frame')
            frame_path = path.parent / 'checkout/frame.png'
            if frame and (frame_path.is_symlink() or digest(frame_path) != frame['sha256']):
                raise ValueError('recorded frame changed')
            if spec['resource'] == 'gpu' and machine.get('gpu_query_error'):
                raise ValueError('current GPU identity unavailable; historical certificate retained, not proof of absent hardware')
            if payload['machine_identity'] != machine_identity(machine, gpu=spec['resource'] == 'gpu'):
                raise ValueError('machine/driver identity changed; current runtime needs revalidation')
            capabilities.update(payload['result']['capabilities'])
            summaries.append({'profile': payload['profile'], 'variant': spec['variant'],
                              'proof_id': proof['identity'], 'evidence_id': payload['evidence_id'],
                              'versions': {key: value for key, value in payload['result'].items()
                                           if key in {'python', 'torch', 'cuda', 'device', 'mujoco', 'sapien'}}})
            status = 'verified'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append({'reason': str(exc)[:250] if isinstance(exc, ValueError) else type(exc).__name__})
            if status != 'verified':
                status = 'stale'
    return {'status': status, 'capabilities': sorted(capabilities), 'checks': summaries,
            'invalid_checks': invalid,
            'scope': 'runtime_family_only; current_repository_probes_required; live_not_byte_frozen'}


def verify(prefix: Path, *, profile: str, output: Path, store: Path,
           wall_seconds: float = 180, gpu_seconds: float = 0, label: str = 'verified-base') -> dict:
    """Operator maintenance: explicit bounded resources, no LLM or downloads."""
    from . import environment_pool as pool, environment_overlay as overlay, provision
    from .budget import RunBudget
    from .common import run_local_environment
    from .native_context import publish_context
    from .task_budget import TaskGPUBudget
    prefix, output, store = prefix.absolute(), output.absolute(), store.resolve()
    for value in (wall_seconds, gpu_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('base check budgets must be finite numbers')
    if not 0 < wall_seconds <= 600 or not 0 <= gpu_seconds <= 600:
        raise ValueError('base checks are short maintenance operations, capped at 600 seconds')
    if (output.exists() or output.is_symlink() or output.resolve().is_relative_to(prefix.resolve())
            or prefix.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(store)
            or store.is_relative_to(output.resolve())):
        raise ValueError('verification needs a fresh output disjoint from base and environment store')
    if (prefix / 'overlay.json').exists():
        raise ValueError('verify the underlying base; a run overlay is not an independent runtime base')
    chosen = pool.describe(prefix)
    spec = probe_spec(profile, chosen)
    if spec['resource'] == 'gpu' and gpu_seconds <= 0:
        raise ValueError('GPU/render checks require explicit --gpu-seconds; no implicit 24-hour allowance')
    output.mkdir(parents=True)
    repo = output / 'checkout'
    repo.mkdir()
    budget = RunBudget(output, wall_seconds=wall_seconds)
    initial_identity = base_identity(prefix)
    machine = provision.platform_facts(timeout=min(5, wall_seconds))
    atomic_text(repo / 'probe.py', spec['code'])
    result = {'schema_version': 1, 'status': 'failed', 'profile': profile,
              'scope': 'runtime_family_not_benchmark', 'started_at': now()}
    try:
        overlay.create(chosen, prefix=output / 'env', output=output, repo=repo,
                       source_binding_ids=[], timeout=max(.001, min(60, budget.remaining())))
        python = output / 'env/bin/python'
        publish_context(output, repo, python, run_local_environment(output))
        env = run_local_environment(output)
        if profile == 'mujoco-headless':
            env['MUJOCO_GL'] = 'egl'
        compute = None
        if spec['resource'] == 'gpu':
            from .compute_decision import decide
            from .common import bounded_run
            def query(argv, **kwargs):
                remaining = budget.remaining()
                if remaining <= 0:
                    raise TimeoutError('base check wall budget exhausted during discovery')
                completed = bounded_run(list(argv), cwd=repo, env=env,
                                        timeout=min(remaining, 15))
                if completed.returncode != 0:
                    return '__error__: ' + completed.stderr[-1000:]
                return completed.stdout
            TaskGPUBudget(output).initialize(repo, cap_seconds=gpu_seconds)
            cuda_table = query([str(python), '-c',
                "import json,torch; print(json.dumps([{'index':i,'uuid':str(torch.cuda.get_device_properties(i).uuid),"
                "'name':torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())]))"])
            try:
                torch_rows = json.loads(cuda_table.strip().splitlines()[-1])
            except (ValueError, IndexError):
                raise ValueError('bounded CUDA discovery failed: ' + cuda_table[-500:]) from None
            compute = decide(python=python, require_gpu=True, prefer='cuda', runner=query, environ=env, torch_rows=torch_rows)
            if profile == 'mujoco-headless':
                env['MUJOCO_EGL_DEVICE_ID'] = str(compute.evidence['selected_device']['index'])
            if (profile == 'mujoco-headless' or spec['variant'] == 'sapien2') and len(machine.get('gpus') or []) != 1:
                raise ValueError('this engine check requires a single physical GPU; multi-GPU renderer identity needs a reviewed probe')
        if budget.remaining() <= 0:
            raise TimeoutError('base check wall budget exhausted before probe')
        command = shlex.quote(str(python)) + ' ' + shlex.quote(str(repo / 'probe.py'))
        receipt = provision.run(command, env=env, cwd=repo, timeout=budget.remaining(),
                                output=output / 'provision.log', compute=compute)
        result['evidence_id'] = receipt['evidence_id']
        if receipt.get('ok') is not True or receipt.get('returncode') != 0:
            result['reason'] = receipt.get('excerpt') or receipt.get('failure_kind')
        else:
            text = (output / receipt['log_ref']).read_text()
            lines = [line[len(RESULT):] for line in text.splitlines() if line.startswith(RESULT)]
            if len(lines) != 1:
                raise ValueError('successful check missing unique capability result')
            observed = json.loads(lines[0])
            if base_identity(prefix) != initial_identity:
                raise ValueError('base changed during verification')
            from .evidence_store import read_attempt_evidence
            sealed = read_attempt_evidence(output, receipt['evidence_id'], limit=1)
            receipt_ref = sealed['receipt_ref']
            payload = {'prefix': str(prefix), 'base_identity': initial_identity,
                       'machine_identity': machine_identity(machine, gpu=spec['resource'] == 'gpu'), 'profile': profile,
                       'probe_identity': spec['identity'], 'result': observed,
                       'evidence_id': receipt['evidence_id'], 'receipt_ref': receipt_ref,
                       'receipt_sha256': digest(output / receipt_ref)}
            frame = repo / 'frame.png'
            if frame.is_file():
                payload['frame'] = {'sha256': digest(frame), 'bytes': frame.stat().st_size}
            proof_path = output / 'base_verification.json'
            atomic_json(proof_path, {'identity': object_digest(payload), 'payload': payload,
                                    'created_at': now()})
            view = verification_view(prefix, [str(proof_path)], machine)
            if view['status'] != 'verified':
                raise ValueError('sealed check failed publication validation')
            pool.register_base(store, prefix, label=label, proof=proof_path)
            result.update(proof_id=object_digest(payload), **view)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        result['reason'] = f'{type(exc).__name__}: {exc}'
        if not result.get('evidence_id'):
            failure = pool.failed_selection(output, base_id=chosen['id'], error=exc, seconds=0,
                                            operation='base_verify', failure_kind='base_check_preflight')
            result['evidence_id'] = failure['evidence_id']
    finally:
        result['finished_at'] = now()
        result['wall_budget'] = budget.record()
        if (output / 'task_gpu_budget.json').is_file():
            result['gpu_budget'] = TaskGPUBudget(output).snapshot()
        atomic_json(output / 'base_check.json', result)
        write_report(output, result)
    return result


def write_report(output: Path, result: dict) -> None:
    """A small human-readable maintenance receipt, never a scored benchmark RUN."""
    titles = {'python-toolchain': 'Python 基础底座', 'torch-cuda': 'GPU 训练底座',
              'mujoco-headless': 'MuJoCo 离屏仿真底座', 'sapien-headless': 'SAPIEN 离屏仿真底座'}
    passed = result.get('status') == 'verified'
    lines = [f"# {titles.get(result['profile'], result['profile'])}验收", '',
             '检查通过，已加入可复用底座目录。' if passed else '检查未通过，未据此登记可用能力。', '',
             '这是底层运行栈检查，不是目标仓库的实验结果。没有下载或安装依赖，没有调用 LLM；'
             '源环境只读，检查在独立增量环境中执行。', '']
    if passed:
        names = {'cuda_compute': 'GPU 张量运算', 'backward_optimizer': '反向传播与参数更新',
                 'weights_roundtrip': '权重保存后重载并核对输出', 'reset': '重置初始状态',
                 'physics_step': '真实物理步进', 'offscreen_frame': '离屏渲染并保存画面',
                 'scene_initialize': '创建物理场景', 'python_subprocess': 'Python 子进程执行', 'local_io': '隔离目录读写'}
        lines += ['已验证：' + '、'.join(names.get(c, c) for c in result.get('capabilities') or []) + '。', '']
    elif result.get('reason'):
        lines += ['失败原因：', '', '```text', str(result['reason'])[:1200], '```', '']
    gpu = (result.get('gpu_budget') or {}).get('settled_gpu_seconds')
    elapsed = (result.get('wall_budget') or {}).get('elapsed_wall_seconds')
    lines += [f"检查墙钟：{elapsed:.2f} 秒。" if elapsed is not None else '墙钟用量未知。',
              f"GPU 租约占用：{gpu:.2f} 秒。" if gpu is not None else '本次没有取得 GPU 租约。', '',
              '[完整检查回执](base_check.json)', '']
    if result.get('evidence_id'):
        lines += [f"[原生证据](evidence/{result['evidence_id']}.json)", '']
    if passed and (output / 'checkout/frame.png').is_file():
        lines += ['## 实际渲染画面', '', '![本次引擎检查的真实画面，不代表 benchmark 成功率](checkout/frame.png)', '']
    lines += ['后续仍需 Agent 连接当前源码、数据和资产，并验证该仓库的原生入口、任务步进与评测。'
              '当前底座是活的只读引用，不是完整字节冻结镜像；版本、驱动和证据变化需要复验。', '']
    atomic_text(output / 'CHECK.md', '\n'.join(lines))

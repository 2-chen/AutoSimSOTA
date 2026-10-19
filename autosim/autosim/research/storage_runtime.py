"""Cheap filesystem observations; LLMs run only for decisions, never each poll."""
import os
import subprocess
import time
from pathlib import Path

from .common import atomic_json, now, read_json
from .storage_guard import preflight
from .storage_lifecycle import catalog, policy, safe


def inspect_producer(output,*,evidence_id,offset=0,limit=24):
    from .storage_lifecycle import inspection_owner
    if (isinstance(offset,bool) or not isinstance(offset,int) or offset<0 or
            isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=64):
        raise ValueError('producer inspection requires nonnegative offset and limit 1..64')
    owner,native=inspection_owner(output,evidence_id)
    files=[]
    for path in owner.rglob('*'):
        if path.is_file() and not path.is_symlink():
            try:
                safe(output,path.relative_to(output))
                files.append({'ref':path.relative_to(output).as_posix(),'bytes':path.stat().st_size})
            except (OSError,ValueError): continue
        if len(files)>32768:
            raise ValueError('producer inventory exceeds explicit scan budget; no complete-set claim')
    files.sort(key=lambda row:row['ref'])
    stage=(native.get('receipt_facts') or {}).get('stage') or (native.get('receipt_facts') or {}).get('node_id')
    return {'evidence_id':evidence_id,'output_ref':owner.relative_to(output).as_posix(),
            'stage':stage,'read_only':True,'data_payload_gc_eligible':False if stage in {'collect','prepare_data'} else None,
            'total_files':len(files),'offset':offset,'files':files[offset:offset+limit],
            'next_offset':offset+limit if offset+limit<len(files) else None,
            'instruction':'Discovery only. Read native save/resume source before classifying exact complete groups.'}


def snapshot(output,*,registry_offset=0,registry_limit=24):
    if (isinstance(registry_offset,bool) or not isinstance(registry_offset,int) or registry_offset<0 or
            isinstance(registry_limit,bool) or not isinstance(registry_limit,int) or not 1<=registry_limit<=64):
        raise ValueError('registry inspection requires offset >= 0 and limit 1..64')
    output=Path(output).resolve()
    storage=preflight(output,stage='train')
    categories=[]
    for name in ('research','verification_outputs','env','environments','home','report','agent_workers','run_state_blobs'):
        path=output/name
        if path.is_dir() and not path.is_symlink():
            try:
                result=subprocess.run(['du','-s','-B1',str(path)],capture_output=True,text=True,timeout=10)
                if result.returncode==0: categories.append({'ref':name,'allocated_bytes':int(result.stdout.split()[0])})
            except (OSError,ValueError,subprocess.TimeoutExpired): pass
    candidates=[]
    receipts=[*output.glob('research/*/attempts/*/receipt.json'),*output.glob('derivation_native/*/receipt.json')]
    for receipt_path in sorted(receipts,key=lambda p:p.stat().st_mtime,reverse=True):
        receipt=read_json(safe(output,receipt_path.relative_to(output)))
        if receipt.get('stage',receipt.get('node_id'))!='train' or receipt.get('status') not in {'completed','failed'}: continue
        raw=Path(str(receipt.get('output_directory') or receipt.get('verification_output') or ''))
        if not raw.is_absolute() or not raw.resolve().is_relative_to(output): continue
        files=[]
        for path in raw.rglob('*'):
            if path.is_symlink() or not path.is_file(): continue
            if not path.resolve().is_relative_to(output): continue
            try:
                files.append({'ref':path.relative_to(output).as_posix(),'bytes':path.stat().st_size})
            except OSError: continue  # concurrent producer rotation is not a GC decision
            if len(files)>=24: break
        candidates.append({'evidence_id':receipt.get('evidence_id') or receipt.get('attempt_id') or receipt.get('id'),
            'output_ref':raw.relative_to(output).as_posix(),'stage_status':receipt['status'],
            'files':files,'inventory_may_be_truncated':len(files)==24,
            'instruction':'Read native source/counters; independently classify and register complete payload groups.'})
        if len(candidates)>=8: break
    registered=sorted(catalog(output),key=lambda row:row['id'])
    page=[]
    for row in registered[registry_offset:registry_offset+registry_limit]:
        page.append({**row,'file_refs':row['file_refs'][:8],
                     'total_files':len(row['file_refs']),'file_refs_truncated':len(row['file_refs'])>8})
    result={'at':now(),'filesystem':storage,'categories':categories,'producer_candidates':candidates,
            'discovery_limits':{'producers':8,'files_per_producer':24,'sources':'sample only; use explicit reads for complete groups'},
            'authorization':policy(output),'registered_artifacts':page,'registry_offset':registry_offset,
            'registry_total':len(registered),
            'registry_next_offset':registry_offset+registry_limit if registry_offset+registry_limit<len(registered) else None,
            'guidance':'Inspect producer/save/load source. Register exact payload groups; protect final/resume/scored references. '
                       'Use reviewed operational controls or planned GC; never arbitrary shell deletion.'}
    atomic_json(safe(output,'storage/inspection.json'),result)
    return result


def view(output):
    """Small Agent/report projection, without du, payload hashing or a model call."""
    output=Path(output)
    latest=[]
    for path in sorted(output.glob('storage/runtime/*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:3]:
        latest.append(read_json(safe(output,path.relative_to(output))))
    inspection=read_json(safe(output,'storage/inspection.json')) if (output/'storage/inspection.json').exists() else {}
    plans=[read_json(safe(output,p.relative_to(output))) for p in output.glob('storage/plans/*.json')]
    controls=read_json(safe(output,'storage/controls/train.json')) if (output/'storage/controls/train.json').exists() else {}
    return {'authorization':policy(output),'recent_observations':latest,
            'last_inspection_at':inspection.get('at'),'categories':inspection.get('categories',[]),
            'controls':{k:controls.get(k) for k in ('id','controls','checkpoint_bytes','projected_bytes')},
            'transactions':[{k:p.get(k) for k in ('id','status','bytes','artifact_ids')} for p in plans[-16:]],
            'guidance':'On storage pressure choose inspect_storage; read managing-research-artifacts skill. '
                'No filename/age-only cleanup. Operator authorization required for deletion.'}


def report_panel(content,root):
    """Deterministic Chinese storage panel; no model calls or live du scans."""
    root=Path(root)
    output=root.parent.parent if root.parent.name=='research' else root
    if not (output/'storage').is_dir(): return content
    state=view(output)
    gib=lambda value: f'{value/1024**3:.2f} GiB' if isinstance(value,(int,float)) else '未估计'
    recent=state['recent_observations']
    observation=recent[0] if recent else {}
    labels={'normal':'空间充足','pressure':'空间偏紧','critical':'空间危险，已请求停止写入'}
    lines=['## 存储与中间产物','',
           f"当前观测：{labels.get(observation.get('status'),'尚无运行期观测')}。自动回收：{'已授权限定范围' if state['authorization'].get('auto_gc') else '未开启，需要操作者授权'}。",'',
           f"最近观测：{observation.get('at') or '尚无'}；最近分项检查：{state['last_inspection_at'] or '尚无'}。",'',
           '| 项目 | 大小 / 状态 |','|---|---|',
           f"| 最近观测可用空间 | {gib((observation.get('filesystem') or {}).get('free_bytes'))} |",
           f"| Agent 预计本轮新增产物 | {gib(state['controls'].get('projected_bytes'))} |",
           f"| Agent 预计一次保存 | {gib(state['controls'].get('checkpoint_bytes'))} |"]
    for row in state['categories']:
        lines.append(f"| `{row['ref']}`（上次检查时占用） | {gib(row['allocated_bytes'])} |")
    transactions=state['transactions']
    completed=sum(row['bytes'] for row in transactions if row['status']=='completed')
    lines += [f'| 已完成清理事务的登记体积（最近 16 条） | {gib(completed)} |','',
              '优先减少不必要的原生保存与重复副本。保护计分策略、训练终点、续训状态和数据；清理清单保留在 `storage/plans/`。登记体积不等于文件系统实际释放空间。']
    unknown=[p for p in transactions if p['status']=='committing']
    if unknown: lines += ['','存在中断的清理事务，需要先核对，不能重放删除或启动新作业。']
    if observation.get('status')=='critical':
        lines += ['','写入停止可能丢失未保存更新；恢复前需核验现有续训点，而非直接重复整轮训练。']
    start='<!-- AUTOSIM_STORAGE_START -->'
    end='<!-- AUTOSIM_STORAGE_END -->'
    panel=start+'\n'+'\n'.join(lines)+'\n'+end
    if start in content and end in content:
        beginning,rest=content.split(start,1)
        _,ending=rest.split(end,1)
        return beginning+panel+ending
    return content.rstrip()+'\n\n'+panel+'\n'


class StorageWatch:
    def __init__(self,output,*,stage,attempt_id,path,deadline_epoch,checkpoint_bytes=None):
        self.output=Path(output)
        self.stage=stage
        self.path=Path(path)
        self.deadline_epoch=deadline_epoch
        self.checkpoint_bytes=checkpoint_bytes
        self.status='unknown'
        self.critical=False
        self.last=None
        self.record_path=self.output/'storage/runtime'/f'{attempt_id}.json'
        self.attempt_id=attempt_id

    def __call__(self,base=None):
        control=dict(base or {'deadline_epoch':self.deadline_epoch,'cancelled':False})
        # A read-only statvfs check each executor poll can catch fast writers. Durable
        # records are coalesced; no full-directory scan or model call on this path.
        hard=max(64*1024**2, int((self.checkpoint_bytes or 0)*1.25))
        row=preflight(self.path,stage=self.stage,required_bytes=hard)
        status='critical' if row['free_bytes']<hard else 'pressure' if not row['sufficient'] else 'normal'
        self.critical=status=='critical'
        if self.last is None or status!=self.status or time.monotonic()-self.last>=30:
            observation={'at':now(),'stage':self.stage,'attempt_id':self.attempt_id,
                'status':status,'filesystem':row,'next_checkpoint_bytes':self.checkpoint_bytes,
                'action':'cancel producer; unsaved work may be lost; Agent must inspect recovery and choose a remedy' if self.critical else 'observe'}
            try:
                atomic_json(safe(self.output,self.record_path.relative_to(self.output)),observation)
            except OSError as exc:
                observation['persistence_error']=str(exc)[:300]
                self.critical=True
            self.observation=observation
            self.last=time.monotonic()
        self.status=status
        if self.critical:
            control['cancelled']=True
            control['cancellation_reason']='storage_pressure'
        return control

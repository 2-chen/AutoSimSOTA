"""One executable signature supplies Agent argument types and local validation."""
import inspect
import json
import math
import re


# Expose nested shape constraints before an operation spends a Supervisor turn.
# Scientific/source validation remains in the native domain validators, not here.
SOURCE_REFS = {'type':'array','minItems':1,'maxItems':12,'items':{
    'type':'object','required':['ref','quote'],'properties':{
        'ref':{'type':'string','minLength':1},
        'quote':{'type':'string','minLength':1}},'additionalProperties':False}}
ID = {'type':'string','pattern':r'^[A-Za-z0-9_-]{1,64}$'}
REASON = {'type':'string','minLength':1,'maxLength':1000}
BASELINE_DIAGNOSIS = {'type':'object','required':['subject_sha256','decision','checks',
    'source_refs','evidence_ids','next_action','reason'], 'additionalProperties':False,
    'properties':{'subject_sha256':{'type':'string','pattern':r'^[0-9a-f]{64}$'},
        'decision':{'type':'string','enum':['investigate','proceed_local']},
        'checks':{'type':'object','required':['training','data','policy','evaluation'],
                  'properties':{k:REASON for k in ('training','data','policy','evaluation')},
                  'additionalProperties':False},
        'source_refs':SOURCE_REFS,'evidence_ids':{'type':'array','minItems':1,'maxItems':8,
            'uniqueItems':True,'items':{'type':'string','pattern':r'^[0-9a-f]{32}$'}},
        'next_action':REASON,'reason':REASON}}
def _text_shape(limit):
    return {'type':'string','minLength':1,'maxLength':limit}


PLAN_ITEMS = {'type':'array','maxItems':20,'items':_text_shape(1500)}
PLAN = {'type':'object','required':['objective','hypotheses','open_questions','next_actions','evidence_refs'],
    'additionalProperties':False,'maxEncodedBytes':16000,
    'description':'Writable fields only; authority is output metadata. Use main_agent.writable_plan.',
    'properties':{'objective':_text_shape(4000),
        **{k:PLAN_ITEMS for k in ('hypotheses','open_questions','next_actions','evidence_refs')},
        'data_strategy':{'type':'object','additionalProperties':False,'maxEncodedBytes':8000,
            'required':['route','why','targets','producer_to_loader','evidence_refs','next_probe'],
            'properties':{'route':_text_shape(400),'why':_text_shape(1500),'next_probe':_text_shape(1500),
                **{k:PLAN_ITEMS for k in ('targets','producer_to_loader','evidence_refs')}}},
        'budget_strategy':{'type':'object','additionalProperties':False,'maxEncodedBytes':6000,
            'required':['why','allocations','evidence_refs','revisit_condition','next_action'],
            'properties':{'why':_text_shape(1500),'revisit_condition':_text_shape(1000),
                'next_action':_text_shape(1000),'allocations':PLAN_ITEMS,'evidence_refs':PLAN_ITEMS}}}}
NESTED = {
    '_step_update_research_plan':{'plan':PLAN},
    '_step_record_baseline_diagnosis':{'diagnosis':BASELINE_DIAGNOSIS},
    '_step_configure_baseline_reference':{'reference':{
        'type':'object','required':['id','kind','reason','source_refs'],
        'properties':{'id':ID,'kind':{'type':'string','enum':['released_checkpoint','unavailable']},
            'reason':REASON,'source_refs':SOURCE_REFS,'checkpoint_ref':{'type':'string','minLength':1},
            'release_url':{'type':'string','minLength':1},'revision':{'type':'string','minLength':1},
            'episodes':{'type':'integer','minimum':1},'expected_metric':{'type':'number'},
            'tolerance':{'type':'number','minimum':0}},
        'if':{'properties':{'kind':{'const':'released_checkpoint'}}},
        'then':{'required':['checkpoint_ref','release_url','revision','episodes','expected_metric','tolerance']}}},
    '_step_register_data_version':{'version':{
        'type':'object','required':['id','dataset_ref','reason','source_refs'],
        'properties':{'id':ID,'dataset_ref':{'type':'string','minLength':1},
            'reason':REASON,'source_refs':SOURCE_REFS,'asset':{'type':'object'}},
        'additionalProperties':False}},
    '_step_configure_confirmation_protocol':{'source_refs':SOURCE_REFS,
        'confirmation':{'type':'object','minProperties':1,'maxProperties':8},
        'protocol_keys':{'type':['array','null'],'maxItems':8,'uniqueItems':True,
                         'items':{'type':'string','minLength':1}},'reason':REASON},
}


def schema(handler) -> dict:
    properties, required = {}, []
    for name, parameter in inspect.signature(handler).parameters.items():
        if name == 'self' or parameter.kind in {parameter.VAR_POSITIONAL,parameter.VAR_KEYWORD}:
            continue
        annotation = str(parameter.annotation)
        kind = next((kind for token,kind in [('dict','object'),('list','array'),('str','string'),
            ('bool','boolean'),('float','number'),('int','integer')] if token in annotation), None)
        row = {'type':kind} if kind else {}
        if 'None' in annotation and kind:
            row['type'] = [kind,'null']
        if parameter.default is parameter.empty:
            required.append(name)
        else:
            row['default'] = parameter.default
        if name in {'reason','retry_reason'}:
            row['maxLength'] = 1000
        properties[name] = row
    for name, row in NESTED.get(handler.__name__,{}).items():
        if name in properties:
            properties[name] = {**properties[name],**row}
    for name in ('evidence_id','job_id'):
        if name in properties:
            properties[name]['pattern'] = r'^(?:[0-9a-f]{32})?$' if properties[name].get('default') == '' else r'^[0-9a-f]{32}$'
            properties[name]['description'] = 'Opaque 32-hex evidence/job ID, not action_id or receipt sha256.'
    return {'type':'object','properties':properties,'required':required,'additionalProperties':False}


def validate(contract: dict, arguments: dict) -> None:
    _validate(contract,arguments,'arguments')


def _validate(row, value, name):
    kinds = {'object':dict,'array':list,'string':str,'boolean':bool,'number':(int,float),'integer':int}
    raw = row.get('type')
    allowed = raw if isinstance(raw,list) else [raw]
    if value is None and 'null' in allowed:
        return
    kind = next((x for x in allowed if x in kinds),None)
    if kind and (not isinstance(value,kinds[kind]) or
            (kind in {'integer','number'} and (isinstance(value,bool) or not math.isfinite(value)))):
        raise ValueError(f'{name} must have type {raw}')
    if 'const' in row and value != row['const']:
        raise ValueError(f'{name} must equal {row["const"]!r}')
    if 'enum' in row and value not in row['enum']:
        raise ValueError(f'{name} must be one of {row["enum"]}')
    if 'maxEncodedBytes' in row:
        size = len(json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8'))
        if size > row['maxEncodedBytes']:
            raise ValueError(f'{name} has {size} UTF-8 bytes; maximum is {row["maxEncodedBytes"]}; cite evidence files instead')
    if isinstance(value,dict):
        properties = row.get('properties',{})
        missing = set(row.get('required',[]))-set(value)
        extra = set(value)-set(properties) if row.get('additionalProperties') is False else set()
        if missing or extra:
            raise ValueError(f'{name}: missing fields={sorted(missing)}; unexpected fields={sorted(extra)}')
        for key,child in value.items():
            if key in properties:
                _validate(properties[key],child,f'{name}.{key}')
        if len(value)<row.get('minProperties',0) or len(value)>row.get('maxProperties',float('inf')):
            raise ValueError(f'{name} violates object size bounds')
        if 'if' in row:
            try:
                _validate(row['if'],value,name)
            except ValueError:
                pass
            else:
                _validate(row.get('then',{}),value,name)
    if isinstance(value,list):
        if len(value)<row.get('minItems',0) or len(value)>row.get('maxItems',float('inf')):
            raise ValueError(f'{name} violates array length bounds')
        if row.get('uniqueItems') and any(item in value[:i] for i,item in enumerate(value)):
            raise ValueError(f'{name} must have unique items')
        for i,child in enumerate(value):
            _validate(row.get('items',{}),child,f'{name}[{i}]')
    if isinstance(value,str):
        if len(value)<row.get('minLength',0):
            raise ValueError(f'{name} must be nonempty')
        if 'maxLength' in row and len(value)>row['maxLength']:
            raise ValueError(f'{name} has {len(value)} characters; maximum is {row["maxLength"]}')
        if 'pattern' in row and not re.fullmatch(row['pattern'],value):
            raise ValueError(f'{name} must match {row["pattern"]}; use its typed handle, not a digest')
    if isinstance(value,(int,float)) and not isinstance(value,bool):
        if not math.isfinite(value) or value<row.get('minimum',-float('inf')):
            raise ValueError(f'{name} must be finite and at least {row.get("minimum")}')

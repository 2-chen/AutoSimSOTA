"""One executable signature supplies Agent argument types and local validation."""
import inspect
import json
import math
import re


# Expose nested shape constraints before an operation spends a Supervisor turn.
# Scientific/source validation remains in the native domain validators, not here.
SOURCE_REFS = {'type':'array','minItems':1,'maxItems':12,'items':{
    'type':'object','required':['ref','quote'],'properties':{
        'ref':{'type':'string','minLength':1,'maxLength':1024,
               'description':'Checkout-relative file[:line[-line]], native-source:ID or checkout-source:ID. '
                   'Never use [LOCAL_RESOURCE]/[LOCAL_PATH]. For redacted JSON/config sources capture '
                   'inspect_native_environment.checkout_source_request and copy its exact returned ref/quote.'},
        'quote':{'type':'string','minLength':1,'maxLength':4000,
                 'description':'Exact source quote, confined to cited lines; for checkout-source:ID copy the whole returned quote.'}},'additionalProperties':False}}
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
        'decision_memory':{'type':'object','additionalProperties':False,'maxEncodedBytes':6000,
            'required':['bottleneck','observations','lessons','failed_directions','next_experiment'],
            'properties':{'bottleneck':_text_shape(1000),'next_experiment':_text_shape(1500),
                **{k:PLAN_ITEMS for k in ('observations','lessons','failed_directions')}}},
        'budget_strategy':{'type':'object','additionalProperties':False,'maxEncodedBytes':6000,
            'required':['why','allocations','evidence_refs','revisit_condition','next_action'],
            'properties':{'why':_text_shape(1500),'revisit_condition':_text_shape(1000),
                'next_action':_text_shape(1000),'allocations':PLAN_ITEMS,'evidence_refs':PLAN_ITEMS}}}}
NESTED = {
    '_step_recover_unscored_baseline':{'artifact_revision':{
        'type':['object','null'],'properties':{'pattern':{'type':'string','minLength':1,'maxLength':500},
            'reason':REASON,'source_refs':SOURCE_REFS},
        'required':['pattern','reason','source_refs'],'additionalProperties':False}},
    '_step_recover_reference_score': {'artifact_revision': {
        'type':['object','null'],'required':['pattern','reason','source_refs'],
        'properties':{'pattern':{'type':'string','minLength':1,'maxLength':512},
                      'reason':{'type':'string','minLength':1,'maxLength':1000},
                      'source_refs':SOURCE_REFS},'additionalProperties':False,
        'description':'Optional source-reviewed attempt-owned result locator for the SAME completed '
            'reference only; independent review and policy/completed-episode/metric checks, no GPU replay '
            'or protocol/stage mutation. Never choose by the metric value.'}},
    '_step_correct_policy_handoff': {
        'artifact_revision': {
            'type':['object','null'],'required':['pattern','reason','source_refs'],
            'properties':{'pattern':{'type':'string','minLength':1,'maxLength':512},
                'reason':REASON,'source_refs':SOURCE_REFS},'additionalProperties':False,
            'description':'Only for an offered completed_evaluation_missing_locator. Main selects '
                'a source-backed result locator; independent review plus native load/rollout/metric '
                'reverification on CPU, no native replay or metric change. Omit checkpoint_ref and job_id.'},
        'checkpoint_ref': {'type':'string',
            'description':'Source-backed file or complete bundle inside the ORIGINAL completed training '
                'producer. Select this to run the frozen evaluation; omit job_id.'},
        'job_id': {'type':'string',
            'description':'Only an offered COMPLETED EVALUATE native-job ID from completed_evaluation_links '
                'to attach an existing evaluation without replay. Not a training attempt/evidence ID. '
                'Omit checkpoint_ref in this mode.'}},
    '_step_derive_a_command': {'artifact_revision': {
        'type':['object','null'], 'required':['pattern','reason','source_refs'],
        'properties':{'pattern':{'type':'string','minLength':1,'maxLength':512},
                      'reason':{'type':'string','minLength':1,'maxLength':1000},
                      'source_refs':SOURCE_REFS}, 'additionalProperties':False}},
    '_step_record_data_campaign_arm':{'boundary':{
        'type':['object','null'],'additionalProperties':False,
        'required':['status','reason','evidence_refs','source_refs'],
        'properties':{'status':{'type':'string','enum':['unsupported','budget_limited']},
            'reason':{'type':'string','minLength':1,'maxLength':2000},
            'source_refs':SOURCE_REFS,
            'evidence_refs':{'type':'array','minItems':1,'maxItems':8,'uniqueItems':True,
                'items':{'type':'string','pattern':r'^[0-9a-f]{32}$'},
                'description':'Sealed native/operation evidence IDs, not job/measurement/decision IDs.'}}}},
    '_step_build_the_environment': {
        'reuse_environment_ref':{'type':['string','null'],
            'pattern':r'^(env|environments/[A-Za-z0-9_-]{1,80})$',
            'description':'Select a metadata-verified run-owned overlay from environment.prepared_overlays. '
                'Requires its matching base_environment_id and overlay mode; omitted source_binding_ids '
                'preserve existing external bindings and installed delta. Native consumers are reverified.'},
        'base_environment_id':{'type':['string','null'],'minLength':1},
        'base_environment_mode':{'type':'string','enum':['overlay','clone','reconstruct']},
        'environment_selection_reason':{'type':'string','maxLength':2000,
            'description':'Short source/evidence-grounded rationale, <=2000 characters. '
                'Omit base selection to continue the existing cursor; an omitted binding list '
                'reasserts the accepted plan, not a new environment. Cite evidence rather than paste logs.'}},
    '_step_review_confirmation_failure': {'diagnostic_evidence_id': {
        'type': 'string', 'pattern': r'^[0-9a-f]{32}$',
        'description': 'CPU repair-probe evidence ID (termination_reason=cpu_native_diagnostic), '
                       'or the sealed confirm_best_engineering envelope which binds that CPU probe. '
                       'Not the original held-out failure ID.'}, 'reason': REASON},
    '_step_inspect_native_job':{'log_query':{'type':['string','null'],'maxLength':128,
        'description':'Optional literal text from native producer source/logs to inspect in a bounded live snapshot. '
                      'Empty string offers the bounded tail. Not regex or a terminal-work audit; excludes evaluation/held-out.'}},
    '_step_submit_native_job':{'evaluation_settings':{
        'type':['object','null'],'minProperties':1,
        'properties':{'episodes':{'type':'integer','minimum':1},
                      'seed':{'type':'integer','minimum':0}},'additionalProperties':False},
        'collection_settings':{'type':['object','null'],'required':['episodes'],
            'properties':{'episodes':{'type':'integer','minimum':1},
                          'seed':{'type':'integer','minimum':0},
                          'controls':{'type':'object','minProperties':1,'maxProperties':12,
                              'description':'Native training-side fields only. The verified builder must read '
                                  'collection_controls; actual observed effect must be checked after the probe.'},
                          'control_source_refs':SOURCE_REFS},'additionalProperties':False}},
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
            properties[name].setdefault('description',
                'Opaque 32-hex evidence/job ID, not action_id or receipt sha256.')
    contract = {'type':'object','properties':properties,'required':required,'additionalProperties':False}
    if handler.__name__ == '_step_recover_unscored_baseline':
        contract.update({'if':{'required':['artifact_revision'],
            'properties':{'artifact_revision':{'type':'object'}}},
            'then':{'properties':{'checkpoint_ref':{'type':'string','maxLength':0}}}})
    if handler.__name__ == '_step_correct_policy_handoff':
        # The same argument contract is used for Main admission AND optional Fix
        # recommendations. Do not label a mutually exclusive recommendation valid
        # merely because both individual fields have the right Python type.
        contract.update({
            'description':'Two exclusive modes: label + checkpoint_ref runs evaluation of the completed '
                'training producer; label + offered completed EVALUATE job_id links existing evaluation. '
                'Never pass a training attempt ID as job_id or combine nonempty checkpoint_ref and job_id.',
            'if':{'required':['job_id'],'properties':{'job_id':{'type':'string','minLength':1}}},
            'then':{'properties':{'checkpoint_ref':{'const':''}}}})
    return contract


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

"""One executable signature supplies Agent argument types and local validation."""
import inspect
import math


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
    return {'type':'object','properties':properties,'required':required,'additionalProperties':False}


def validate(contract: dict, arguments: dict) -> None:
    if not isinstance(arguments,dict):
        raise ValueError('arguments must be an object')
    missing = set(contract['required'])-set(arguments)
    extra = set(arguments)-set(contract['properties'])
    if missing or extra:
        raise ValueError(f'missing fields={sorted(missing)}; unexpected fields={sorted(extra)}')
    kinds = {'object':dict,'array':list,'string':str,'boolean':bool,'number':(int,float),'integer':int}
    for name,value in arguments.items():
        row = contract['properties'][name]
        raw = row.get('type')
        allowed = raw if isinstance(raw,list) else [raw]
        if value is None and 'null' in allowed:
            continue
        kind = next((x for x in allowed if x in kinds),None)
        if kind and (not isinstance(value,kinds[kind])
                or (kind in {'integer','number'} and (isinstance(value,bool) or not math.isfinite(value)))):
            raise ValueError(f'{name} must have type {raw}')
        if 'maxLength' in row and isinstance(value,str) and len(value)>row['maxLength']:
            raise ValueError(f'{name} has {len(value)} characters; maximum is {row["maxLength"]}')

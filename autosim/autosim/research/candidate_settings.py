"""Agent-selected local training work without rewriting immutable baseline settings."""
from .training_work import consumes_steps


def proposed(research, base: dict, idea) -> dict:
    """Project settings without applying a patch or choosing another hypothesis."""
    settings=research._base_settings(base)
    if idea.granularity in {'param','algo'}:
        from .decision import validate_proposal
        proposal=validate_proposal(research._envelope_for(idea,evidence_id='candidate-work-preflight'),
            space=research.space,evidence_id='candidate-work-preflight')
        for key,value in (proposal.get('training') or {}).items():
            if isinstance(value,dict):
                settings.update({f'{key}.{k}':v for k,v in value.items()})
            else:
                settings[key]=value
    return settings


def apply(research, settings: dict, requested: dict) -> dict:
    if not isinstance(requested,dict) or not requested or len(requested)>64:
        raise ValueError('candidate training_settings needs a nonempty bounded object')
    axes={(f'{a.group}.{a.name}' if a.group else a.name):a for a in research.space.training}
    for key,value in requested.items():
        axis=axes.get(key)
        if key=='steps' and consumes_steps(research.sources.get('train','')):
            # Same explicit caller input already supported for fresh baselines. Do not
            # infer defaults, alias arbitrary fields, or convert native epochs to updates.
            axis=axis or next((a for a in research.space.training if a.name=='steps'),None)
            valid=type(value) is int and value>0 and (axis is None or axis.accepts(value))
        else:
            valid=axis is not None and axis.accepts(value)
        if not valid:
            raise ValueError(f'candidate training setting {key} lacks a valid declared/source-backed input')
    result={**settings,**requested}
    if research._comparison_protocol_violation(result,target='evaluate'):
        raise ValueError('candidate training_settings changes the frozen evaluation protocol')
    return result

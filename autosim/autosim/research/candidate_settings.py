"""Agent-selected local training work without rewriting immutable baseline settings."""
import re

from .training_work import consumes_input, consumes_steps

TRAINING_SEED = '_autosim_training_seed'


def scoped_training_inputs(research, requested: dict) -> dict:
    """Keep the public training seed separate from the frozen evaluation seed.

    This private executor slot is not a native flag or a new optimization axis.
    Only the train builder's existing seed input receives it; collection and
    evaluation retain their own settings. Missing is not an invented seed.
    """
    result = dict(requested)
    if 'seed' in result:
        value = result.pop('seed')
        if (type(value) is not int or value < 0 or
                not consumes_input(research.sources.get('train',''), 'seed')):
            raise ValueError('training seed needs a nonnegative integer and a verified trainer seed input')
        result[TRAINING_SEED] = value
    return result


def merge_training(research, base: dict, idea, proposal: dict) -> dict:
    """Apply only requested axes; validator-filled defaults are not interventions.

    Library ideas are sparse deltas. Their complete validation envelope includes
    declaration defaults, which may differ from the measured baseline (notably
    its evaluation seed). Never turn these validation placeholders into changes.
    Full proposals remain full changes because their idea explicitly names them.
    """
    change = dict(idea.change or {}) if idea.granularity in {'param', 'algo'} else {}
    sections = set(research.space.sections())
    if any(key in sections for key in change):
        requested = dict(change.get('training') or {})
        requested.update({key: value for key, value in change.items()
                          if key not in sections and
                          any(axis.name == key for axis in research.space.training)})
    else:
        requested = change

    def flatten(values):
        result = {}
        for key, value in values.items():
            if isinstance(value, dict):
                result.update({f'{key}.{member}': item for member, item in value.items()})
            else:
                result[key] = value
        return result

    explicit = flatten(requested)
    validated = flatten(proposal.get('training') or {})
    if set(explicit) - set(validated):
        raise ValueError('validated candidate is missing an explicitly requested training axis')
    return {**research._base_settings(base),
            **scoped_training_inputs(research, {key: validated[key] for key in explicit})}


def canonical_dataset_input(research, key, value):
    """Admit a registered selection for the existing executor slot, not an alias.

    Input admission is not scientific approval: resolve() still verifies the registry,
    current source review and bytes. AST access does not certify native consumption.
    """
    sources = getattr(research, 'sources', {})
    return (key == 'dataset' and isinstance(value, str) and isinstance(sources, dict)
        and re.fullmatch(r'data-version:[A-Za-z0-9_-]{1,64}', value) is not None
        and consumes_input(sources.get('train', ''), 'dataset'))


def proposed(research, base: dict, idea) -> dict:
    """Project settings without applying a patch or choosing another hypothesis."""
    settings=research._base_settings(base)
    if idea.granularity in {'param','algo'}:
        from .decision import validate_proposal
        proposal=validate_proposal(research._envelope_for(idea,evidence_id='candidate-work-preflight'),
            space=research.space,evidence_id='candidate-work-preflight')
        settings = merge_training(research, settings, idea, proposal)
    return settings


def apply(research, settings: dict, requested: dict) -> dict:
    if not isinstance(requested,dict) or not requested or len(requested)>64:
        raise ValueError('candidate training_settings needs a nonempty bounded object')
    axes={(f'{a.group}.{a.name}' if a.group else a.name):a for a in research.space.training}
    for key,value in requested.items():
        axis=axes.get(key)
        if canonical_dataset_input(research, key, value):
            valid = axis is None or axis.accepts(value)
        elif key == 'seed':
            valid = (type(value) is int and value >= 0
                and consumes_input(research.sources.get('train',''), 'seed')
                and (axis is None or axis.accepts(value)))
        elif key=='steps' and consumes_steps(research.sources.get('train','')):
            # Same explicit caller input already supported for fresh baselines. Do not
            # infer defaults, alias arbitrary fields, or convert native epochs to updates.
            axis=axis or next((a for a in research.space.training if a.name=='steps'),None)
            valid=type(value) is int and value>0 and (axis is None or axis.accepts(value))
        else:
            valid=axis is not None and axis.accepts(value)
        if not valid:
            raise ValueError(f'candidate training setting {key} lacks a valid declared/source-backed input')
    result={**settings,**scoped_training_inputs(research, requested)}
    violation = research._comparison_protocol_violation(result,target='evaluate')
    if violation:
        raise ValueError(f'candidate training_settings changes the frozen evaluation protocol: {violation}')
    return result

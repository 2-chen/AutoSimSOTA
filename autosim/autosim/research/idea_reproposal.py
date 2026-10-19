"""Explicit new proposal identity with exact historical payload, never execution approval."""
from copy import deepcopy
import json
import re

from .common import object_digest

FIELDS=('label','granularity','mechanism','change','risk','crosses','why','evidence','touches')


def proposal_snapshot(idea):
    try:
        value={key:deepcopy(getattr(idea,key)) for key in FIELDS}
        if (not isinstance(value['change'],dict) or not isinstance(value['evidence'],list)
                or not isinstance(value['touches'],list)
                or len(json.dumps(value,ensure_ascii=False,allow_nan=False).encode())>256*1024):
            raise ValueError('malformed or oversized historical proposal')
        return value
    except (TypeError,AttributeError) as exc:
        raise ValueError('malformed historical proposal') from exc


def expand_reproposal(candidate, library):
    required={'label','reuse_idea_label','expected_idea_sha256','reason'}
    if not isinstance(candidate,dict) or set(candidate)!=required:
        raise ValueError('reproposal requires exactly label, reuse_idea_label, expected_idea_sha256, reason')
    if any(not isinstance(candidate[k],str) or not candidate[k].strip() for k in required):
        raise ValueError('reproposal fields must be nonempty strings')
    if len(candidate['label'])>160 or len(candidate['reuse_idea_label'])>160 or len(candidate['reason'])>2000:
        raise ValueError('reproposal identity/reason exceeds its bound')
    origin=library.get(candidate['reuse_idea_label'])
    if origin is None or type(origin.times_tried) is not int or origin.times_tried<1:
        raise ValueError('reproposal origin must be a previously executed idea in this same library')
    if candidate['label'].strip()==origin.label or library.get(candidate['label'].strip()) is not None:
        raise ValueError('reproposal needs a distinct unused label; old trials and outcomes are immutable')
    snapshot=proposal_snapshot(origin)
    expected=candidate['expected_idea_sha256']
    if not re.fullmatch('[a-f0-9]{64}',expected) or expected!=object_digest(snapshot):
        raise ValueError('reproposal origin identity changed or does not match the offered snapshot')
    snapshot['label']=candidate['label'].strip()
    snapshot['why']=candidate['reason'].strip()
    snapshot['evidence']=[*snapshot['evidence'],
        'exact_prior_proposal:'+origin.label+':'+expected]
    return snapshot,origin.label


def reproposal_offers(library):
    rows=[idea for idea in library.ideas if type(idea.times_tried) is int and idea.times_tried>0]
    offers=[]
    for idea in rows[-8:]:
        try:
            identity=object_digest(proposal_snapshot(idea))
        except (ValueError,TypeError,AttributeError):
            offers.append({'label':str(idea.label)[:160],'status':'unavailable',
                'authority':'Malformed historical proposal; no reproduction permission or hidden fallback.'})
            continue
        offers.append({'label':idea.label,'expected_idea_sha256':identity,
        'status':idea.status,'times_tried':idea.times_tried,
        'last_outcome':str(idea.outcome or '')[:240],
        'granularity':idea.granularity,
        'authority':'historical proposal identity only; not installed source, a selectable idea, '
            'new trial settings, prior score transfer or execution approval. Main may explicitly '
            'propose a new label with this exact payload; fresh shape/red-line/source/execution checks still apply.'})
    return offers

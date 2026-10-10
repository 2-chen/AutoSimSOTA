"""Exact model-visible citations mapped to retained, independently checked originals."""
from .common import sanitize_model_payload


def project(packet: dict, citations: list, *, local_roots=()) -> tuple[dict, list]:
    shown = sanitize_model_payload({**packet, 'review_sources':[
        {**row, 'source_id':f'review-source-{i+1}'} for i,row in enumerate(citations)]},
        local_roots=local_roots)
    return shown, shown['review_sources']


def resolve(original: list, shown: list, citations: list) -> list:
    if not isinstance(citations, list) or not citations:
        raise ValueError('Supervisor must cite exact supplied reference evidence')
    resolved = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError('Supervisor citation must be an object')
        matches = [i for i,row in enumerate(shown)
            if citation.get('quote') == row['quote'] and
            (citation.get('source_id') == row.get('source_id') if citation.get('source_id')
             else citation.get('ref') == row['ref']) and
            ('ref' not in citation or citation['ref'] == row['ref'])]
        if len(matches) != 1:
            raise ValueError('Supervisor must cite one exact supplied source_id and displayed quote')
        resolved.append(dict(original[matches[0]]))
    return resolved


INSTRUCTION = (' Cite exact {source_id,quote} pairs from review_sources; copy displayed '
               'redaction placeholders verbatim. Never reconstruct private paths. ')

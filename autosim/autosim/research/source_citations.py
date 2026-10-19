"""Exact model-visible citations mapped to retained, independently checked originals."""
from .common import sanitize_model_payload


class ReviewCitationError(ValueError):
    pass


def project(packet: dict, citations: list, *, local_roots=()) -> tuple[dict, list]:
    shown = sanitize_model_payload({**packet, 'review_sources':[
        {**row, 'source_id':f'review-source-{i+1}'} for i,row in enumerate(citations)]},
        local_roots=local_roots)
    return shown, shown['review_sources']


def resolve(original: list, shown: list, citations: list) -> list:
    if not isinstance(citations, list) or not citations:
        raise ReviewCitationError('Supervisor must select supplied source IDs as citation evidence')
    resolved = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise ReviewCitationError('Supervisor citation must be an object')
        # Selecting an executor-supplied ID is sufficient; avoid forcing the model
        # to retranscribe source bytes. A supplied quote still MUST match exactly.
        if 'source_id' in citation and set(citation)<= {'source_id','ref'}:
            matches = [i for i, row in enumerate(shown)
                if citation['source_id'] == row.get('source_id') and
                ('ref' not in citation or citation['ref']==row.get('ref'))]
            if len(matches) != 1:
                raise ReviewCitationError('Supervisor must select one exact supplied source_id')
            resolved.append(dict(original[matches[0]]))
            continue
        matches = [i for i,row in enumerate(shown)
            if citation.get('quote') == row['quote'] and
            (citation.get('source_id') == row.get('source_id') if citation.get('source_id')
             else citation.get('ref') == row['ref']) and
            ('ref' not in citation or citation['ref'] == row['ref'])]
        if len(matches) != 1:
            raise ReviewCitationError('Supervisor must select a supplied source_id alone, or copy its entire displayed quote exactly')
        resolved.append(dict(original[matches[0]]))
    return resolved


INSTRUCTION = (' Cite {source_id} objects selected from review_sources; the executor maps IDs to retained '
               'exact source evidence, so you do not need to repeat quotes. An optional redundant ref '
               'must exactly match that supplied source ID. If you include a quote, '
               'copy the ENTIRE displayed quote verbatim, including redaction placeholders. '
               'No subquotes, guessed IDs or reconstructed private paths. ')

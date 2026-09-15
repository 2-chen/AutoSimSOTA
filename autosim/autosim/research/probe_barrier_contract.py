"""Pure identity gate; process liveness is measured by the trusted coordinator."""
from __future__ import annotations


def ready_members_match(expected: list[dict], ready: list[dict], generation: str) -> bool:
    """Release only the complete, unique, live membership of this generation."""
    if not isinstance(generation, str) or not generation:
        return False
    if not isinstance(expected, list) or not isinstance(ready, list) or not expected or len(expected) != len(ready):
        return False
    fields = ("worker", "uuid", "generation", "claim", "attempt_id")
    identities = {}
    uuids = set()
    for item in expected:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k] for k in fields):
            return False
        if item["generation"] != generation or item["worker"] in identities or item["uuid"] in uuids:
            return False
        identities[item["worker"]] = tuple(item[k] for k in fields)
        uuids.add(item["uuid"])
    seen = set()
    for item in ready:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k] for k in fields):
            return False
        worker = item["worker"]
        if worker in seen or worker not in identities or tuple(item[k] for k in fields) != identities[worker]:
            return False
        if item.get("state") != "ready" or item.get("alive") is not True:
            return False
        seen.add(worker)
    return seen == set(identities)

"""Reject all malformed proposal fields before paying for scientific review."""
FIELDS = {"failure_evidence_id", "execution_aliases_digest", "commands", "probes",
          "repair_mode", "original_operation_disposition", "install_replacement_evidence",
          "probe_replacement_evidence", "retire_operations", "capability_evidence_ids",
          "resource_requests", "reasoning", "unbuildable", "resource_assessment"}


def source_reference_name(value):
    """Common citation spellings identify the same checkout file, not new authority."""
    if isinstance(value, str):
        return value if 1 <= len(value) <= 512 and value.strip() else None
    if isinstance(value, dict):
        names = [value[key] for key in ('path', 'file', 'source', 'ref') if key in value]
        if (names and all(isinstance(name, str) and 1 <= len(name) <= 512
                          and name.strip() for name in names) and len(set(names)) == 1):
            return names[0]
    return None


def proposal_errors(value):
    errors = []
    for key in sorted(set(value)-FIELDS):
        hint = "; use probes, not probe_commands" if key == "probe_commands" else ""
        errors.append(f"{key}: unknown repair field{hint}")
    if value.get("repair_mode", "prerequisites") not in ("prerequisites", "replace_operation"):
        errors.append("repair_mode: expected prerequisites or replace_operation")
    for field in ("commands", "probes", "retire_operations"):
        entries = value.get(field, [])
        if not isinstance(entries, list) or len(entries) > 64 or any(
                not isinstance(item, str) or not item.strip() or len(item) > 32768 for item in entries):
            errors.append(f"{field}: expected <=64 nonempty strings of <=32768 characters")
    for field in ("install_replacement_evidence", "probe_replacement_evidence"):
        if field not in value:
            continue
        evidence = value[field]
        if not isinstance(evidence, dict):
            errors.append(f"{field}: expected object, not list/prose")
            continue
        if not isinstance(evidence.get("same_capability"), str) or not 1 <= len(evidence['same_capability'].strip()) <= 2000:
            errors.append(f"{field}.same_capability: expected nonempty string <=2000 characters")
        refs = evidence.get("source_refs")
        if not isinstance(refs, list) or not 1 <= len(refs) <= 6:
            errors.append(f"{field}.source_refs: expected 1..6 checkout source citations")
        else:
            for index, reference in enumerate(refs):
                if source_reference_name(reference) is None:
                    errors.append(f"{field}.source_refs[{index}]: expected checkout-relative string "
                        "or object with path/file/source/ref naming one file; conflicting names are invalid")
        if field == "install_replacement_evidence" and evidence.get("failure_evidence_id") != value.get("failure_evidence_id"):
            errors.append(f"{field}.failure_evidence_id: must match proposal failure_evidence_id")
    return errors

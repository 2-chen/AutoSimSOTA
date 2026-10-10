"""Source-backed review of a bounded training launch, never terminal work or score."""
import json
from pathlib import Path

from .agent_client import role_scope
from .common import atomic_json, digest, read_json
from .evidence_store import read_attempt_evidence
from .training_progress_audit import sources


class VerificationWindowBoundary(RuntimeError):
    """A local time bound must leave invocation repair, not masquerade as a crash."""

    def __init__(self, outcome):
        super().__init__(outcome["error"])
        self.outcome = outcome


def review(client, *, repo, root, argv, evidence_ref, timeout):
    result = {"status": "unverified", "authority": "startup_only_not_completion_or_score"}
    try:
        if not getattr(client, "supports_native_progress_audit", False):
            return {**result, "reason": "independent startup reviewer unavailable"}
        identity = Path(evidence_ref).stem
        if evidence_ref != f"evidence/{identity}.json":
            raise ValueError("invalid startup evidence reference")
        sealed = read_json(root / evidence_ref)
        receipt_path = root / str(sealed.get("receipt_ref") or "")
        if (receipt_path.is_symlink() or not receipt_path.resolve().is_relative_to(root.resolve())):
            raise ValueError("unsafe native receipt")
        receipt = read_json(receipt_path)
        if (receipt.get("id") != identity or receipt.get("stage") != "train"
                or receipt.get("argv") != list(argv)
                or receipt.get("capture_error") or receipt.get("orphaned_children")
                or receipt.get("termination_reason") == "storage_pressure"):
            raise ValueError("native timeout is not a clean contained training window")
        logs = [read_attempt_evidence(root, identity, offset=offset, limit=12000)
                for offset in dict.fromkeys((0, max(0, int(sealed.get("log_bytes") or 0)-12000)))]
        if any(log.get("status") != "timed_out" for log in logs):
            raise ValueError("startup boundary requires sealed timeout evidence")
        code = sources(repo, root, argv, cwd=receipt.get("cwd") or repo)
        if not code:
            raise ValueError("no native source trace for training semantics")
        # Bounded context without dropping the tail where native update/save code
        # often lives. Incomplete traces force abstention, never a heuristic pass.
        excerpts = [{"id": row["id"], "sha256": row["sha256"],
                     "excerpt": (row["excerpt"] if len(row["excerpt"]) <= 10000
                                 else row["excerpt"][:5000] + "\n…\n" + row["excerpt"][-5000:])}
                    for row in code]
        packet = {"scope": "startup_only", "native_evidence_ref": evidence_ref,
                  "logs": [{"offset": log.get("offset"), "text": log["text"]} for log in logs],
                  "sources": excerpts}
        with role_scope(client, "supervisor"):
            content, _ = client.chat_with_metadata(
                "Independently review a bounded training STARTUP, not completion or score. "
                "The process was contained and terminated at its local time bound. Decide "
                "whether exact native output AND producer source prove data was consumed "
                "and at least one optimizer update completed. A requested epoch, loader "
                "count, initialization, evaluation-only success or elapsed time is not proof. "
                "Abstain if the excerpts do not establish the chain. Do not claim a terminal "
                "or usable policy, modify code or execute commands. Return JSON only: "
                '{"verdict":"verified|unverified","why":"short reason",'
                '"log_quote":"exact nonempty output substring",'
                '"source_evidence":[{"source_id":"offered id","quote":"exact substring"}]}.',
                json.dumps(packet, ensure_ascii=False), timeout=timeout, max_tokens=1800,
                read_only=True, auto_skills=False, include_research_context=False)
        from .execution_derive import _object
        answer = _object(content)
        if answer.get("verdict") != "verified":
            return {**result, "reason": str(answer.get("why") or "review abstained")[:800]}
        quote = answer.get("log_quote")
        if not isinstance(quote, str) or not quote.strip() or not any(quote in log["text"] for log in logs):
            raise ValueError("startup review lacks an exact native output quote")
        citations = answer.get("source_evidence")
        if not isinstance(citations, list) or not citations or len(citations) > 8:
            raise ValueError("startup review lacks bounded source citations")
        for citation in citations:
            offered = next((row for row in excerpts if row["id"] == citation.get("source_id")), None)
            text = citation.get("quote")
            if (offered is None or not isinstance(text, str) or not text.strip()
                    or text not in offered["excerpt"]):
                raise ValueError("startup review source quote is not an offered substring")
        # Revalidate the sealed log and source bytes after model review.
        read_attempt_evidence(root, identity, limit=1)
        if any(digest(row["path"]) != row["sha256"] for row in code):
            raise ValueError("startup source changed during review")
        result.update(status="verified", native_evidence_ref=evidence_ref,
                      why=str(answer.get("why") or "")[:800],
                      log_quote=quote, source_evidence=citations,
                      completion_verified=False, artifact_verified=False,
                      next="Resource Agent budgets a formal job; Scheduler submits it. "
                           "Preserve partial artifacts, inspect native resume support; "
                           "formal loading/rollout/scoring remain mandatory.")
        path = root / "startup_reviews" / f"{identity}.json"
        atomic_json(path, {"result": result, "answer": answer})
        result["review_ref"] = path.relative_to(root).as_posix()
        return result
    except Exception as exc:
        # Model/runtime failure is an abstention, never a reason to repeat training
        # inside invocation repair or discard the sealed native time-bound evidence.
        return {**result, "reason": f"{type(exc).__name__}: {exc}"[:800]}

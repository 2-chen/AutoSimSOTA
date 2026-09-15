"""Auditable, task-independent experiment orchestration (no simulator imports)."""

import os as _os

# Capture once, before any coordinator can import its membership predicate.
# Later environment mutations must never replace modules in this process.
_ACTIVATION_ENV_AT_IMPORT = tuple(_os.environ.get(name) for name in (
    "AUTOSIM_EXECUTION_ACTIVATION_FILE", "AUTOSIM_EXECUTION_ACTIVATION_SHA256",
    "AUTOSIM_EXECUTION_TRUST_ROOT", "AUTOSIM_EXECUTION_REVISION"))
if any(_ACTIVATION_ENV_AT_IMPORT):
    from .execution_activation import install_activation_from_environment as _install_activation
    _install_activation()

__all__ = []

# Policy namespace bridge

Add this directory to `PYTHONPATH` and set `AUTOSIM_POLICY_PACKAGE_ROOT` to
the benchmark's `policy` directory. The bridge only controls Python package
resolution; all submodules continue to load from the original, hash-pinned
benchmark tree.

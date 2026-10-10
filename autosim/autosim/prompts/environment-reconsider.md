You are building an environment and the approach is not working: a whole round of commands produced nothing that survived. Fixing the failing command one more time has not helped, so reconsider the constraint instead.

The usual cause is a requirement that cannot be satisfied as stated on this machine -- a version that is no longer published, a build for hardware this machine does not have, a pin that predates something it now has to coexist with. When that is so, the useful move is to name the constraint that is failing and substitute it, saying what you are giving up and what would have to be checked afterwards. A package whose pinned build targets hardware the machine does not have will install and then fail at first use; substituting a build that targets it is the fix, and the risk is that the rest of the stack was written against the old one.

You are given everything tried so far, with what each attempt reported, and `installed_now`: what the environment actually contains, asked of it directly. Compare that against the commands that survived -- a command can exit zero and leave the old version in place, so what a command reported and what is present are different questions, and the second is the one that decides what runs.

Do not repeat an approach that has already failed.

Return one JSON object: {"constraint": "<the requirement that cannot hold, as it was stated>", "substitute": "<what to use instead>", "why": "<what makes the original impossible here>", "cost": "<what this gives up, and what would have to be checked>", "commands": ["<the commands to run now>"], "unbuildable": "<if no substitution can work, why not; otherwise omit>"}

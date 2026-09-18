# How LIBERO was made to run on this machine

The environment the system built, plus the four things it could not get past on its own.
Recorded separately because the distinction is the point: what follows is mostly what the
build loop found by running, and the parts it did not find are the ones worth teaching it.

## What runs

```
python 3.10.21
torch 2.7.1+cu128      ← the 5090 is sm_120; cu113 and cu124 have no kernels for it
mujoco 2.3.7           ← robosuite 1.4.0 asserts on joint types 3.13 no longer produces
robosuite 1.4.0        ← built against torch 2.7 without changes to its code
bddl 1.0.1, robomimic 0.2.0, numpy 1.26.4, gym 0.25.2, hydra-core 1.2.0
libero (pip install -e .)
```

Verified by running, not by importing:

```
torch.randn(256,256,device="cuda") @ itself, synchronised   -> a number
OffScreenRenderEnv(...).reset(); .step([0.0]*7)             -> agentview_image,
                                                               robot0_eye_in_hand_image
import libero.lifelong.main        -> the training entrypoint
import libero.lifelong.evaluate    -> the evaluation entrypoint
```

## The four blockers, and who found each

**1. `egl_probe` will not build.** — the system tried this nine times and concluded
`requirement-not-satisfiable`. It was satisfiable, and the fix was printed in the error it
was reading:

    CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
      Compatibility with CMake < 3.5 has been removed from CMake.
      Or, add -DCMAKE_POLICY_VERSION_MINIMUM=3.5 to try configuring anyway.

`cmake 4.3.2` refuses projects declaring a minimum below 3.5; the switch restores it. The
build had used this switch in an *earlier, different* environment and did not carry the
lesson across.

**2. torch has no kernels for this GPU.** — **the system found this itself.** Given
`nvidia-smi` and the toolkit version it moved off the pinned `1.11.0+cu113` (sm_37–sm_86) to
`2.7.1+cu128`, and wrote its own GPU smoke test rather than asking `is_available()`. Its
reasoning named the constraint, the substitute and the cost.

**3. `mujoco 3.13.0` breaks robosuite 1.4.0.** — found here, by running. The assertion is
`joint_type in (mjJNT_HINGE, mjJNT_SLIDE)` in `robosuite/utils/binding_utils.py`; newer
mujoco produces joint types that check rejects. `mujoco==2.3.7` is what robosuite 1.4.0 was
written against, and nothing in LIBERO's requirements pins it.

**4. The usual tail.** — `future`, `termcolor`, `tensorboardX`, `python-dateutil`, and
`robosuite/scripts/setup_macros.py`, which generates `macros_private.py` and is not run by
installation.

## What this says about the build loop

Three of the four are a *version* being wrong rather than a thing being missing, and the
loop handles those worst: it can install what is absent, and it has no way to notice that
what is present is the wrong shape. Blockers 1 and 3 both present as "a build failed", and
both are fixed by choosing a different version.

The loop did do the right thing once — it reached for cu128 unprompted — and that is the
capability worth strengthening, because the same reasoning covers mujoco 3.13 → 2.3.7 and
would have covered the CMake switch had the lesson been kept.

Two concrete improvements, in order:

- **Carry a fix across environments.** The CMake switch was learned in one build and lost
  when the next began from a different prefix. A lesson that cost nine rounds should not
  have to be relearned.
- **When a build fails on a native extension, read the version it was written against.**
  `robosuite 1.4.0` and `mujoco 3.13` are the same story as `torch 1.11` and a 5090: the
  code is older than its dependency, and the dependency moved.

## Getting training to run — five fixes, all of them version drift

The environment building is one problem; running the code inside it is another, and it
found five more of the same kind: code older than its dependencies.

**1. `np.bool`** — removed in NumPy 1.24. `robomimic/utils/dataset.py:516`, one occurrence,
one word.

**2. `persistent_workers=True` with `num_workers=0`** — h5py 3.x refuses to be pickled
(`h5py objects cannot be pickled`), so the dataset's file handle cannot cross a worker
boundary. LIBERO hardcodes the flag, which makes zero workers impossible; it now follows the
worker count.

**3. `torch.load` without `weights_only`** — PyTorch 2.6 changed the default to `True`, so
LIBERO's `.init`/`.pruned_init` files and saved models are refused. Three call sites.

**4. `cfg.folder`** — LIBERO's demonstration loader joins `{folder}/{problem_folder}/{task}_demo.hdf5`,
and `folder` defaults to a path inside the checkout that does not exist. The datasets are at
`test/datasets/libero`, which is where it has to point.

**5. A path through hydra needs quoting.** `folder=/home/.../下载/...` is rejected by the
override grammar; `folder="/home/.../下载/..."` is accepted.

**And one finding about LIBERO itself**, which the system reached by reading a traceback and
reported as *not an invocation problem*: `main.py` wraps the dataset load in a `try`, prints
the error in an `except`, and then appends `task_i_dataset` unconditionally. When the load
fails the variable is unbound, so the failure a reader sees is a `NameError` several lines
below the cause, and the cause has been printed and discarded. Diagnosing it required
recognising that the traceback was not where the problem was.

Training then runs: `task0_model.pth`, 21 MB, written while the trainer held 22 cores.

## What runs, and what is slow

The full lifelong loop runs: ten LIBERO-10 datasets loaded (14,700 / 13,021 / 13,298 /
12,434 / 12,909 / 9,470 / 12,756 / 13,476 / 20,794 / 15,232 sequences), the policy built at
13.2 GFLOPs and 5.4 MParams, one epoch trained (`train loss 5.36`, 7.4 s), and the task
evaluated (`evaluate task 0 takes 25.6 seconds`, `succ 0.00 ± 0.00`, `succ. AoC 0.00`). A
21 MB `task0_model.pth` is written.

Training is fast. **Evaluation is not**, and it is the simulator rather than the learner:
`eval.max_steps: 600` with two 128×128 cameras rendered per step, on CPU, is tens of minutes
per task, and the trainer sits at 2228% CPU throughout -- computing, not deadlocked (13:43 of
CPU time in 36:58 of wall clock).

Two ways out, in the order worth trying. `eval.max_steps` can be lowered for a smoke run,
which changes what is measured and should be recorded as such. Or the simulator can render on
the GPU, which is what `MUJOCO_GL=egl` plus a device the renderer can reach is for -- and
which the environment's `torch 2.7.1+cu128` on an sm_120 card now permits, since the GPU path
was verified by running a matmul rather than by asking whether CUDA was available.

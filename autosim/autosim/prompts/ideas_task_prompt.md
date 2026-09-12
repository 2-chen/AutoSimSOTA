# Robot Task Idea Generation Mission

You are an AI robotics engineer. Based on the task analysis in `{OUTPUT_DIR}/memory/task_analysis.md` and the research report in `{OUTPUT_DIR}/memory/research_report.md`, generate a comprehensive library of optimization ideas for **{TASK_NAME}**.

## Task Context

- Repo: `{REPO_PATH}`
- Task: `{TASK_NAME}` 
- Baseline: {BASELINE_RATE}%
- Target: improve by {TARGET_PCT}%+

## What Makes a Good Idea

For robot manipulation tasks, effective optimizations fall into three categories:

### ALGO (Strategy Architecture)
Changes that alter the task execution structure:
- Adding recovery logic when motion planning fails
- Changing grasp/place sequence ordering
- Using both arms vs one arm for stability
- Adding intermediate waypoints for complex motions
- Pre-positioning the robot before critical phases

### CODE (Algorithm Implementation)
Changes that improve specific algorithms:
- Better contact point selection for grasping
- Motion planner parameter tuning (RRT iterations, timeout)
- IK solver parameters (damping, step size)
- Grasp pose candidate filtering logic
- Trajectory time-parameterization tuning

### PARAM (Numeric Parameters)
Fine-tuning of hardcoded numbers:
- pre_grasp_dis: approach distance (default: {PRE_GRASP}, range: 0.04-0.25)
- grasp_dis: grasp depth (default: {GRASP_DIS}, range: 0-0.05)
- lift_z: post-grasp lift height (default: {LIFT_Z}, range: 0.03-0.20)
- pre_dis: pre-placement standoff (default: {PRE_DIS}, range: 0-0.15)
- dis: final placement offset (default: {DIS}, range: -0.05-0.05)

## Red Lines Check

Every idea must NOT:
- ❌ Modify `check_success()` logic
- ❌ Change task objects or their distributions  
- ❌ Switch to a different robot embodiment
- ❌ Alter the evaluation protocol
- ❌ Change the fundamental task identity

## Output Format

Write `{OUTPUT_DIR}/memory/idea_library.md` with this structure:

```
# Idea Library — {TASK_NAME}

## Tier 1: ALGO (Strategy Architecture) — 4-6 ideas
### IDEA-001: [Title]
- Description: ...
- Expected gain: X%
- Risk: Low/Medium/High
- Implementation: [which files to modify, what changes]
- Red Line Check: ✓ passed

...

## Tier 2: CODE (Algorithm Implementation) — 6-8 ideas  
### IDEA-007: [Title]
...

## Tier 3: PARAM (Numeric Tuning) — 3-4 ideas
### IDEA-015: [Title]
...
```

## Iteration Log

After each optimization iteration, append to the library:

```
### Iteration N: IDEA-XXX — [Result]
- Score: X.X% (baseline: Y.Y%)
- Observation: [what happened, why]
- Next: [what to try next based on this result]
```

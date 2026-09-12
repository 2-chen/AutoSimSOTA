# Robot Task Onboarding Mission

You are an AI robotics engineer. Your mission: explore the local robotics repository at `{REPO_PATH}`, discover the **{TASK_NAME}** manipulation task, understand its strategy, and extract all relevant information needed for optimization.

## Environment

| Parameter | Value |
|-----------|-------|
| Repo path | `{REPO_PATH}` |
| Task name | `{TASK_NAME}` |
| Embodiment | `{EMBODIMENT}` |

## What You Need to Discover

### 1. Task Strategy

Read `{TASK_FILE}` thoroughly. The task defines a `play_once()` method containing the execution strategy.
- What actions does it perform? (grasp, lift, move, place, etc.)
- Which arm(s) does it use? How does it choose?
- What are the success conditions in `check_success()`?
- What are the failure modes?

### 2. Tunable Parameters

List ALL numeric parameters hardcoded in `play_once()`:
- pre_grasp_dis, grasp_dis (approach/grasp distances)
- lift distances, place offsets
- Contact point IDs
- Constraint types (free/align/auto)
- Target heights/positions

For each parameter, provide:
- Default value
- Reasonable optimization range
- Description of what it controls

### 3. Baseline Performance

Run the task with default parameters:
```
cd {REPO_PATH}
python -c "
... (use eval harness to measure success rate with 30+ seeds)
"
```

Record the baseline success rate.

### 4. Output

Write `{OUTPUT_DIR}/task_analysis.md` containing:
1. Task name and description
2. Strategy decomposition (phase-by-phase)
3. Parameter table with defaults and ranges
4. Baseline success rate
5. Embodiment info
6. Red lines (what must not change)

Also generate `{OUTPUT_DIR}/onboard_config.yaml`:
```yaml
repo_path: {REPO_PATH}
task_name: {TASK_NAME}
task_file: {TASK_FILE}
primary_metric: success_rate
metric_direction: higher
embodiment: {EMBODIMENT}
baseline_rate: 0.XX
params:
  pre_grasp_dis:
    default: 0.12
    range: [0.04, 0.20]
    description: "Approach distance before grasping"
  ... (all other params)
```

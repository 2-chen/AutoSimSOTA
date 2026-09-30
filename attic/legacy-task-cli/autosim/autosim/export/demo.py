"""
Export and demo generation — produce comparison video from optimization results.

Uses the save_aloha_video.py pipeline to create 4x4 grid comparison videos.
"""

import os, sys, types, warnings, json, subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_comparison_demo(repo_path: str, task_name: str,
                        default_params: dict, optimized_params: dict,
                        cases: int = 8, output: str = "output/comparison.mp4") -> str:
    """
    Generate a comparison demo video using save_aloha_video.py.
    Returns path to the generated video.
    """
    video_script = os.path.join(SCRIPT_DIR, "demo", "save_aloha_video.py")
    if not os.path.exists(video_script):
        raise FileNotFoundError(f"Video script not found: {video_script}")

    # Write params to temp file
    params_file = "/tmp/autosim_demo_params.json"
    with open(params_file, "w") as f:
        json.dump({
            "default": default_params,
            "optimized": optimized_params,
        }, f)

    # Run the video generation script
    env = os.environ.copy()
    env['AUTOSIM_DEMO_PARAMS'] = params_file

    cmd = [
        sys.executable, video_script,
        "--output", output,
        "--cases", str(cases),
        "--fps", "15",
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=SCRIPT_DIR,
                           capture_output=True, text=True, timeout=900)

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"Video generation failed: {result.stderr[-200:]}")

    # Print relevant output lines
    for line in result.stdout.split('\n'):
        if any(kw in line for kw in ['OK', 'FAIL', 'Done', 'Sync', 'MB', 'AutoSim']):
            print(f"  {line.strip()}")

    # Copy to output dir if needed
    abs_output = os.path.join(SCRIPT_DIR, output) if not os.path.isabs(output) else output
    if not os.path.exists(abs_output):
        # Check in RoboTwin dir
        alt = os.path.join(repo_path, output)
        if os.path.exists(alt):
            os.makedirs(os.path.dirname(abs_output), exist_ok=True)
            import shutil
            shutil.copy(alt, abs_output)

    return abs_output

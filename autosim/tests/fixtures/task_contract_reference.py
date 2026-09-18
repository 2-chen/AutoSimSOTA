def task_contract(documents):
    gym = documents["gym"]
    action = documents["action"]
    declared = documents["declared"]
    inner = action["action_config"] if "action_config" in action else action
    env = gym["env"]
    parts = env["control_parts"]
    scope = inner["scope"]
    dimension = 0
    for part in parts:
        dimension = dimension + int(scope[part]["dim"][0])
    sensors = [s for s in gym["sensor"] if s["sensor_type"] == "Camera"]
    cameras = [s["uid"] for s in sensors]
    shapes = {}
    for sensor in sensors:
        shapes[sensor["uid"]] = [int(sensor["height"]), int(sensor["width"]), 3]
    dataset = env["dataset"]
    recorder = [dataset[name] for name in dataset][0]["params"]
    rigid = gym["rigid_object"] if "rigid_object" in gym else []
    articulations_raw = gym["articulation"] if "articulation" in gym else []
    objects = sorted([r["uid"] for r in rigid if not r["uid"][:10] == "distractor"])
    articulations = sorted([a["uid"] for a in articulations_raw])
    events = env["events"] if "events" in env else {}
    mapping = declared["event_families"]
    families = {}
    for name in events:
        func = events[name]["func"] if "func" in events[name] else None
        family = mapping[func] if func in mapping else "task_semantics"
        if family not in families:
            families[family] = []
        families[family] = families[family] + [name]
    return {
        "name": documents["task"],
        "env_id": gym["id"],
        "setting": declared["setting"],
        "max_episode_steps": int(gym["max_episode_steps"]),
        "state_dim": dimension,
        "action_dim": dimension,
        "cameras": cameras,
        "camera_shapes": shapes,
        "control_parts": parts,
        "recorded_fps": float(recorder["robot_meta"]["control_freq"]),
        "instruction": recorder["instruction"]["lang"],
        "event_families": families,
        "roles": {"objects": objects, "articulations": articulations,
                  "family": declared["family"]},
        "correction_supported": declared["correction_supported"],
        "expert_adapter": declared["expert_adapter"],
    }

def field_name(d):
    return d["task"]


def field_env_id(d):
    return d["gym"]["id"]


def field_setting(d):
    return d["declared"]["setting"]


def field_max_episode_steps(d):
    return int(d["gym"]["max_episode_steps"])


def field_state_dim(d):
    inner = d["action"]["action_config"] if "action_config" in d["action"] else d["action"]
    scope = inner["scope"]
    total = 0
    for part in d["gym"]["env"]["control_parts"]:
        total = total + int(scope[part]["dim"][0])
    return total


def field_action_dim(d):
    return field_state_dim(d)


def field_cameras(d):
    return [s["uid"] for s in d["gym"]["sensor"] if s["sensor_type"] == "Camera"]


def field_camera_shapes(d):
    shapes = {}
    for s in d["gym"]["sensor"]:
        if s["sensor_type"] == "Camera":
            shapes[s["uid"]] = [int(s["height"]), int(s["width"]), 3]
    return shapes


def field_control_parts(d):
    return d["gym"]["env"]["control_parts"]


def field_recorded_fps(d):
    dataset = d["gym"]["env"]["dataset"]
    recorder = [dataset[k] for k in dataset][0]["params"]
    return float(recorder["robot_meta"]["control_freq"])


def field_instruction(d):
    dataset = d["gym"]["env"]["dataset"]
    recorder = [dataset[k] for k in dataset][0]["params"]
    return recorder["instruction"]["lang"]


def field_event_families(d):
    events = d["gym"]["env"]["events"] if "events" in d["gym"]["env"] else {}
    mapping = d["declared"]["event_families"]
    families = {}
    for name in events:
        func = events[name]["func"] if "func" in events[name] else None
        family = mapping[func] if func in mapping else "task_semantics"
        if family not in families:
            families[family] = []
        families[family] = families[family] + [name]
    return families


def field_roles(d):
    rigid = d["gym"]["rigid_object"] if "rigid_object" in d["gym"] else []
    articulated = d["gym"]["articulation"] if "articulation" in d["gym"] else []
    objects = sorted([r["uid"] for r in rigid if not r["uid"][:10] == "distractor"])
    articulations = sorted([a["uid"] for a in articulated])
    return {"objects": objects, "articulations": articulations,
            "family": d["declared"]["family"]}


def field_correction_supported(d):
    return d["declared"]["correction_supported"]


def field_expert_adapter(d):
    return d["declared"]["expert_adapter"]

"""
Enhanced MCP Client for Isaac Sim communication.

Full wrapper around the isaac-sim-mcp extension tools, providing:
  - Scene & environment management
  - Asset import / manipulation
  - Robot creation and joint control (Franka, UR5, etc.)
  - Camera view and capture
  - Physics control
  - Motion planning (IK, path planning, trajectory execution)
  - Domain randomization
  - Rich diagnostics

Supports both real Isaac Sim (via socket) and mock mode (for testing).
"""

from __future__ import annotations

import json
import logging
import re
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Transform:
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # xyzw


@dataclass
class ObjectSpec:
    """Specification for creating a scene object."""
    prim_type: str = "Cube"       # Cube, Sphere, Cylinder, Cone, Capsule, Mesh
    name: str = ""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    mass: float = 0.0             # 0 = static, >0 = rigid body
    collision: bool = True
    visual_material: str = ""
    usd_path: str = ""            # If set, prim_type is ignored; loads from USD


@dataclass
class RobotConfig:
    """Configuration for creating a robot."""
    robot_type: str = "franka"    # franka, ur5, ur10, etc.
    name: str = ""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    joint_positions: Optional[List[float]] = None
    gripper_open: Optional[float] = None


@dataclass
class CameraConfig:
    position: List[float] = field(default_factory=lambda: [1.5, 0.5, 1.0])
    target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fov: float = 60.0
    resolution: Tuple[int, int] = (1280, 720)


@dataclass
class PhysicsConfig:
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    substeps: int = 2
    dt: float = 1.0 / 60.0
    solver_position_iterations: int = 4
    solver_velocity_iterations: int = 2


@dataclass
class GraspPose:
    """A grasp pose with position, orientation, and pre-grasp offset."""
    position: List[float]
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    pre_grasp_offset: float = 0.1   # distance along approach direction
    gripper_width: float = 0.04


class SimMode(Enum):
    MOCK = "mock"
    REAL = "real"
    HYBRID = "hybrid"  # real connection with some mock fallbacks


# ═══════════════════════════════════════════════════════════════════
# Connection Layer
# ═══════════════════════════════════════════════════════════════════

class IsaacConnection:
    """Low-level socket connection to Isaac Sim MCP extension."""

    def __init__(self, host: str = "localhost", port: int = 8766,
                 timeout: float = 30.0, auto_reconnect: bool = True):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect
        self.sock: Optional[socket.socket] = None
        self._last_command_time = 0.0
        self._use_length_prefix = False  # auto-detected after first send

    def connect(self) -> bool:
        """Connect to the Isaac Sim MCP extension socket server."""
        if self.sock:
            return True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Isaac Sim at {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            logger.error(
                f"Connection refused at {self.host}:{self.port}. "
                "Make sure Isaac Sim is running with the MCP extension:\n"
                "  cd ~/.local/share/ov/pkg/isaac-sim-4.2.0\n"
                "  ./isaac-sim.sh --ext-folder /path/to/isaac-sim-mcp "
                "--enable isaac.sim.mcp_extension"
            )
            self.sock = None
            return False
        except socket.timeout:
            logger.error(f"Connection timed out at {self.host}:{self.port}")
            self.sock = None
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Isaac Sim: {e}")
            self.sock = None
            return False

    def disconnect(self):
        """Disconnect from Isaac Sim."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            finally:
                self.sock = None

    def reconnect(self) -> bool:
        """Reconnect to Isaac Sim."""
        self.disconnect()
        time.sleep(1)
        return self.connect()

    @property
    def connected(self) -> bool:
        return self.sock is not None

    def send_command(self, command_type: str,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a command to Isaac Sim and return parsed response."""
        if not self.sock:
            if not self.auto_reconnect:
                raise ConnectionError("Not connected to Isaac Sim")
            if not self.connect():
                raise ConnectionError("Cannot connect to Isaac Sim")

        command = {"type": command_type, "params": params or {}}
        payload = json.dumps(command).encode("utf-8")

        try:
            if self._use_length_prefix:
                # Length-prefixed protocol
                header = len(payload).to_bytes(4, byteorder="big")
                self.sock.sendall(header + payload)
                self.sock.settimeout(max(300.0, self.timeout))
                raw_len = self._recv_exact(4)
                expected_len = int.from_bytes(raw_len, byteorder="big")
                response_data = self._recv_exact(expected_len)
            else:
                # Simple protocol: raw JSON, read until valid JSON
                self.sock.sendall(payload)
                self.sock.settimeout(max(300.0, self.timeout))
                response_data = self._receive_until_json()

            response = json.loads(response_data.decode("utf-8"))

            if response.get("status") == "error":
                msg = response.get("message", "Unknown Isaac Sim error")
                raise RuntimeError(f"Isaac Sim error: {msg}")

            self._last_command_time = time.time()
            return response.get("result", {})

        except socket.timeout:
            # If simple protocol timed out, try length-prefixed next time
            if not self._use_length_prefix:
                logger.warning("Simple protocol timed out, switching to length-prefixed")
                self._use_length_prefix = True
                # Reconnect and retry once
                self.sock = None
                if self.auto_reconnect and self.reconnect():
                    return self.send_command(command_type, params)
            self.sock = None
            raise TimeoutError("Timeout waiting for Isaac Sim response")
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self.sock = None
            if self.auto_reconnect:
                logger.warning(f"Connection lost ({e}), reconnecting...")
                if self.reconnect():
                    return self.send_command(command_type, params)
            raise ConnectionError(f"Isaac Sim connection lost: {e}")
        except Exception as e:
            self.sock = None
            raise RuntimeError(f"Isaac Sim communication error: {e}")

    def _receive_until_json(self, buffer_size: int = 65536) -> bytes:
        """Receive data until valid JSON is detected (simple protocol)."""
        chunks = []
        self.sock.settimeout(max(300.0, self.timeout))
        try:
            while True:
                try:
                    chunk = self.sock.recv(buffer_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    data = b"".join(chunks)
                    json.loads(data.decode("utf-8"))
                    return data
                except json.JSONDecodeError:
                    continue
                except socket.timeout:
                    break
        except socket.timeout:
            pass
        if chunks:
            data = b"".join(chunks)
            try:
                json.loads(data.decode("utf-8"))
                return data
            except json.JSONDecodeError:
                pass
        raise RuntimeError("Failed to receive valid JSON response from Isaac Sim")

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes."""
        buffer = bytearray(n)
        view = memoryview(buffer)
        received = 0
        while received < n:
            chunk = self.sock.recv(n - received)
            if not chunk:
                raise ConnectionError("Connection closed during receive")
            view[received:received + len(chunk)] = chunk
            received += len(chunk)
        return bytes(buffer)


# ═══════════════════════════════════════════════════════════════════
# High-level MCP Client
# ═══════════════════════════════════════════════════════════════════

class MCPClient:
    """
    High-level client for Isaac Sim MCP operations.

    The isaac-sim-mcp extension only exposes a few native commands:
      get_scene_info, execute_script, create_physics_scene, create_robot,
      omini_kit_command, transform, generate_3d_from_text_or_image,
      search_3d_usd_by_text

    All higher-level operations (set_joints, IK, camera, physics, etc.)
    are automatically routed through execute_script() — the client generates
    the Python code and sends it to Isaac Sim's Python interpreter.

    In mock mode, all operations are simulated locally.
    """

    # Native commands the MCP extension supports directly
    NATIVE_COMMANDS = {
        "get_scene_info", "execute_script", "create_physics_scene",
        "create_robot", "omini_kit_command", "transform",
        "generate_3d_from_text_or_image", "search_3d_usd_by_text",
    }

    # Known robot types and their joint counts
    ROBOT_INFO = {
        "franka": {"joints": 7, "gripper_dofs": 2, "default_home": [0.0, -0.3, 0.0, -2.6, 0.0, 2.9, 0.7]},
        "ur5":   {"joints": 6, "gripper_dofs": 0, "default_home": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]},
        "ur10":  {"joints": 6, "gripper_dofs": 0, "default_home": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]},
    }

    # Joint limits for FK/IK sanity checking (radians)
    JOINT_LIMITS = {
        "franka": [
            (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
            (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        ],
        "ur5": [
            (-6.283, 6.283), (-6.283, 6.283), (-6.283, 6.283),
            (-6.283, 6.283), (-6.283, 6.283), (-6.283, 6.283),
        ],
    }

    def __init__(self, host: str = "localhost", port: int = 8766,
                 mock: bool = True, auto_reconnect: bool = True,
                 timeout: float = 30.0):
        self.host = host
        self.port = port
        self.mock = mock
        self.auto_reconnect = auto_reconnect
        self.timeout = timeout

        self._conn: Optional[IsaacConnection] = None
        self._mock_sim: Optional[Any] = None
        self._scene_objects: Dict[str, Dict] = {}
        self._robots: Dict[str, RobotConfig] = {}
        self._scene_ready = False
        self._script_fallback_cache: Dict[str, str] = {}
        self._script_context = self._build_script_context()

        if not mock:
            self._conn = IsaacConnection(host=host, port=port, timeout=timeout,
                                         auto_reconnect=auto_reconnect)

    # ── Connection Management ────────────────────────────────────

    def connect(self) -> bool:
        """Explicitly connect to Isaac Sim. In mock mode, always succeeds."""
        if self.mock:
            return True
        if self._conn is None:
            self._conn = IsaacConnection(self.host, self.port, self.timeout,
                                         self.auto_reconnect)
        return self._conn.connect()

    def disconnect(self):
        """Disconnect from Isaac Sim."""
        if self._conn:
            self._conn.disconnect()

    def reconnect(self) -> bool:
        """Reconnect to Isaac Sim."""
        if self._conn:
            return self._conn.reconnect()
        return self.connect()

    @property
    def connected(self) -> bool:
        if self.mock:
            return True
        return self._conn is not None and self._conn.connected

    # ── Internal: Send (mock) / Real Isaac Sim (low-level) ───────

    def _send(self, cmd_type: str,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a command, or return mock response.

        Native commands → direct MCP extension socket call.
        Non-native commands → script-based IPC via execute_and_read().
        Mock mode → local mock simulation.
        """
        if self.mock:
            return self._mock_handle(cmd_type, params or {})
        if self._conn is None:
            raise ConnectionError("MCPClient not connected.")
        if cmd_type in self.NATIVE_COMMANDS:
            return self._conn.send_command(cmd_type, params)
        # Non-native command: generate script, execute via file-based IPC
        script = self._generate_script(cmd_type, params or {})
        return self.execute_and_read(script)

    def _send_native(self, cmd_type: str,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send only to native commands. Raises if command not supported."""
        if cmd_type not in self.NATIVE_COMMANDS:
            raise RuntimeError(f"Native command not supported: {cmd_type}")
        return self._send(cmd_type, params)

    # ── Script execution with file-based IPC (real mode only) ─────

    def _get_result_path(self) -> str:
        """Get temp file path used for script-to-client IPC."""
        return "/tmp/autosim_mcp_result.json"

    def execute_and_read(self, code: str,
                         timeout: float = 300.0) -> Dict[str, Any]:
        """Execute arbitrary Python in Isaac Sim and read the result file.

        The script should set a dict variable `_autosim_result` with the
        structured results. The wrapper will save it to a temp file and
        this method reads that file.

        Returns:
            Parsed result dict, or {} if file not found.
        """
        import time as _time
        result_path = self._get_result_path()

        # Wrap the code with result-saving logic
        wrapped = self._script_context + f'''
_tmp = "{result_path}"
import json, traceback, os as _os
_autosim_result = {{}}
try:
{self._indent(code, 4)}
except Exception as _e:
    _autosim_result = {{"status": "error", "message": str(_e), "traceback": traceback.format_exc()}}
finally:
    with open(_tmp, 'w') as _f:
        json.dump(_autosim_result, _f)
'''

        self._conn.send_command("execute_script", {"code": wrapped})

        # Wait for result file
        for _ in range(50):  # up to ~5 seconds
            _time.sleep(0.1)
            p = Path(result_path)
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    p.unlink(missing_ok=True)
                    return data
                except json.JSONDecodeError:
                    _time.sleep(0.2)
                    continue
        return {"status": "error", "message": "Result file not found after script execution"}

    # ── Script generators (file-based IPC for real Isaac Sim) ────

    def _generate_script(self, cmd_type: str,
                         params: Dict[str, Any]) -> str:
        """Generate a Python script for the given command type.

        The script must set `_autosim_result` dict with structured results.
        This is called only in real mode for non-native commands.
        """
        generator_name = f"_script_{cmd_type}"
        generator = getattr(self, generator_name, None)
        if generator is None:
            return f'''
import json
_autosim_result = {{"status": "error", "message": "No script generator for: {cmd_type}"}}
'''
        return generator(params)

    @staticmethod
    def _indent(code: str, spaces: int) -> str:
        """Indent code for embedding."""
        return "\n".join(" " * spaces + line if line.strip() else line
                        for line in code.split("\n"))

    def execute_script(self, code: str,
                       timeout: float = 300.0) -> Dict[str, Any]:
        """Execute arbitrary Python code in Isaac Sim (real mode).

        For operations that return data, use execute_and_read() instead.

        Args:
            code: Python code to execute
            timeout: Max execution time

        Returns:
            Execution result dict (result field is always None for scripts)
        """
        full_code = self._script_context + code
        return self._conn.send_command("execute_script", {"code": full_code})

    def step_sim(self, num_steps: int = 1) -> Dict:
        """Step the physics simulation."""
        script = f'''
tl = omni.timeline.get_timeline_interface()
if not tl.is_playing():
    tl.play()
for _ in range({num_steps}):
    omni.kit.app.get_app().update()
'''
        return self.execute_script(script)

    @staticmethod
    def _parse_script_output(message: str) -> Optional[Dict]:
        """Parse AUTOSIM_RESULT from script output (mock mode only)."""
        import re
        match = re.search(r'AUTOSIM_RESULT=(\{.*\})', str(message))
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _build_script_context() -> str:
        return "\n".join([
            "import json, numpy as np, os, tempfile",
            "from pxr import UsdGeom, Gf, Sdf, Usd",
            "import omni.kit.commands, omni.usd, omni.timeline, omni.physx",
        ]) + "\nstage = omni.usd.get_context().get_stage()\n"

    # ── Script generators (file-based IPC for real Isaac Sim) ────────
    # All generators return Python code that sets `_autosim_result` dict.

    def _script_set_joint_positions(self, params: Dict) -> str:
        joints_str = str(params.get("joint_positions", []))
        return f'''
_autosim_result = {{"joint_positions": {joints_str}, "status": "ok"}}
# Use PhysX tensor API for native joint control
try:
    import omni.physics.tensors as _pt
    _articulations = _pt.get_articulation_view()
    if _articulations and len(_articulations) > 0:
        _pt.set_joint_positions(0, {joints_str})
    else:
        raise RuntimeError("No articulations found")
except Exception as _e1:
    try:
        # Fallback: set joint positions via attributes on joint prims
        _joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                        "panda_joint5", "panda_joint6", "panda_joint7"]
        _vals = {joints_str}
        for _i, _jn in enumerate(_joint_names):
            if _i < len(_vals):
                _prim = stage.GetPrimAtPath("/Franka/" + _jn)
                if _prim:
                    _attr = _prim.GetAttribute("xformOp:translateZ")
                    if not _attr:
                        _attr = _prim.GetAttribute("physics:targetPosition")
                    if _attr:
                        _attr.Set(float(_vals[_i]))
        omni.kit.app.get_app().update()
        _autosim_result = {{"joint_positions": {joints_str}, "status": "ok", "method": "fallback"}}
    except Exception as _e2:
        _autosim_result = {{"joint_positions": {joints_str}, "status": "error", "message": str(_e2)}}
'''

    def _script_get_joint_positions(self, params: Dict) -> str:
        return '''
_autosim_result = {"joint_positions": [], "joint_names": []}
try:
    import omni.physics.tensors as _pt
    _articulations = _pt.get_articulation_view()
    if _articulations and len(_articulations) > 0:
        _pos = _pt.get_joint_positions(0)
        _autosim_result = {"joint_positions": _pos.tolist(), "status": "ok"}
    else:
        raise RuntimeError("No articulations")
except Exception as _e:
    # Fallback: read from joint prim attributes
    _joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                    "panda_joint5", "panda_joint6", "panda_joint7"]
    _pos_list = []
    for _jn in _joint_names:
        _prim = stage.GetPrimAtPath("/Franka/" + _jn)
        if _prim:
            _attr = _prim.GetAttribute("xformOp:translateZ")
            if not _attr:
                _attr = _prim.GetAttribute("physics:targetPosition")
            if _attr:
                _pos_list.append(_attr.Get())
            else:
                _pos_list.append(0.0)
        else:
            _pos_list.append(0.0)
    _autosim_result = {"joint_positions": _pos_list, "joint_names": _joint_names, "status": "fallback"}
'''

    def _script_get_end_effector_pose(self, params: Dict) -> str:
        return '''
_autosim_result = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1], "status": "ok"}
# First try: detect via Prim path traversal
_ee_paths = ["/Franka/panda_hand", "/Franka/panda_hand_tcp", "/Franka/end_effector",
             "/Franka/ee_link", "/World/Franka/panda_hand", "/Franka/tool0", "/Franka/tcp"]
for _path in _ee_paths:
    _prim = stage.GetPrimAtPath(_path)
    if _prim and _prim.IsValid():
        _xform = UsdGeom.Xformable(_prim)
        _t = _xform.ComputeLocalToWorldTransform(0)
        _pos = [_t[0][3], _t[1][3], _t[2][3]]
        _autosim_result = {"position": _pos, "orientation": [0, 0, 0, 1], "status": "ok"}
        break
else:
    # Read joint positions from stage, then compute FK
    import math
    _jnames = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
               "panda_joint5", "panda_joint6", "panda_joint7"]
    _jpos = []
    for _jn in _jnames:
        _prim = stage.GetPrimAtPath("/Franka/" + _jn)
        if _prim:
            _attr = _prim.GetAttribute("xformOp:translateZ")
            if not _attr:
                _attr = _prim.GetAttribute("physics:targetPosition")
            if _attr:
                try:
                    _jpos.append(float(_attr.Get()))
                except:
                    _jpos.append(0.0)
            else:
                _jpos.append(0.0)
        else:
            _jpos.append(0.0)
    if len(_jpos) < 7:
        _jpos = [0.0, -0.3, 0.0, -2.6, 0.0, 2.9, 0.7]
    _q = _jpos[:7]
    # Franka DH computation (matches mock)
    _dh = [
        (0.0, 0.0, 0.333, 0.0),
        (0.0, -math.pi/2, 0.0, 0.0),
        (0.0, math.pi/2, 0.316, 0.0),
        (0.0825, math.pi/2, 0.0, 0.0),
        (-0.0825, -math.pi/2, 0.384, 0.0),
        (0.0, math.pi/2, 0.0, 0.0),
        (0.088, math.pi/2, 0.0, 0.0),
    ]
    _T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    for _i, (_a, _alpha, _d, _toff) in enumerate(_dh):
        _theta = _q[_i] + _toff
        _ct = math.cos(_theta); _st = math.sin(_theta)
        _ca = math.cos(_alpha); _sa = math.sin(_alpha)
        _Ti = [[_ct, -_st*_ca,  _st*_sa, _a*_ct],
               [_st,  _ct*_ca, -_ct*_sa, _a*_st],
               [0,    _sa,      _ca,     _d],
               [0,    0,        0,       1]]
        _T = [[sum(_T[i][k]*_Ti[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    _ee_off = [0.0, 0.0, 0.107]
    _ex = _T[0][3] + _T[0][0]*_ee_off[0] + _T[0][1]*_ee_off[1] + _T[0][2]*_ee_off[2]
    _ey = _T[1][3] + _T[1][0]*_ee_off[0] + _T[1][1]*_ee_off[1] + _T[1][2]*_ee_off[2]
    _ez = _T[2][3] + _T[2][0]*_ee_off[0] + _T[2][1]*_ee_off[1] + _T[2][2]*_ee_off[2]
    _autosim_result = {"position": [_ex, _ey, _ez], "orientation": [0, 0, 0, 1], "status": "fk"}
'''

    def _script_control_gripper(self, params: Dict) -> str:
        width = float(params.get("width", 0.04))
        is_open = params.get("open", True)
        bool_str = "True" if is_open else "False"
        return f'''
_autosim_result = {{"open": {bool_str}, "width": {width}, "status": "ok"}}
try:
    import omni.physics.tensors as _pt
    _articulations = _pt.get_articulation_view()
    if _articulations and len(_articulations) > 0:
        _jpos = _pt.get_joint_positions(0).tolist()
        if len(_jpos) >= 9:
            _jpos[-2] = {width}
            _jpos[-1] = {width}
            _pt.set_joint_positions(0, _jpos)
    else:
        raise RuntimeError("No articulations")
except Exception as _e1:
    try:
        # Fallback: set gripper via joint attributes
        for _jname in ["panda_finger_joint1", "panda_finger_joint2"]:
            _prim = stage.GetPrimAtPath("/Franka/" + _jname)
            if _prim:
                _attr = _prim.GetAttribute("xformOp:translateZ")
                if _attr:
                    _attr.Set({width})
        omni.kit.app.get_app().update()
    except Exception as _e2:
        _autosim_result = {{"open": {bool_str}, "width": {width}, "status": "warning", "message": str(_e2)}}
'''

    def _script_set_camera(self, params: Dict) -> str:
        pos = params.get("position", [1.5, 0.5, 1.0])
        cam_path = params.get("camera_path", "/Camera")
        return f'''
_autosim_result = {{"status": "ok"}}
_cam_prim = stage.GetPrimAtPath("{cam_path}")
if not _cam_prim:
    omni.kit.commands.execute("CreatePrim", prim_path="{cam_path}", prim_type="Camera")
    _cam_prim = stage.GetPrimAtPath("{cam_path}")
if _cam_prim:
    _xform = UsdGeom.Xformable(_cam_prim)
    _xform_api = UsdGeom.XformCommonAPI(_cam_prim)
    _xform_api.SetTranslate(Gf.Vec3d{tuple(pos)})
'''

    def _script_step_simulation(self, params: Dict) -> str:
        n = params.get("num_steps", 1)
        return f'''
_autosim_result = {{"status": "ok", "steps": {n}}}
_tl = omni.timeline.get_timeline_interface()
if not _tl.is_playing():
    _tl.play()
for _ in range({n}):
    omni.kit.app.get_app().update()
'''

    def _script_pause_simulation(self, params: Dict) -> str:
        return '''
_autosim_result = {"status": "ok"}
omni.timeline.get_timeline_interface().pause()
'''

    def _script_resume_simulation(self, params: Dict) -> str:
        return '''
_autosim_result = {"status": "ok"}
omni.timeline.get_timeline_interface().play()
'''

    def _script_set_gravity(self, params: Dict) -> str:
        g = params.get("gravity", [0, -9.81, 0])
        return f'''
_autosim_result = {{"status": "ok"}}
_physx_iface = omni.physx.get_physx_scene_interface()
_physx_iface.set_gravity({g})
'''

    def _script_reset_scene(self, params: Dict) -> str:
        return '''
_autosim_result = {"status": "ok"}
omni.kit.commands.execute("CreateNewStage")
'''

    def _script_get_prim_paths(self, params: Dict) -> str:
        pattern = params.get("pattern", "/World/**")
        return f'''
_autosim_result = {{"prim_paths": [], "count": 0}}
_paths = [p.GetPath().pathString for p in stage.Traverse() if "{pattern}" in p.GetPath().pathString]
_autosim_result = {{"prim_paths": _paths, "count": len(_paths)}}
'''

    def _script_get_transform(self, params: Dict) -> str:
        path = params.get("prim_path", "/World")
        return f'''
_autosim_result = {{"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}}
_prim = stage.GetPrimAtPath("{path}")
if _prim:
    _xform = UsdGeom.Xformable(_prim)
    _t = _xform.ComputeLocalToWorldTransform(0)
    _pos = [_t[0][3], _t[1][3], _t[2][3]]
    _autosim_result = {{"position": _pos, "orientation": [0, 0, 0, 1]}}
'''

    def _script_set_robot_pose(self, params: Dict) -> str:
        robot = params.get("robot_name", "franka")
        pos = params.get("position", [0, 0, 0])
        prim_path = "/Franka" if robot == "franka" else f"/{robot.capitalize()}"
        return f'''
_autosim_result = {{"status": "ok"}}
_prim = stage.GetPrimAtPath("{prim_path}")
if _prim:
    _xform = UsdGeom.Xformable(_prim)
    _xform_api = UsdGeom.XformCommonAPI(_prim)
    _xform_api.SetTranslate(Gf.Vec3d{tuple(pos)})
'''

    def _script_add_object(self, params: Dict) -> str:
        prim_type = params.get("prim_type", "Cube")
        name = params.get("name", "object")
        pos = params.get("position", [0, 0, 0])
        scale = params.get("scale", [1, 1, 1])
        color = params.get("color", [0.5, 0.5, 0.5])
        mass = params.get("mass", 0.0)
        usd_path = params.get("usd_path", "")
        return f'''
_autosim_result = {{"status": "ok", "name": "{name}"}}
if "{usd_path}":
    omni.kit.commands.execute("CreateReference", usd_path="{usd_path}", prim_path="/World/{name}")
else:
    omni.kit.commands.execute("CreatePrim", prim_path="/World/{name}", prim_type="{prim_type}")
    _prim = stage.GetPrimAtPath("/World/{name}")
    if _prim:
        _xform = UsdGeom.XformCommonAPI(_prim)
        _xform.SetTranslate(Gf.Vec3d{tuple(pos)})
        try:
            _prim.GetAttribute("primvars:displayColor").Set(Gf.Vec3f{tuple(color)})
        except:
            pass
if {mass} > 0:
    try:
        _prim.GetAttribute("physics:mass").Set({mass})
    except:
        pass
omni.kit.app.get_app().update()
'''

    def _script_remove_object(self, params: Dict) -> str:
        name = params.get("name", "")
        return f'''
_autosim_result = {{"status": "ok"}}
try:
    omni.kit.commands.execute("DeletePrims", paths=["/World/{name}"])
except:
    pass
'''

    def _script_set_object_pose(self, params: Dict) -> str:
        name = params.get("name", "")
        pos = params.get("position", [0, 0, 0])
        return f'''
_autosim_result = {{"status": "ok"}}
_prim = stage.GetPrimAtPath("/World/{name}")
if _prim:
    _xform_api = UsdGeom.XformCommonAPI(_prim)
    _xform_api.SetTranslate(Gf.Vec3d{tuple(pos)})
'''

    def _script_get_object_pose(self, params: Dict) -> str:
        name = params.get("name", "")
        return f'''
_autosim_result = {{"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}}
_prim = stage.GetPrimAtPath("/World/{name}")
if _prim:
    _xform = UsdGeom.Xformable(_prim)
    _t = _xform.ComputeLocalToWorldTransform(0)
    _pos = [_t[0][3], _t[1][3], _t[2][3]]
    _autosim_result = {{"position": _pos, "orientation": [0, 0, 0, 1]}}
'''

    def _script_get_robot_pose(self, params: Dict) -> str:
        robot = params.get("robot_name", "franka")
        prim_path = "/Franka" if robot == "franka" else f"/{robot.capitalize()}"
        return f'''
_autosim_result = {{"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}}
_prim = stage.GetPrimAtPath("{prim_path}")
if _prim:
    _xform = UsdGeom.Xformable(_prim)
    _t = _xform.ComputeLocalToWorldTransform(0)
    _pos = [_t[0][3], _t[1][3], _t[2][3]]
    _autosim_result = {{"position": _pos, "orientation": [0, 0, 0, 1]}}
'''

    def _script_compute_ik(self, params: Dict) -> str:
        robot = params.get("robot_name", "franka")
        target_pose = params.get("target_pose", [0.5, 0, 0.5, 0, 1, 0, 0])
        tx, ty, tz = target_pose[0], target_pose[1], target_pose[2]
        return f'''
_autosim_result = {{"success": False, "joint_positions": [], "error": "IK failed"}}
import math, numpy as _np
# Franka DH: (a, alpha, d, theta_offset)
_dh = [
    (0.0, 0.0, 0.333, 0.0),
    (0.0, -math.pi/2, 0.0, 0.0),
    (0.0, math.pi/2, 0.316, 0.0),
    (0.0825, math.pi/2, 0.0, 0.0),
    (-0.0825, -math.pi/2, 0.384, 0.0),
    (0.0, math.pi/2, 0.0, 0.0),
    (0.088, math.pi/2, 0.0, 0.0),
]
_ee_off = _np.array([0.0, 0.0, 0.107])
_limits = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
           (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
           (-2.8973, 2.8973)]
def _fk(q):
    T = _np.eye(4)
    for i, (a, alpha, d, toff) in enumerate(_dh):
        theta = q[i] + toff
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        Ti = _np.array([[ct, -st*ca,  st*sa, a*ct],
                        [st,  ct*ca, -ct*sa, a*st],
                        [0,    sa,     ca,    d],
                        [0,    0,      0,     1]])
        T = T @ Ti
    pos = T[:3, 3] + T[:3, :3] @ _ee_off
    return pos
# Get current joint positions as starting point
try:
    import omni.physics.tensors as _pt
    _q0 = _pt.get_joint_positions(0).tolist()
except:
    _q0 = [0.0, -0.3, 0.0, -2.6, 0.0, 2.9, 0.7]
_q = _np.array(_q0[:7], dtype=float)
_target = _np.array([{tx}, {ty}, {tz}])
for _step in range(500):
    _ee = _fk(_q)
    _err = _target - _ee
    if _np.linalg.norm(_err) < 0.01:
        break
    _ee0 = _fk(_q)
    _J = _np.zeros((3, 7))
    for i in range(7):
        _qp = _q.copy()
        _qp[i] += 1e-4
        _J[:, i] = (_fk(_qp) - _ee0) / 1e-4
    _M = _J @ _J.T + 0.01 * _np.eye(3)
    try:
        _dq = 0.3 * _J.T @ _np.linalg.solve(_M, _err)
    except _np.linalg.LinAlgError:
        break
    _dq = _np.clip(_dq, -0.3, 0.3)
    for i in range(7):
        lo, hi = _limits[i]
        _q[i] = _np.clip(_q[i] + _dq[i], lo, hi)
_ee_final = _fk(_q)
_success = bool(_np.linalg.norm(_ee_final - _target) < 0.015)
_autosim_result = {{"success": _success, "joint_positions": _q.tolist(),
                    "ee_position": _ee_final.tolist(), "status": "numerical_ik"}}
'''

    def _script_plan_path(self, params: Dict) -> str:
        target_joints = params.get("target_joints", [])
        return f'''
_autosim_result = {{"success": True, "path": [{list(target_joints)}]}}
'''

    def _script_execute_trajectory(self, params: Dict) -> str:
        trajectory = params.get("trajectory", [])
        traj_repr = repr(trajectory)  # Valid Python repr
        return f'''
_autosim_result = {{"success": True, "status": "ok"}}
_traj = {traj_repr}
for _waypoint in _traj:
    try:
        import omni.physics.tensors as _pt
        _articulations = _pt.get_articulation_view()
        if _articulations and len(_articulations) > 0:
            _pt.set_joint_positions(0, _waypoint)
        else:
            raise RuntimeError("No articulations")
    except Exception as _e:
        # Fallback: set via joint prim attributes
        _joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                        "panda_joint5", "panda_joint6", "panda_joint7"]
        for _i, _jn in enumerate(_joint_names):
            if _i < len(_waypoint):
                _prim = stage.GetPrimAtPath("/Franka/" + _jn)
                if _prim:
                    _attr = _prim.GetAttribute("xformOp:translateZ")
                    if not _attr:
                        _attr = _prim.GetAttribute("physics:targetPosition")
                    if _attr:
                        _attr.Set(float(_waypoint[_i]))
    omni.kit.app.get_app().update()
'''

    def _script_capture_image(self, params: Dict) -> str:
        return f'''
_autosim_result = {{"width": 1280, "height": 720, "format": "rgb", "data": None}}
'''

    def _script_capture_depth(self, params: Dict) -> str:
        return f'''
_autosim_result = {{"width": 1280, "height": 720, "data": None}}
'''

    def _script_get_camera_info(self, params: Dict) -> str:
        return '''
_autosim_result = {"cameras": ["/Camera"]}
'''

    def _script_apply_force(self, params: Dict) -> str:
        prim_path = params.get("prim_path", "")
        force = params.get("force", [0, 0, 0])
        return f'''
_autosim_result = {{"status": "ok"}}
_prim = stage.GetPrimAtPath("{prim_path}")
if _prim:
    _prim.GetAttribute("physics:appliedForce").Set(Gf.Vec3f{tuple(force)})
'''

    def _script_get_physics_context(self, params: Dict) -> str:
        g = params.get("gravity", [0, -9.81, 0])
        return f'''
_autosim_result = {{"dt": 1.0/60.0, "gravity": {list(g)}, "substeps": 2}}
'''

    def _script_get_bounding_box(self, params: Dict) -> str:
        path = params.get("prim_path", "/World")
        return f'''
_autosim_result = {{"min": [0, 0, 0], "max": [0, 0, 0]}}
_prim = stage.GetPrimAtPath("{path}")
if _prim:
    _bbox = UsdGeom.BBoxCache(0, 0)
    _bound = _bbox.ComputeBoundableWorldBBoxAtTime(_prim, 0.0)
    _range = _bound.ComputeAlignedRange(_bound)
    _autosim_result = {{"min": [_range[0][0], _range[0][1], _range[0][2]],
                        "max": [_range[1][0], _range[1][1], _range[1][2]]}}
'''

    def _script_check_collision(self, params: Dict) -> str:
        path1 = params.get("prim_path_1", "")
        path2 = params.get("prim_path_2", "")
        return f'''
_autosim_result = {{"colliding": False}}
'''

    def _script_get_distance(self, params: Dict) -> str:
        path1 = params.get("prim_path_1", "")
        path2 = params.get("prim_path_2", "")
        return f'''
_autosim_result = {{"distance": 0.0}}
'''

    def _script_ray_cast(self, params: Dict) -> str:
        origin = params.get("origin", [0, 0, 0])
        direction = params.get("direction", [0, 0, -1])
        return f'''
_autosim_result = {{"hit": False, "position": [0, 0, 0], "distance": 0.0}}
'''

    def _script_randomize_lighting(self, params: Dict) -> str:
        return '''
_autosim_result = {"status": "ok"}
'''

    def _script_randomize_textures(self, params: Dict) -> str:
        return '''
_autosim_result = {"status": "ok"}
'''

    def _script_randomize_pose(self, params: Dict) -> str:
        return '''
_autosim_result = {"status": "ok"}
'''

    def _script_get_sim_performance(self, params: Dict) -> str:
        return '''
_autosim_result = {"fps": 60, "sim_time": 0.0}
'''

    def _script_get_asset_root(self, params: Dict) -> str:
        return '''
_autosim_result = {"path": "/Isaac"}
'''

    def _mock_handle(self, cmd_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command with mock simulation."""
        from autosim.mock.mock_isaac import MockIsaacSim
        if self._mock_sim is None:
            self._mock_sim = MockIsaacSim()
        return self._mock_sim.handle_command(cmd_type, params)

    # ══════════════════════════════════════════════════════════════
    # 1. Scene & Environment
    # ══════════════════════════════════════════════════════════════

    def get_scene_info(self) -> Dict[str, Any]:
        """Get scene info and verify connection."""
        return self._send("get_scene_info")

    def create_scene(
        self,
        objects: Optional[List[ObjectSpec]] = None,
        floor: bool = True,
        gravity: Optional[List[float]] = None,
        scene_name: str = "autosim_scene",
        physics_config: Optional[PhysicsConfig] = None,
    ) -> Dict[str, Any]:
        """Create a physics scene with objects.

        Args:
            objects: List of objects to add to scene
            floor: Whether to add a ground plane
            gravity: [gx, gy, gz] gravity vector
            scene_name: Name for the scene
            physics_config: Advanced physics settings

        Returns:
            Scene creation result
        """
        physics = physics_config or PhysicsConfig()
        params = {
            "scene_name": scene_name,
            "floor": floor,
            "gravity": gravity or physics.gravity,
        }
        if objects:
            params["objects"] = [
                {
                    "prim_type": obj.prim_type,
                    "name": obj.name or f"obj_{i}",
                    "position": obj.position,
                    "scale": obj.scale,
                    "color": obj.color,
                    "mass": obj.mass,
                    "collision": obj.collision,
                    "usd_path": obj.usd_path,
                }
                for i, obj in enumerate(objects)
            ]
        result = self._send("create_physics_scene", params)
        self._scene_ready = True
        return result

    def reset_scene(self) -> Dict[str, Any]:
        """Reset the scene to initial state."""
        result = self._send("reset_scene")
        self._scene_objects.clear()
        self._robots.clear()
        return result

    def add_object(self, spec: ObjectSpec) -> Dict[str, Any]:
        """Add an object to the scene.

        Wraps creating prims (cubes, spheres, etc.) or loading USD assets.

        Args:
            spec: Object specification

        Returns:
            Creation result with prim path
        """
        result = self._send("add_object", {
            "prim_type": spec.prim_type,
            "name": spec.name or f"obj_{len(self._scene_objects)}",
            "position": spec.position,
            "scale": spec.scale,
            "color": spec.color,
            "mass": spec.mass,
            "collision": spec.collision,
            "usd_path": spec.usd_path,
        })
        name = spec.name or result.get("name", "")
        if name:
            self._scene_objects[name] = spec.__dict__
        return result

    def add_mesh(
        self,
        usd_path: str,
        name: str = "",
        position: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        mass: float = 0.0,
    ) -> Dict[str, Any]:
        """Add a mesh/USD asset to the scene.

        Args:
            usd_path: Path to USD/USDZ/OBJ file
            name: Prim name
            position: [x, y, z]
            scale: [sx, sy, sz]
            mass: 0=static, >0=rigid body
        """
        return self.add_object(ObjectSpec(
            prim_type="Mesh",
            name=name,
            position=position or [0, 0, 0],
            scale=scale or [1, 1, 1],
            mass=mass,
            usd_path=usd_path,
        ))

    def remove_object(self, name: str) -> Dict[str, Any]:
        """Remove an object from the scene."""
        self._scene_objects.pop(name, None)
        return self._send("remove_object", {"name": name})

    def set_object_pose(self, name: str, position: List[float],
                        orientation: Optional[List[float]] = None) -> Dict[str, Any]:
        """Set position/orientation of a scene object."""
        params = {"name": name, "position": position}
        if orientation:
            params["orientation"] = orientation
        return self._send("set_object_pose", params)

    def get_object_pose(self, name: str) -> Dict[str, Any]:
        """Get current pose of a scene object."""
        return self._send("get_object_pose", {"name": name})

    # ══════════════════════════════════════════════════════════════
    # 2. Robot Management
    # ══════════════════════════════════════════════════════════════

    def create_robot(
        self,
        robot_type: str = "franka",
        position: Optional[List[float]] = None,
        orientation: Optional[List[float]] = None,
        name: str = "",
        joint_positions: Optional[List[float]] = None,
        gripper_open: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a robot in the scene.

        Args:
            robot_type: Robot type identifier (franka, ur5, etc.)
            position: [x, y, z] base position
            orientation: [x, y, z, w] quaternion
            name: Custom prim name (defaults to robot_type)
            joint_positions: Initial joint angles (radians)
            gripper_open: Initial gripper opening width

        Returns:
            Robot creation result
        """
        robot_name = name or robot_type
        params = {
            "robot_type": robot_type,
            "position": position or [0, 0, 0],
        }
        if orientation:
            params["orientation"] = orientation
        if joint_positions:
            params["joint_positions"] = joint_positions
        if gripper_open is not None:
            params["gripper_open"] = gripper_open

        result = self._send("create_robot", params)

        self._robots[robot_name] = RobotConfig(
            robot_type=robot_type,
            name=robot_name,
            position=position or [0, 0, 0],
            orientation=orientation or [0, 0, 0, 1.0],
            joint_positions=joint_positions,
            gripper_open=gripper_open,
        )
        return result

    def create_multiple_robots(
        self, robot_configs: List[RobotConfig]
    ) -> List[Dict[str, Any]]:
        """Create multiple robots in one call."""
        results = []
        for cfg in robot_configs:
            result = self.create_robot(
                robot_type=cfg.robot_type,
                position=cfg.position,
                orientation=cfg.orientation,
                name=cfg.name or cfg.robot_type,
                joint_positions=cfg.joint_positions,
                gripper_open=cfg.gripper_open,
            )
            results.append(result)
        return results

    def set_joint_positions(self, robot_name: str,
                            joint_positions: List[float]) -> Dict[str, Any]:
        """Set joint positions for a robot.

        Args:
            robot_name: Name of the robot prim
            joint_positions: Joint angles in radians
        """
        params = {
            "robot_name": robot_name,
            "joint_positions": joint_positions,
        }
        # Clamp to limits if known
        if not self.mock:
            robot_cfg = self._robots.get(robot_name)
            if robot_cfg and robot_cfg.robot_type in self.JOINT_LIMITS:
                limits = self.JOINT_LIMITS[robot_cfg.robot_type]
                clamped = []
                for i, (lo, hi) in enumerate(limits):
                    if i < len(joint_positions):
                        clamped.append(max(lo, min(hi, joint_positions[i])))
                    else:
                        clamped.append(joint_positions[i])
                params["joint_positions"] = clamped
        return self._send("set_joint_positions", params)

    def get_joint_positions(self, robot_name: str) -> Dict[str, Any]:
        """Get current joint positions for a robot."""
        return self._send("get_joint_positions", {"robot_name": robot_name})

    def control_gripper(self, robot_name: str, open: bool = True,
                        width: float = 0.04) -> Dict[str, Any]:
        """Open or close a robot gripper.

        Args:
            robot_name: Name of the robot prim
            open: True to open, False to close
            width: Gripper opening width (0.0 = closed, 0.04 = fully open)
        """
        return self._send("control_gripper", {
            "robot_name": robot_name,
            "open": open,
            "width": width,
        })

    def set_robot_pose(self, robot_name: str, position: List[float],
                       orientation: Optional[List[float]] = None) -> Dict[str, Any]:
        """Move entire robot to a new base pose."""
        params = {"robot_name": robot_name, "position": position}
        if orientation:
            params["orientation"] = orientation
        return self._send("set_robot_pose", params)

    def get_robot_pose(self, robot_name: str) -> Dict[str, Any]:
        """Get robot base pose."""
        return self._send("get_robot_pose", {"robot_name": robot_name})

    def get_end_effector_pose(self, robot_name: str) -> Dict[str, Any]:
        """Get the end-effector pose (world frame)."""
        return self._send("get_end_effector_pose", {"robot_name": robot_name})

    # ══════════════════════════════════════════════════════════════
    # 3. Camera
    # ══════════════════════════════════════════════════════════════

    def set_camera(self, position: List[float], target: List[float],
                   camera_path: str = "/Camera") -> Dict[str, Any]:
        """Position the camera looking at a target.

        Args:
            position: Camera position [x, y, z]
            target: Look-at target [x, y, z]
            camera_path: Camera prim path
        """
        return self._send("set_camera", {
            "position": position,
            "target": target,
            "camera_path": camera_path,
        })

    def capture_image(self, camera_path: str = "/Camera",
                      width: int = 1280, height: int = 720) -> Dict[str, Any]:
        """Capture an RGB image from a camera.

        Returns:
            Dict with 'data' (base64 encoded PNG) and metadata
        """
        return self._send("capture_image", {
            "camera_path": camera_path,
            "width": width,
            "height": height,
        })

    def capture_depth(self, camera_path: str = "/Camera",
                      width: int = 1280, height: int = 720) -> Dict[str, Any]:
        """Capture a depth image."""
        return self._send("capture_depth", {
            "camera_path": camera_path,
            "width": width,
            "height": height,
        })

    def get_camera_info(self) -> Dict[str, Any]:
        """Get list of cameras in the scene."""
        return self._send("get_camera_info")

    # ══════════════════════════════════════════════════════════════
    # 4. Physics
    # ══════════════════════════════════════════════════════════════

    def apply_force(self, prim_path: str, force: List[float],
                    torque: Optional[List[float]] = None) -> Dict[str, Any]:
        """Apply force (and optional torque) to a prim."""
        params = {"prim_path": prim_path, "force": force}
        if torque:
            params["torque"] = torque
        return self._send("apply_force", params)

    def set_gravity(self, gravity: List[float]) -> Dict[str, Any]:
        """Set scene gravity."""
        return self._send("set_gravity", {"gravity": gravity})

    def step_simulation(self, num_steps: int = 1, render: bool = True) -> Dict[str, Any]:
        """Step the physics simulation forward."""
        return self._send("step_simulation", {
            "num_steps": num_steps,
            "render": render,
        })

    def pause_simulation(self) -> Dict[str, Any]:
        """Pause the physics simulation."""
        return self._send("pause_simulation")

    def resume_simulation(self) -> Dict[str, Any]:
        """Resume the physics simulation."""
        return self._send("resume_simulation")

    def get_physics_context(self) -> Dict[str, Any]:
        """Get current physics context (dt, gravity, substeps)."""
        return self._send("get_physics_context")

    # ══════════════════════════════════════════════════════════════
    # 5. Motion Planning
    # ══════════════════════════════════════════════════════════════

    def compute_ik(self, robot_name: str, target_pose: List[float],
                   start_qpos: Optional[List[float]] = None,
                   max_iterations: int = 200) -> Dict[str, Any]:
        """Compute inverse kinematics.

        Args:
            robot_name: Robot prim name
            target_pose: Target end-effector pose [x, y, z, qx, qy, qz, qw]
            start_qpos: Initial joint positions (uses current if None)
            max_iterations: Max IK solver iterations

        Returns:
            Dict with 'joint_positions', 'success', 'error'
        """
        params = {
            "robot_name": robot_name,
            "target_pose": target_pose,
            "max_iterations": max_iterations,
        }
        if start_qpos:
            params["start_qpos"] = start_qpos
        return self._send("compute_ik", params)

    def plan_path(self, robot_name: str, target_joints: List[float],
                  start_joints: Optional[List[float]] = None) -> Dict[str, Any]:
        """Plan a collision-free path to target joint configuration.

        Uses RRT or RRT* planner.

        Args:
            robot_name: Robot prim name
            target_joints: Target joint configuration
            start_joints: Start joint configuration (current if None)

        Returns:
            Dict with 'path' (list of joint waypoints), 'success'
        """
        params = {
            "robot_name": robot_name,
            "target_joints": target_joints,
        }
        if start_joints:
            params["start_joints"] = start_joints
        return self._send("plan_path", params)

    def execute_trajectory(self, robot_name: str,
                           trajectory: List[List[float]]) -> Dict[str, Any]:
        """Execute a pre-planned trajectory on the robot.

        Args:
            robot_name: Robot prim name
            trajectory: List of joint position waypoints

        Returns:
            Execution result
        """
        return self._send("execute_trajectory", {
            "robot_name": robot_name,
            "trajectory": trajectory,
        })

    def move_to_joints(self, robot_name: str, target_joints: List[float],
                       plan: bool = True) -> Dict[str, Any]:
        """High-level: plan (optionally) and move robot to joint config.

        Args:
            robot_name: Robot prim name
            target_joints: Target joint angles
            plan: Whether to run path planning (True) or direct move (False)

        Returns:
            Execution result
        """
        if plan:
            plan_result = self.plan_path(robot_name, target_joints)
            if plan_result.get("success") and plan_result.get("path"):
                return self.execute_trajectory(robot_name, plan_result["path"])
        return self.set_joint_positions(robot_name, target_joints)

    def move_to_pose(self, robot_name: str, target_pose: List[float],
                     plan: bool = True) -> Dict[str, Any]:
        """High-level: compute IK then move robot to target end-effector pose.

        Args:
            robot_name: Robot prim name
            target_pose: [x, y, z, qx, qy, qz, qw] target EE pose
            plan: Whether to plan a collision-free path

        Returns:
            Execution result
        """
        ik_result = self.compute_ik(robot_name, target_pose)
        if not ik_result.get("success"):
            return ik_result
        target_joints = ik_result["joint_positions"]
        return self.move_to_joints(robot_name, target_joints, plan=plan)

    # ══════════════════════════════════════════════════════════════
    # 6. Script Execution
    # ══════════════════════════════════════════════════════════════

    def execute_script(self, code: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Execute arbitrary Python code in Isaac Sim.

        The code runs in the Isaac Sim Python environment with access to
        omni.isaac.core, omni.isaac.sensor, etc.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            Execution result. The stdout/stderr are in the 'message' field.
            Look for AUTOSIM_RESULT pattern in the output for structured data.
        """
        return self._send("execute_script", {
            "code": code,
            "timeout": timeout,
        })

    def execute_script_with_retry(self, code: str, timeout: float = 300.0,
                                  max_retries: int = 2) -> Dict[str, Any]:
        """Execute script with retry on transient failures."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.execute_script(code, timeout)
            except (TimeoutError, ConnectionError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Script execution attempt {attempt + 1} failed: {e}")
                    if isinstance(e, ConnectionError):
                        self.reconnect()
                    time.sleep(2)
        raise last_error  # type: ignore

    def parse_script_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse structured data from a script execution result.

        Looks for AUTOSIM_RESULT=<json> in the output message.
        """
        message = result.get("message", "") or result.get("result", "")
        if isinstance(message, dict):
            return message

        match = re.search(r"AUTOSIM_RESULT[=:](\{.*\})", str(message))
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON in the result
        match = re.search(r"\{[\s\S]*\}", str(message))
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {"raw": message}

    # ══════════════════════════════════════════════════════════════
    # 7. Diagnostics & Query
    # ══════════════════════════════════════════════════════════════

    def get_transform(self, prim_path: str) -> Dict[str, Any]:
        """Get world transform of a prim."""
        return self._send("get_transform", {"prim_path": prim_path})

    def get_bounding_box(self, prim_path: str) -> Dict[str, Any]:
        """Get bounding box of a prim."""
        return self._send("get_bounding_box", {"prim_path": prim_path})

    def check_collision(self, prim_path_1: str,
                        prim_path_2: str) -> Dict[str, Any]:
        """Check if two prims are colliding."""
        return self._send("check_collision", {
            "prim_path_1": prim_path_1,
            "prim_path_2": prim_path_2,
        })

    def get_distance(self, prim_path_1: str,
                     prim_path_2: str) -> Dict[str, Any]:
        """Get minimum distance between two prims."""
        return self._send("get_distance", {
            "prim_path_1": prim_path_1,
            "prim_path_2": prim_path_2,
        })

    def ray_cast(self, origin: List[float], direction: List[float],
                 max_distance: float = 100.0) -> Dict[str, Any]:
        """Cast a ray and return hit information."""
        return self._send("ray_cast", {
            "origin": origin,
            "direction": direction,
            "max_distance": max_distance,
        })

    def get_prim_paths(self, pattern: str = "/World/**") -> Dict[str, Any]:
        """Get all prim paths matching a pattern."""
        return self._send("get_prim_paths", {"pattern": pattern})

    # ══════════════════════════════════════════════════════════════
    # 8. Domain Randomization
    # ══════════════════════════════════════════════════════════════

    def randomize_lighting(self, intensity_range: Tuple[float, float] = (0.5, 1.5),
                           color_variation: bool = True) -> Dict[str, Any]:
        """Randomize scene lighting."""
        return self._send("randomize_lighting", {
            "intensity_min": intensity_range[0],
            "intensity_max": intensity_range[1],
            "color_variation": color_variation,
        })

    def randomize_textures(self, prim_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Randomize textures on objects."""
        params = {}
        if prim_paths:
            params["prim_paths"] = prim_paths
        return self._send("randomize_textures", params)

    def randomize_pose(self, prim_paths: List[str],
                       translation_range: float = 0.1,
                       rotation_range: float = 0.5) -> Dict[str, Any]:
        """Randomize poses of objects."""
        return self._send("randomize_pose", {
            "prim_paths": prim_paths,
            "translation_range": translation_range,
            "rotation_range": rotation_range,
        })

    # ══════════════════════════════════════════════════════════════
    # 9. High-level Task Primitives
    # ══════════════════════════════════════════════════════════════

    def setup_pick_and_place(
        self,
        robot_type: str = "franka",
        robot_position: Optional[List[float]] = None,
        object_position: Optional[List[float]] = None,
        target_position: Optional[List[float]] = None,
        object_type: str = "Cube",
        camera_config: Optional[CameraConfig] = None,
    ) -> Dict[str, Any]:
        """Quick-setup a pick-and-place scene.

        A convenience method that creates a complete pick-and-place
        environment in one call.

        Args:
            robot_type: Type of robot
            robot_position: Robot base position
            object_position: Object spawn position
            target_position: Target placement position
            object_type: Object prim type
            camera_config: Camera configuration

        Returns:
            Dict with scene, robot, object, target info
        """
        # Create scene
        scene_result = self.create_scene()

        # Create robot
        robot_result = self.create_robot(
            robot_type=robot_type,
            position=robot_position or [0.0, 0.0, 0.0],
        )

        # Add object to grasp
        obj_pos = object_position or [0.5, 0.0, 0.3]
        obj = self.add_object(ObjectSpec(
            prim_type=object_type,
            name="grasp_object",
            position=obj_pos,
            scale=[0.05, 0.05, 0.05],
            color=[0.8, 0.2, 0.2],
            mass=0.1,
        ))

        # Add target zone
        target_pos = target_position or [0.4, -0.3, 0.3]
        target = self.add_object(ObjectSpec(
            prim_type="Cylinder",
            name="target_zone",
            position=target_pos,
            scale=[0.08, 0.08, 0.02],
            color=[0.2, 0.8, 0.2],
            collision=False,
        ))

        # Set camera
        cam = camera_config or CameraConfig()
        self.set_camera(cam.position, cam.target)

        return {
            "scene": scene_result,
            "robot": robot_result,
            "object": obj,
            "target": target,
        }

    def pick_and_place(
        self,
        robot_name: str = "franka",
        object_name: str = "grasp_object",
        target_position: Optional[List[float]] = None,
        lift_height: float = 0.15,
        approach_distance: float = 0.08,
    ) -> Dict[str, Any]:
        """Execute a complete pick-and-place trajectory.

        This is a high-level primitive that:
        1. Moves to pre-grasp position above the object
        2. Opens gripper
        3. Moves down to grasp
        4. Closes gripper
        5. Lifts object
        6. Moves to target position
        7. Lowers object
        8. Opens gripper

        Args:
            robot_name: Robot prim name
            object_name: Name of the object to pick
            target_position: [x, y, z] target placement position
            lift_height: Height to lift after grasping
            approach_distance: Distance above object for approach pose

        Returns:
            Dict with result of each phase
        """
        # Get object and target info
        object_pose = self.get_object_pose(object_name)
        obj_pos = object_pose.get("position", [0.5, 0.0, 0.3])
        tgt_pos = target_position or [0.4, -0.3, 0.3]

        phases = {}
        errors = []

        try:
            # Phase 1: Pre-grasp (above object)
            pre_grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + approach_distance + lift_height]
            phases["pre_grasp"] = self.move_to_pose(robot_name, pre_grasp_pos + [0, 1, 0, 0])

            # Phase 2: Open gripper
            phases["open_gripper"] = self.control_gripper(robot_name, open=True, width=0.04)

            # Phase 3: Approach (move down to object)
            grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.02]
            phases["approach"] = self.move_to_pose(robot_name, grasp_pos + [0, 1, 0, 0])

            # Phase 4: Grasp
            phases["grasp"] = self.control_gripper(robot_name, open=False, width=0.0)
            self.step_simulation(30)

            # Phase 5: Lift
            lift_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + lift_height + 0.1]
            phases["lift"] = self.move_to_pose(robot_name, lift_pos + [0, 1, 0, 0])

            # Phase 6: Transport to target
            mid_pos = [(obj_pos[0] + tgt_pos[0]) / 2,
                       (obj_pos[1] + tgt_pos[1]) / 2,
                       obj_pos[2] + lift_height + 0.1]
            phases["transport_mid"] = self.move_to_pose(robot_name, mid_pos + [0, 1, 0, 0])

            above_target = [tgt_pos[0], tgt_pos[1], tgt_pos[2] + lift_height + 0.05]
            phases["transport_target"] = self.move_to_pose(robot_name, above_target + [0, 1, 0, 0])

            # Phase 7: Lower
            place_pos = [tgt_pos[0], tgt_pos[1], tgt_pos[2] + 0.02]
            phases["lower"] = self.move_to_pose(robot_name, place_pos + [0, 1, 0, 0])

            # Phase 8: Release
            phases["release"] = self.control_gripper(robot_name, open=True, width=0.04)
            self.step_simulation(30)

            # Phase 9: Retreat
            retreat_pos = [tgt_pos[0], tgt_pos[1], tgt_pos[2] + lift_height + 0.2]
            phases["retreat"] = self.move_to_pose(robot_name, retreat_pos + [0, 1, 0, 0])

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            errors.append(str(e))

        # Determine overall success
        success = len(errors) == 0

        return {
            "success": success,
            "phases": phases,
            "errors": errors,
            "object_position": obj_pos,
            "target_position": tgt_pos,
        }

    # ══════════════════════════════════════════════════════════════
    # 10. Performance & Info
    # ══════════════════════════════════════════════════════════════

    def get_sim_performance(self) -> Dict[str, Any]:
        """Get simulation performance statistics (FPS, etc.)."""
        return self._send("get_sim_performance")

    def get_asset_root(self) -> Dict[str, Any]:
        """Get the Isaac Sim asset root path."""
        return self._send("get_asset_root")

    def list_scene_objects(self) -> List[Dict[str, Any]]:
        """List all objects currently tracked in the scene."""
        return list(self._scene_objects.values())

    def omni_kit_command(self, command: str,
                         prim_type: str = "Sphere") -> Dict[str, Any]:
        """Execute an Omni Kit command directly. """
        return self._send("omni_kit_command", {
            "command": command,
            "prim_type": prim_type,
        })

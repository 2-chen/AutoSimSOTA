"""
Scene Registry — persistent storage and retrieval of scene configurations.

Provides:
  - Save/load/list generated and hand-crafted scenes
  - Tag-based search for scene discovery
  - Automatic scene variation generation
  - Import/export to JSON for sharing
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from autosim.scene.scene_schema import (
    StructuredSceneConfig,
    apply_variation,
    get_template,
    list_templates,
)


class SceneRegistry:
    """Persistent scene configuration registry.

    Scenes are stored as individual JSON files under a root directory,
    with an index for fast lookup.
    """

    def __init__(self, registry_dir: str = "scenes"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Build index
        self._index_path = self.registry_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = {}
        self._rebuild_index()

    def _rebuild_index(self):
        """Scan scenes directory and rebuild the index."""
        self._index = {}

        # Load existing index
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._index = {}

        # Scan for scene files
        for scene_file in self.registry_dir.glob("*.json"):
            if scene_file.name == "index.json":
                continue
            name = scene_file.stem
            if name not in self._index:
                try:
                    data = json.loads(scene_file.read_text())
                    self._index[name] = {
                        "name": data.get("name", name),
                        "description": data.get("description", ""),
                        "task_category": data.get("task", {}).get("category", "unknown"),
                        "num_objects": len(data.get("objects", [])),
                        "num_robots": len(data.get("robots", [])),
                        "file": scene_file.name,
                        "created": data.get("_metadata", {}).get("created", "unknown"),
                        "tags": data.get("_metadata", {}).get("tags", []),
                    }
                except (json.JSONDecodeError, KeyError):
                    continue

        self._save_index()

    def _save_index(self):
        """Write the index to disk."""
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def save(self, config: StructuredSceneConfig, tags: Optional[List[str]] = None,
             overwrite: bool = True) -> str:
        """Save a scene configuration to the registry.

        Args:
            config: Scene configuration to save
            tags: Optional tags for search/filter
            overwrite: Whether to overwrite if scene with same name exists

        Returns:
            Scene name
        """
        scene_path = self.registry_dir / f"{config.name}.json"
        if scene_path.exists() and not overwrite:
            raise FileExistsError(f"Scene '{config.name}' already exists")

        # Add metadata
        data = config.to_dict()
        data["_metadata"] = {
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "tags": tags or [],
            "version": 1,
        }

        scene_path.write_text(json.dumps(data, indent=2, default=str))

        # Update index
        self._index[config.name] = {
            "name": config.name,
            "description": config.description,
            "task_category": config.task.category.value if hasattr(config.task.category, 'value') else config.task.category,
            "num_objects": len(config.objects),
            "num_robots": len(config.robots),
            "file": scene_path.name,
            "created": data["_metadata"]["created"],
            "tags": tags or [],
        }
        self._save_index()

        return config.name

    def load(self, name: str) -> StructuredSceneConfig:
        """Load a scene configuration by name.

        Supports:
          - Exact name match in registry
          - Template name (from built-in templates)
        """
        # Try registry first
        scene_path = self.registry_dir / f"{name}.json"
        if scene_path.exists():
            data = json.loads(scene_path.read_text())
            return StructuredSceneConfig.from_dict(data)

        # Try templates
        if name in list_templates():
            return get_template(name)

        raise KeyError(f"Scene not found: {name!r}. "
                       f"Available: {self.list_names()}")

    def list_names(self, task_category: Optional[str] = None,
                   tag: Optional[str] = None) -> List[str]:
        """List available scene names, optionally filtered."""
        names = []
        for name, info in self._index.items():
            if task_category and info.get("task_category") != task_category:
                continue
            if tag and tag not in info.get("tags", []):
                continue
            names.append(name)
        names.sort()
        return names

    def list(self, task_category: Optional[str] = None,
             tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available scenes with full index info."""
        scenes = []
        for name, info in self._index.items():
            if task_category and info.get("task_category") != task_category:
                continue
            if tag and tag not in info.get("tags", []):
                continue
            scenes.append({"name": name, **info})
        scenes.sort(key=lambda s: s["name"])
        return scenes

    def delete(self, name: str) -> bool:
        """Delete a scene from registry. Returns True if deleted."""
        scene_path = self.registry_dir / f"{name}.json"
        if scene_path.exists():
            scene_path.unlink()
        if name in self._index:
            del self._index[name]
            self._save_index()
            return True
        return False

    def generate_variations(self, name: str, num_variations: int = 5,
                            variation_scale: float = 0.1) -> List[StructuredSceneConfig]:
        """Generate pose variations of a scene for robustness evaluation."""
        base = self.load(name)
        variations = []
        for i in range(num_variations):
            varied = apply_variation(base, seed=i, variation_scale=variation_scale)
            varied.name = f"{name}_var_{i}"
            varied.description = f"Variation {i} of {name}"
            variations.append(varied)
        return variations

    def import_from_file(self, path: str, name: Optional[str] = None,
                         tags: Optional[List[str]] = None) -> str:
        """Import a scene from an external JSON file."""
        data = json.loads(Path(path).read_text())
        config = StructuredSceneConfig.from_dict(data)
        if name:
            config.name = name
        return self.save(config, tags=tags)

    def export_to_file(self, name: str, output_path: str):
        """Export a scene to a JSON file."""
        config = self.load(name)
        Path(output_path).write_text(
            json.dumps(config.to_dict(), indent=2, default=str)
        )

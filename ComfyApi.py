import os
import sys
import json
import uuid
import random
import math
import shutil
import time
import asyncio
import inspect
import logging
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

import cv2
from PIL import Image

from src.model_names import resolve_placeholders

COMFY_API_BUILD = "2026-02-14-scene-direct-map"


class _LocalServer:
    """Minimal server stub for PromptExecutor."""

    def __init__(self):
        self.client_id = None
        self.last_node_id = None
        self.sockets_metadata = {}

    def send_sync(self, *args, **kwargs):
        return None


class ComfyApi:
    """
    Local ComfyUI executor for video workflows (t2v/i2v/v2v) without starting the server.
    """

    _nodes_initialized = False

    def __init__(
        self,
        base_dir: Optional[str] = None,
        workflows_dir: Optional[str] = None,
        preload_models: bool = True,
    ) -> None:
        logging.warning("ComfyApi build=%s", COMFY_API_BUILD)
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.workflows_dir = workflows_dir or os.path.abspath(
            os.path.join(self.base_dir, "..", "workflows")
        )
        self._ensure_comfyui_ready()

        self.server = _LocalServer()
        self.executor = self.execution.PromptExecutor(
            self.server,
            cache_type=self.execution.CacheType.CLASSIC,
            cache_args={"lru": 0, "ram": 0},
        )

        self._preloaded: List[Any] = []
        if preload_models:
            self._preload_models_from_workflows()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _ensure_comfyui_ready(self) -> None:
        # Some custom nodes import matplotlib; make it work in headless Spaces.
        os.environ.setdefault("MPLBACKEND", "Agg")

        if self.base_dir not in sys.path:
            sys.path.insert(0, self.base_dir)

        # HF Spaces/git doesn't preserve empty directories. ComfyUI expects these
        # folders to exist and will crash if they don't.
        for rel in ("custom_nodes", "input", "output", "temp", "user"):
            try:
                os.makedirs(os.path.join(self.base_dir, rel), exist_ok=True)
            except Exception:
                # Keep startup resilient; ComfyUI may still work if some dirs are unavailable.
                pass

        # Ensure "utils" resolves to ComfyUIVideo/utils (avoid site-packages `utils` shadowing).
        # Some packages unfortunately ship a top-level `utils.py`/`utils` module.
        if "utils" in sys.modules:
            mod = sys.modules["utils"]
            mod_file = os.path.abspath(getattr(mod, "__file__", "") or "")
            expected_prefix = os.path.abspath(os.path.join(self.base_dir, "utils")) + os.sep
            if (not hasattr(mod, "__path__")) or (not mod_file.startswith(expected_prefix)):
                for k in list(sys.modules.keys()):
                    if k == "utils" or k.startswith("utils."):
                        del sys.modules[k]

        # Import cli args first so we can force base_directory before folder_paths is loaded.
        from comfy import cli_args as comfy_cli_args

        comfy_cli_args.args.base_directory = self.base_dir
        # Optional CPU fallback via env var (set OMNI_FORCE_CPU=1 to force CPU mode).
        if os.getenv("OMNI_FORCE_CPU") == "1":
            comfy_cli_args.args.cpu = True

        # (Re)load folder_paths after setting base_directory.
        import importlib
        importlib.import_module("utils")
        import folder_paths

        importlib.reload(folder_paths)

        # Provide a minimal PromptServer.instance so custom nodes can register routes.
        try:
            import server as comfy_server

            class _DummyRoutes:
                def get(self, *args, **kwargs):
                    def _decorator(fn):
                        return fn
                    return _decorator

                def post(self, *args, **kwargs):
                    def _decorator(fn):
                        return fn
                    return _decorator

                def put(self, *args, **kwargs):
                    def _decorator(fn):
                        return fn
                    return _decorator

                def delete(self, *args, **kwargs):
                    def _decorator(fn):
                        return fn
                    return _decorator

            class _DummyPromptQueue:
                def put(self, *args, **kwargs):
                    return None

            class _DummyServerInstance:
                routes = _DummyRoutes()
                prompt_queue = _DummyPromptQueue()

                def add_on_prompt_handler(self, *args, **kwargs):
                    return None

                def send_sync(self, *args, **kwargs):
                    return None

                def __getattr__(self, _name):
                    def _noop(*args, **kwargs):
                        return None
                    return _noop

            if not hasattr(comfy_server.PromptServer, "instance"):
                comfy_server.PromptServer.instance = _DummyServerInstance()
            elif comfy_server.PromptServer.instance is None:
                comfy_server.PromptServer.instance = _DummyServerInstance()
        except Exception:
            # If server import fails, custom nodes that depend on it may be skipped.
            pass

        import nodes
        import execution

        if not ComfyApi._nodes_initialized:
            asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=True))
            ComfyApi._nodes_initialized = True

        self.nodes = nodes
        self.execution = execution
        self.folder_paths = folder_paths

    # ------------------------------------------------------------------
    # Model preload
    # ------------------------------------------------------------------
    def _preload_models_from_workflows(self) -> None:
        model_nodes = self._collect_model_nodes()
        if not model_nodes:
            return

        for class_type, inputs in model_nodes:
            cls = self.nodes.NODE_CLASS_MAPPINGS.get(class_type)
            if cls is None:
                logging.warning("Model node class not found: %s", class_type)
                continue

            try:
                obj = cls()
                fn_name = getattr(obj, "FUNCTION", None)
                if not fn_name:
                    logging.warning("Model node missing FUNCTION: %s", class_type)
                    continue
                fn = getattr(obj, fn_name)
                if inspect.iscoroutinefunction(fn):
                    result = asyncio.run(fn(**inputs))
                else:
                    result = fn(**inputs)
                self._preloaded.append(result)
            except Exception as exc:
                logging.warning("Model preload failed for %s: %s", class_type, exc)

    def _collect_model_nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        loader_types = {
            "UNETLoader",
            "UnetLoaderGGUF",
            "UnetLoaderGGUFAdvanced",
            "CLIPLoader",
            "CLIPLoaderGGUF",
            "DualCLIPLoader",
            "DualCLIPLoaderGGUF",
            "VAELoader",
        }
        seen = set()
        model_nodes: List[Tuple[str, Dict[str, Any]]] = []

        for filename in ("video_t2v_api.json", "video_i2v_api.json", "video_v2v_api.json"):
            path = os.path.join(self.workflows_dir, filename)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            # Resolve public placeholders to actual model filenames.
            workflow = resolve_placeholders(workflow)
            for node_id, node in workflow.items():
                class_type = node.get("class_type")
                if class_type not in loader_types:
                    continue
                inputs = node.get("inputs", {}) or {}
                if any(isinstance(v, list) for v in inputs.values()):
                    continue
                key = (class_type, tuple(sorted(inputs.items())))
                if key in seen:
                    continue
                seen.add(key)
                model_nodes.append((class_type, inputs))

        return model_nodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_workflow(self, filename: str) -> Dict[str, Any]:
        workflow_path = os.path.join(self.workflows_dir, filename)
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        # Resolve public placeholders to actual model filenames.
        workflow = resolve_placeholders(workflow)
        self._normalize_workflow_paths(workflow)
        return workflow

    def _run_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        prompt_id = uuid.uuid4().hex
        self._last_run_started = time.time()
        self._last_workflow = workflow
        # Avoid cache-only runs returning empty history outputs.
        self.executor.reset()
        execute_outputs = []
        missing_classes = set()
        for node_id, node in workflow.items():
            class_type = node.get("class_type")
            cls = self.nodes.NODE_CLASS_MAPPINGS.get(class_type)
            if cls is None:
                if class_type:
                    missing_classes.add(class_type)
                continue
            if getattr(cls, "OUTPUT_NODE", False):
                execute_outputs.append(node_id)

        if not execute_outputs:
            fallback_output_types = {
                "VHS_VideoCombine",
                "SaveImage",
                "SaveAnimatedWEBP",
                "SaveVideo",
            }
            for node_id, node in workflow.items():
                if node.get("class_type") in fallback_output_types:
                    execute_outputs.append(node_id)

        if not execute_outputs:
            # Last-resort: run all nodes so we don't silently return success with empty outputs.
            execute_outputs = list(workflow.keys())
            logging.warning(
                "No OUTPUT_NODE detected; fallback to execute all nodes. missing_classes=%s",
                sorted(missing_classes),
            )
        logging.warning(
            "Workflow execute outputs=%s (count=%d) missing_classes=%s",
            execute_outputs[:8],
            len(execute_outputs),
            sorted(missing_classes),
        )

        self._last_execute_outputs = list(execute_outputs)
        self.executor.execute(workflow, prompt_id, extra_data={}, execute_outputs=execute_outputs)
        if not self.executor.success:
            status_tail = getattr(self.executor, "status_messages", [])[-8:]
            logging.error("Workflow execution failed; status_tail=%s", status_tail)
            raise RuntimeError("ComfyUI workflow execution failed")
        outputs = self.executor.history_result.get("outputs", {})
        if not outputs:
            status_tail = getattr(self.executor, "status_messages", [])[-8:]
            logging.warning("Workflow finished with empty outputs; status_tail=%s", status_tail)
            for node_id in self._last_execute_outputs[:8]:
                try:
                    cached = self.executor.caches.outputs.get(node_id)
                except Exception:
                    cached = None
                if cached is None:
                    logging.warning("Output cache debug node=%s cache=none", node_id)
                    continue
                cache_outputs = getattr(cached, "outputs", None)
                cache_ui = getattr(cached, "ui", None)
                output_len = len(cache_outputs) if isinstance(cache_outputs, (list, tuple)) else None
                ui_keys = sorted(cache_ui.keys()) if isinstance(cache_ui, dict) else None
                logging.warning(
                    "Output cache debug node=%s outputs_type=%s outputs_len=%s ui_keys=%s",
                    node_id,
                    type(cache_outputs).__name__,
                    output_len,
                    ui_keys,
                )
                logging.warning(
                    "Output cache payload node=%s payload=%s",
                    node_id,
                    self._summarize_value(cache_outputs),
                )
            for node_id in self._last_execute_outputs[:2]:
                self._debug_upstream_for_output_node(node_id, max_depth=4, max_nodes=24)
        return outputs

    def _normalize_workflow_paths(self, workflow: Dict[str, Any]) -> None:
        def _fix(value):
            if isinstance(value, str) and "\\" in value:
                return value.replace("\\", "/")
            return value

        for node in workflow.values():
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                for k, v in list(inputs.items()):
                    if isinstance(v, list):
                        continue
                    inputs[k] = _fix(v)

    def _copy_to_input(self, local_path: str, prefix: str) -> Tuple[str, str]:
        dest_dir = self.folder_paths.get_input_directory()
        os.makedirs(dest_dir, exist_ok=True)
        ext = os.path.splitext(local_path)[1]
        dest_name = f"{prefix}_{uuid.uuid4().hex}{ext}"
        dest_path = os.path.join(dest_dir, dest_name)
        shutil.copy(local_path, dest_path)
        return dest_name, dest_path

    def _extract_first_frame(self, video_path: str) -> str:
        cap = cv2.VideoCapture(video_path)
        try:
            success, frame = cap.read()
            if not success or frame is None:
                raise RuntimeError(f"Failed to read first frame: {video_path}")
            frame_path = os.path.join(self.folder_paths.get_temp_directory(), f"first_frame_{uuid.uuid4().hex}.png")
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            cv2.imwrite(frame_path, frame)
            return frame_path
        finally:
            cap.release()

    def _get_video_dimensions(self, video_path: str) -> Tuple[int, int]:
        cap = cv2.VideoCapture(video_path)
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video size: {video_path}")
        return width, height

    def _round_to_multiple(self, value: int, multiple: int) -> int:
        return int(math.ceil(value / multiple) * multiple)

    def _summarize_cache_entry(self, value) -> str:
        if value is None:
            return "none"
        cache_ui = getattr(value, "ui", None)
        cache_outputs = getattr(value, "outputs", None)
        ui_keys = sorted(cache_ui.keys()) if isinstance(cache_ui, dict) else None
        outputs_len = len(cache_outputs) if isinstance(cache_outputs, (list, tuple)) else None
        return (
            f"type={type(value).__name__} ui_keys={ui_keys} "
            f"outputs_type={type(cache_outputs).__name__} outputs_len={outputs_len}"
        )

    def _summarize_value(self, value, depth: int = 0) -> str:
        if value is None:
            return "None"
        message = getattr(value, "message", None)
        if message is not None and type(value).__name__ == "ExecutionBlocker":
            return f"ExecutionBlocker(message={message!r})"
        if depth >= 2:
            return type(value).__name__
        if isinstance(value, (str, int, float, bool)):
            text = repr(value)
            if len(text) > 120:
                text = text[:117] + "..."
            return f"{type(value).__name__}({text})"
        if isinstance(value, dict):
            keys = list(value.keys())[:8]
            return f"dict(keys={keys})"
        if isinstance(value, (list, tuple)):
            sample = [self._summarize_value(v, depth + 1) for v in value[:3]]
            return f"{type(value).__name__}(len={len(value)}, sample={sample})"
        if hasattr(value, "shape"):
            return f"{type(value).__name__}(shape={getattr(value, 'shape', None)})"
        return type(value).__name__

    def _debug_upstream_for_output_node(self, output_node_id: str, max_depth: int = 4, max_nodes: int = 24) -> None:
        workflow = getattr(self, "_last_workflow", None)
        if not isinstance(workflow, dict):
            return
        if output_node_id not in workflow:
            return
        queue = deque([(output_node_id, 0, "output")])
        seen = set()
        visited = 0
        while queue and visited < max_nodes:
            node_id, depth, via = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            visited += 1

            node = workflow.get(node_id, {})
            class_type = node.get("class_type")
            inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
            try:
                cached = self.executor.caches.outputs.get(node_id)
            except Exception:
                cached = None
            cache_outputs = getattr(cached, "outputs", None) if cached is not None else None
            logging.warning(
                "Output upstream node=%s depth=%d via=%s class=%s cache=%s payload=%s",
                node_id,
                depth,
                via,
                class_type,
                self._summarize_cache_entry(cached),
                self._summarize_value(cache_outputs),
            )

            if depth >= max_depth or not isinstance(inputs, dict):
                continue
            for input_name, input_value in inputs.items():
                if (
                    isinstance(input_value, list)
                    and len(input_value) >= 2
                    and isinstance(input_value[0], str)
                ):
                    parent_id = input_value[0]
                    if parent_id not in seen:
                        queue.append((parent_id, depth + 1, input_name))
        if visited >= max_nodes:
            logging.warning(
                "Output upstream traversal truncated for node=%s at max_nodes=%d",
                output_node_id,
                max_nodes,
            )

    def _resolve_scene_prompts(self, prompt: str, num_scene: Optional[int]) -> Tuple[List[str], int]:
        prompts = [p.strip() for p in (prompt or "").split("¥")] if prompt else [""]
        while len(prompts) > 1 and prompts[-1] == "":
            prompts.pop()
        if not prompts:
            prompts = [""]

        auto_scene_count = max(1, min(len(prompts), 4))
        if num_scene is None:
            scene_count = auto_scene_count
        else:
            try:
                requested = int(num_scene)
            except (TypeError, ValueError):
                requested = auto_scene_count
            requested = max(1, min(requested, 4))
            # If caller forces multi-scene but the prompt only has one segment,
            # default back to single-scene to avoid empty branch outputs.
            if requested > 1 and auto_scene_count == 1:
                logging.warning(
                    "num_scene=%s requested with single-segment prompt; fallback to 1 scene",
                    requested,
                )
                scene_count = 1
            else:
                scene_count = requested

        while len(prompts) < scene_count:
            prompts.append(prompts[-1])
        return prompts, scene_count

    def _save_output_video(self, outputs: Dict[str, Any], output_path: str) -> str:
        video_exts = {".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp"}

        def _extract_video_entry(node_outputs: Optional[Dict[str, Any]]):
            if not isinstance(node_outputs, dict):
                return None
            for key in ("gifs", "videos", "video"):
                if key in node_outputs and node_outputs[key]:
                    return node_outputs[key][0]
            return None

        def _normalize_candidate_path(path_value):
            if path_value is None:
                return None
            try:
                path = os.fspath(path_value)
            except Exception:
                return None
            if not isinstance(path, str):
                return None
            path = path.strip()
            if not path:
                return None

            if os.path.isabs(path):
                return path if os.path.exists(path) else None

            candidate_bases = (
                self.folder_paths.get_output_directory(),
                self.folder_paths.get_temp_directory(),
                self.base_dir,
            )
            for base_dir in candidate_bases:
                if not base_dir:
                    continue
                candidate = os.path.abspath(os.path.join(base_dir, path))
                if os.path.exists(candidate):
                    return candidate
            return None

        def _extract_path_from_cached(value):
            # VideoCombine usually returns ((save_output, [file1, file2, ...]),)
            # in the node result cache. Walk recursively and pick the newest video.
            found = []

            def _walk(obj):
                normalized = _normalize_candidate_path(obj)
                if normalized:
                    lower = normalized.lower()
                    if os.path.splitext(lower)[1] in video_exts:
                        found.append(normalized)
                    return
                if hasattr(obj, "outputs") and hasattr(obj, "ui"):
                    _walk(getattr(obj, "ui", None))
                    _walk(getattr(obj, "outputs", None))
                    return
                if isinstance(obj, dict):
                    filename = obj.get("filename")
                    if filename:
                        subfolder = obj.get("subfolder") or ""
                        folder_type = str(obj.get("type") or "").lower()
                        base_dirs = []
                        if folder_type == "temp":
                            base_dirs.append(self.folder_paths.get_temp_directory())
                        else:
                            base_dirs.append(self.folder_paths.get_output_directory())
                        for base_dir in base_dirs:
                            if not base_dir:
                                continue
                            candidate = os.path.join(base_dir, subfolder, filename) if subfolder else os.path.join(base_dir, filename)
                            candidate = _normalize_candidate_path(candidate)
                            if candidate and os.path.splitext(candidate.lower())[1] in video_exts:
                                found.append(candidate)
                    for v in obj.values():
                        _walk(v)
                    return
                if isinstance(obj, (list, tuple)):
                    for v in obj:
                        _walk(v)

            _walk(value)
            if not found:
                return None
            found = sorted(set(found), key=lambda p: os.path.getmtime(p))
            return found[-1]

        mode = os.path.basename(output_path).split("_", 1)[0].lower()
        prefix_map = {
            "t2v": ("omni_t2v",),
            "i2v": ("omni_i2v",),
            "v2v": ("omni_v2v",),
        }
        expected_prefixes = prefix_map.get(mode, ("omni_",))

        if not outputs:
            # Try extracting file path directly from node cache when UI outputs are empty.
            for node_id in getattr(self, "_last_execute_outputs", []):
                try:
                    cached = self.executor.caches.outputs.get(node_id)
                except Exception:
                    cached = None
                logging.warning("Output recovery inspect node=%s cache=%s", node_id, self._summarize_cache_entry(cached))
                path = _extract_path_from_cached(cached)
                if path:
                    logging.warning("Using cached node video output from node=%s path=%s", node_id, path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy(path, output_path)
                    return output_path
                if cached is not None:
                    logging.warning(
                        "Output recovery payload node=%s payload=%s",
                        node_id,
                        self._summarize_value(getattr(cached, "outputs", None)),
                    )

            fallback = self._find_latest_output_video(
                require_recent=True,
                expected_prefixes=expected_prefixes,
            )
            if fallback:
                logging.warning("Using fallback video output: %s", fallback)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy(fallback, output_path)
                return output_path
            logging.warning(
                "No fallback video found for prefixes=%s require_recent=%s",
                expected_prefixes,
                True,
            )
            for node_id in getattr(self, "_last_execute_outputs", [])[:2]:
                self._debug_upstream_for_output_node(node_id, max_depth=4, max_nodes=24)

        for _, node_outputs in outputs.items():
            video_entry = _extract_video_entry(node_outputs)
            if not video_entry:
                continue
            filename = video_entry["filename"]
            subfolder = video_entry.get("subfolder", "")
            base_dir = self.folder_paths.get_output_directory()
            src_path = os.path.join(base_dir, subfolder, filename) if subfolder else os.path.join(base_dir, filename)
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Output video not found: {src_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(src_path, output_path)
            return output_path

        raise KeyError(f"No video output found, outputs keys={list(outputs.keys())}")

    def _find_latest_output_video(
        self,
        require_recent: bool = True,
        expected_prefixes: Optional[Tuple[str, ...]] = None,
    ) -> Optional[str]:
        candidate_dirs = [
            self.folder_paths.get_output_directory(),
            self.folder_paths.get_temp_directory(),
        ]
        exts = {".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp"}
        newest_path = None
        newest_mtime = 0.0
        since = getattr(self, "_last_run_started", 0) - 5
        prefixes = tuple(p.lower() for p in (expected_prefixes or ()))
        seen_dirs = set()
        for base_dir in candidate_dirs:
            if not base_dir:
                continue
            base_dir = os.path.abspath(base_dir)
            if base_dir in seen_dirs or not os.path.isdir(base_dir):
                continue
            seen_dirs.add(base_dir)
            for root, _, files in os.walk(base_dir):
                for name in files:
                    if os.path.splitext(name)[1].lower() not in exts:
                        continue
                    if prefixes and not any(name.lower().startswith(p) for p in prefixes):
                        continue
                    path = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(path)
                    except OSError:
                        continue
                    if require_recent and mtime < since:
                        continue
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest_path = path
        return newest_path

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def video_t2v(
        self,
        prompt: str,
        vid_resolution: int = 448,
        num_frames: int = 81,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_scene: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> str:
        seed = random.randint(0, 9999999999)
        prompts, num_scene = self._resolve_scene_prompts(prompt, num_scene)

        vid_resolution = int(vid_resolution or 448)
        width = int(width) if width else 0
        height = int(height) if height else 0
        workflow = self._load_workflow("video_t2v_api.json")
        actual_length = int(num_frames or 81)
        workflow["607"]["inputs"]["desired_width"] = vid_resolution
        workflow["607"]["inputs"]["desired_height"] = vid_resolution
        workflow["608"]["inputs"]["value"] = actual_length
        workflow["605"]["inputs"]["value"] = 10
        workflow["1252:1250"]["inputs"]["seed"] = seed
        workflow["1252:1244"]["inputs"]["text"] = prompts[0]
        if "1262:308" in workflow:
            workflow["1262:308"]["inputs"]["seed"] = seed
        if "1262:1257" in workflow:
            workflow["1262:1257"]["inputs"]["text"] = prompts[1] if num_scene >= 2 else prompts[0]
        if "1312:1267" in workflow:
            workflow["1312:1267"]["inputs"]["text"] = prompts[2] if num_scene >= 3 else prompts[min(1, len(prompts) - 1)]
        if "1335:1267" in workflow:
            workflow["1335:1267"]["inputs"]["text"] = prompts[3] if num_scene >= 4 else prompts[min(2, len(prompts) - 1)]
        if "1312:309" in workflow:
            workflow["1312:309"]["inputs"]["seed"] = seed
        if "1335:309" in workflow:
            workflow["1335:309"]["inputs"]["seed"] = seed

        if "1318" in workflow and "index" in workflow["1318"]["inputs"]:
            workflow["1318"]["inputs"]["index"] = 1 if num_scene > 1 else 0
        if "1321" in workflow and "index" in workflow["1321"]["inputs"]:
            workflow["1321"]["inputs"]["index"] = 1 if num_scene > 2 else 0
        if "1338" in workflow and "index" in workflow["1338"]["inputs"]:
            workflow["1338"]["inputs"]["index"] = 1 if num_scene > 3 else 0
        if "1322" in workflow and "index" in workflow["1322"]["inputs"]:
            workflow["1322"]["inputs"]["index"] = num_scene - 1
        selector_inputs = workflow.get("1322", {}).get("inputs", {})
        selector_index = selector_inputs.get("index")
        selector_key = f"value{selector_index}" if isinstance(selector_index, int) else None
        selector_link = selector_inputs.get(selector_key) if selector_key else None
        logging.warning(
            "T2V scene routing num_scene=%s gate_index={1318:%s,1321:%s,1338:%s} selector={1322:%s %s:%s}",
            num_scene,
            workflow.get("1318", {}).get("inputs", {}).get("index"),
            workflow.get("1321", {}).get("inputs", {}).get("index"),
            workflow.get("1338", {}).get("inputs", {}).get("index"),
            selector_index,
            selector_key,
            selector_link,
        )
        t2v_scene_output_map = {
            1: "1252:1249",
            2: "1262:1253",
            3: "1312:1263",
            4: "1335:1263",
        }
        target_node_id = t2v_scene_output_map.get(num_scene)
        if target_node_id and "1039" in workflow and target_node_id in workflow:
            workflow["1039"]["inputs"]["images"] = [target_node_id, 0]
            logging.warning(
                "T2V scene direct map enabled: num_scene=%s 1039.images -> %s",
                num_scene,
                [target_node_id, 0],
            )
        if width > 0 and height > 0:
            workflow["97"]["inputs"]["width"] = width
            workflow["97"]["inputs"]["height"] = height

        outputs = self._run_workflow(workflow)
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "output_video_t2v.mp4")
        return self._save_output_video(outputs, output_path)

    def video_i2v(
        self,
        prompt: str,
        img_path: str,
        vid_resolution: int = 448,
        num_frames: int = 81,
        num_scene: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> str:
        seed = random.randint(0, 9999999999)
        prompts, num_scene = self._resolve_scene_prompts(prompt, num_scene)

        uploaded_filename, copied_path = self._copy_to_input(img_path, prefix="i2v_img")
        try:
            vid_resolution = int(vid_resolution or 448)
            workflow = self._load_workflow("video_i2v_api.json")
            actual_length = int(num_frames or 81)
            workflow["607"]["inputs"]["desired_width"] = vid_resolution
            workflow["607"]["inputs"]["desired_height"] = vid_resolution
            workflow["608"]["inputs"]["value"] = actual_length
            workflow["605"]["inputs"]["value"] = 10
            workflow["97"]["inputs"]["image"] = uploaded_filename
            workflow["1252:1250"]["inputs"]["seed"] = seed
            if "1262:308" in workflow:
                workflow["1262:308"]["inputs"]["seed"] = seed
            workflow["1252:1244"]["inputs"]["text"] = prompts[0]
            if "1262:1257" in workflow:
                workflow["1262:1257"]["inputs"]["text"] = prompts[1] if num_scene >= 2 else prompts[0]
            if "1301:1257" in workflow:
                workflow["1301:1257"]["inputs"]["text"] = prompts[2] if num_scene >= 3 else prompts[min(1, len(prompts) - 1)]
            if "1313:1257" in workflow:
                workflow["1313:1257"]["inputs"]["text"] = prompts[3] if num_scene >= 4 else prompts[min(2, len(prompts) - 1)]
            if "1301:308" in workflow:
                workflow["1301:308"]["inputs"]["seed"] = seed
            if "1313:308" in workflow:
                workflow["1313:308"]["inputs"]["seed"] = seed

            if "1281" in workflow and "index" in workflow["1281"]["inputs"]:
                workflow["1281"]["inputs"]["index"] = 1 if num_scene > 1 else 0
            if "1288" in workflow and "index" in workflow["1288"]["inputs"]:
                workflow["1288"]["inputs"]["index"] = 1 if num_scene > 2 else 0
            if "1300" in workflow and "index" in workflow["1300"]["inputs"]:
                workflow["1300"]["inputs"]["index"] = 1 if num_scene > 3 else 0
            if "1287" in workflow and "index" in workflow["1287"]["inputs"]:
                workflow["1287"]["inputs"]["index"] = num_scene - 1
            selector_inputs = workflow.get("1287", {}).get("inputs", {})
            selector_index = selector_inputs.get("index")
            selector_key = f"value{selector_index}" if isinstance(selector_index, int) else None
            selector_link = selector_inputs.get(selector_key) if selector_key else None
            logging.warning(
                "I2V scene routing num_scene=%s gate_index={1281:%s,1288:%s,1300:%s} selector={1287:%s %s:%s}",
                num_scene,
                workflow.get("1281", {}).get("inputs", {}).get("index"),
                workflow.get("1288", {}).get("inputs", {}).get("index"),
                workflow.get("1300", {}).get("inputs", {}).get("index"),
                selector_index,
                selector_key,
                selector_link,
            )
            i2v_scene_output_map = {
                1: "1252:1249",
                2: "1262:1253",
                3: "1301:1253",
                4: "1313:1253",
            }
            target_node_id = i2v_scene_output_map.get(num_scene)
            if target_node_id and "1039" in workflow and target_node_id in workflow:
                workflow["1039"]["inputs"]["images"] = [target_node_id, 0]
                logging.warning(
                    "I2V scene direct map enabled: num_scene=%s 1039.images -> %s",
                    num_scene,
                    [target_node_id, 0],
                )

            outputs = self._run_workflow(workflow)
        finally:
            if copied_path and os.path.exists(copied_path):
                os.remove(copied_path)

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "output_video_i2v.mp4")
        return self._save_output_video(outputs, output_path)

    def video_v2v(
        self,
        prompt: str,
        video_path: str,
        num_frames: int = 81,
        vid_resolution: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> str:
        seed = random.randint(0, 9999999999)
        actual_frames = max(1, min(int(num_frames), 129))
        first_frame_path = self._extract_first_frame(video_path)
        orig_width, orig_height = self._get_video_dimensions(video_path)
        width = max(16, self._round_to_multiple(orig_width, 16))
        height = max(16, self._round_to_multiple(orig_height, 16))

        if vid_resolution:
            target_res = self._round_to_multiple(int(float(vid_resolution)), 16)
            width = target_res
            height = max(16, self._round_to_multiple(int(target_res * (orig_height / orig_width)), 16))

        first_frame_name, copied_frame_path = self._copy_to_input(first_frame_path, prefix="first_frame")
        video_filename, copied_video_path = self._copy_to_input(video_path, prefix="video_v2v")

        try:
            workflow = self._load_workflow("video_v2v_api.json")
            workflow["369"]["inputs"]["video"] = video_filename
            workflow["162"]["inputs"]["value"] = width
            workflow["163"]["inputs"]["value"] = height
            workflow["164"]["inputs"]["value"] = actual_frames
            workflow["165"]["inputs"]["value"] = 10
            workflow["284"]["inputs"]["image"] = first_frame_name
            workflow["138"]["inputs"]["text"] = prompt
            workflow["200"]["inputs"]["value"] = seed

            outputs = self._run_workflow(workflow)
        finally:
            for p in (copied_video_path, copied_frame_path, first_frame_path):
                if p and os.path.exists(p):
                    os.remove(p)

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "output_video_v2v.mp4")
        return self._save_output_video(outputs, output_path)

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
from typing import Dict, Any, List, Tuple, Optional

import cv2
from PIL import Image

from src.model_names import resolve_placeholders


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
        else:
            logging.info(
                "Workflow execute outputs=%s missing_classes=%s",
                execute_outputs[:8],
                sorted(missing_classes),
            )

        self.executor.execute(workflow, prompt_id, extra_data={}, execute_outputs=execute_outputs)
        if not self.executor.success:
            raise RuntimeError("ComfyUI workflow execution failed")
        outputs = self.executor.history_result.get("outputs", {})
        if not outputs:
            events = [event for event, _ in getattr(self.executor, "status_messages", [])]
            logging.warning("Workflow finished with empty outputs; status_events=%s", events[-8:])
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

    def _save_output_video(self, outputs: Dict[str, Any], output_path: str) -> str:
        def _extract_video_entry(node_outputs: Optional[Dict[str, Any]]):
            if not isinstance(node_outputs, dict):
                return None
            for key in ("gifs", "videos", "video"):
                if key in node_outputs and node_outputs[key]:
                    return node_outputs[key][0]
            return None

        if not outputs:
            fallback = self._find_latest_output_video(require_recent=True)
            if not fallback:
                # A cache-only run may not create a fresh output in this invocation.
                fallback = self._find_latest_output_video(require_recent=False)
            if fallback:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy(fallback, output_path)
                return output_path

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

    def _find_latest_output_video(self, require_recent: bool = True) -> Optional[str]:
        base_dir = self.folder_paths.get_output_directory()
        exts = {".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp"}
        newest_path = None
        newest_mtime = 0.0
        since = getattr(self, "_last_run_started", 0) - 5
        for root, _, files in os.walk(base_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() not in exts:
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
        num_scene: int = 1,
        output_path: Optional[str] = None,
    ) -> str:
        seed = random.randint(0, 9999999999)
        try:
            num_scene = int(num_scene or 1)
        except (TypeError, ValueError):
            num_scene = 1
        num_scene = max(1, min(num_scene, 4))
        prompts = [p.strip() for p in (prompt or "").split("¥")] if prompt else [""]
        while len(prompts) < num_scene:
            prompts.append(prompts[-1] if prompts else "")

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
        num_scene: int = 1,
        output_path: Optional[str] = None,
    ) -> str:
        seed = random.randint(0, 9999999999)
        try:
            num_scene = int(num_scene or 1)
        except (TypeError, ValueError):
            num_scene = 1
        num_scene = max(1, min(num_scene, 4))
        prompts = [p.strip() for p in (prompt or "").split("¥")] if prompt else [""]
        while len(prompts) < num_scene:
            prompts.append(prompts[-1] if prompts else "")

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

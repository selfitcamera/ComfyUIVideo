import os
import subprocess
from pathlib import Path

"""
This file lives under ComfyUIVideo/custom_nodes/.

ComfyUI's custom node loader imports every .py file in custom_nodes at startup.
This script originally ran `pip install -r ...` on import (side effect), which makes
*every* launch slow/noisy and can be dangerous (it used os.getcwd()).

Keep it as an *optional* utility: run it explicitly as a script when you actually
want to install requirements.
"""

def main() -> int:
    base_dir = Path(__file__).resolve().parent
    for node_path in base_dir.iterdir():
        if not node_path.is_dir():
            continue
        req_file = node_path / "requirements.txt"
        if req_file.exists():
            print(f"Installing dependencies for {node_path.name}...")
            subprocess.run(["python3", "-m", "pip", "install", "-r", str(req_file)], check=False)
        else:
            print(f"No requirements.txt found for {node_path.name}, skipping...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

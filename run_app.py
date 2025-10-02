import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "app.py"
    if not app_path.exists():
        print("app.py not found next to launcher.")
        return 1

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"]
    try:
        return subprocess.call(cmd, env=env)
    except Exception as exc:
        print(f"Failed to start Streamlit: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

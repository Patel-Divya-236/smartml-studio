"""Run the FastAPI backend and the Vite dev server together.

Two processes have to be alive for the React app to work: the API on port 8000 and Vite
on 5173, which proxies /api (including the training WebSocket) to it. Starting them by
hand in two terminals works equally well; this exists so a demo is one command.

The interpreter matters. This project's dependencies live in `.venv`, and the system
Python usually has only some of them — enough for the app to import and then fail on the
first upload or the first training run. So the venv is located explicitly rather than
inheriting whichever Python happened to launch this script.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
FRONTEND = ROOT / "frontend"

# Imported by the backend at startup or on the first request. A missing one here is far
# easier to act on than a traceback thrown mid-upload.
REQUIRED_MODULES = (
    "fastapi",
    "uvicorn",
    "multipart",     # python-multipart, needed for file upload
    "pandas",
    "sklearn",
    "xgboost",
    "shap",
)


def find_python() -> Path:
    """Return the interpreter to run the backend with.

    Prefers the project's own virtual environment. Falls back to the current interpreter
    only when no venv exists, which is the case worth warning about.
    """
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",   # Windows
        ROOT / ".venv" / "bin" / "python",           # POSIX
        ROOT / "venv" / "Scripts" / "python.exe",
        ROOT / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    print("No .venv found — falling back to the interpreter running this script.")
    return Path(sys.executable)


def check_dependencies(python: Path) -> list[str]:
    """Return the names of required modules that `python` cannot import."""
    probe = (
        "import importlib, sys\n"
        f"missing = [m for m in {REQUIRED_MODULES!r} "
        "if importlib.util.find_spec(m) is None]\n"
        "print(','.join(missing))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    return [name for name in result.stdout.strip().split(",") if name]


def main() -> int:
    """Start both servers and shut them down together."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--web-port", type=int, default=5173)
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks.")
    args = parser.parse_args()

    if not (FRONTEND / "node_modules").exists():
        print("Frontend dependencies are missing. Run:  cd frontend && npm install")
        return 1

    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        print("npm was not found on PATH.")
        return 1

    python = find_python()

    if not args.skip_checks:
        missing = check_dependencies(python)
        if missing:
            print(f"\n  {python} is missing: {', '.join(missing)}\n")
            print("  Install them into that interpreter:\n")
            print(f'    "{python}" -m pip install -r requirements.txt\n')
            return 1

    processes = [
        subprocess.Popen(
            [str(python), "-m", "uvicorn", "backend.main:app",
             "--port", str(args.api_port), "--reload"],
            cwd=ROOT,
        ),
        subprocess.Popen(
            [npm, "run", "dev", "--", "--port", str(args.web_port)],
            cwd=FRONTEND,
            shell=os.name == "nt",
            # Vite reads this to point its /api proxy at the API we just started, so
            # --api-port cannot leave the proxy aimed at a dead port.
            env={**os.environ, "VITE_API_TARGET": f"http://127.0.0.1:{args.api_port}"},
        ),
    ]

    print(f"\n  Python  {python}")
    print(f"  API     http://127.0.0.1:{args.api_port}/docs")
    print(f"  App     http://localhost:{args.web_port}\n")

    try:
        processes[1].wait()
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

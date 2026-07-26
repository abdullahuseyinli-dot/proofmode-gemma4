"""Windows-friendly launcher and floating control for ProofMode.

The module intentionally depends only on the Python standard library so it can
start the project's virtual environment and services before the application is
available.  Importing it has no side effects; subprocesses are created only by
``main()``/``run_launcher()``.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence


HOST = "127.0.0.1"
LLAMA_PORT = 8080
STREAMLIT_PORT = 8501
LLAMA_HEALTH_URL = f"http://{HOST}:{LLAMA_PORT}/health"
APP_HEALTH_URL = f"http://{HOST}:{STREAMLIT_PORT}/_stcore/health"
APP_URL = f"http://{HOST}:{STREAMLIT_PORT}"


def project_root() -> Path:
    """Return the directory containing this launcher."""

    return Path(__file__).resolve().parent


def find_gemma_home(
    root: Path | None = None, env: Mapping[str, str] | None = None
) -> Path:
    """Locate the Gemma runtime from an override or the project's sibling.

    ``PROOFMODE_GEMMA_HOME`` is useful when the model is on another drive.  The
    default layout is ``ProofMode`` and ``gemma4`` beside one another.
    """

    environment = os.environ if env is None else env
    configured = environment.get("PROOFMODE_GEMMA_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    base = project_root() if root is None else Path(root).resolve()
    return base.parent / "gemma4"


def find_project_python(root: Path, env: Mapping[str, str] | None = None) -> Path:
    """Prefer the configured interpreter, then the project's virtualenv."""

    environment = os.environ if env is None else env
    configured = environment.get("PROOFMODE_PYTHON", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    scripts = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    candidate = Path(root) / ".venv" / scripts / executable
    if candidate.is_file():
        return candidate.resolve()
    return Path(sys.executable).resolve()


def find_streamlit_entrypoint(
    root: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Find the Streamlit application, allowing an explicit override."""

    environment = os.environ if env is None else env
    configured = environment.get("PROOFMODE_APP", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = Path(root) / candidate
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Configured Streamlit app does not exist: {candidate}")

    for name in ("app.py", "streamlit_app.py", "Home.py"):
        candidate = Path(root) / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No Streamlit entrypoint found in {root}; expected app.py or set PROOFMODE_APP."
    )


def _first_file(directory: Path, preferred: str, pattern: str) -> Path:
    exact = directory / preferred
    if exact.is_file():
        return exact.resolve()
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"Missing {preferred} under {directory}")


def build_llama_command(gemma_home: Path, context_size: int = 8192) -> list[str]:
    """Build and validate the local multimodal Gemma server command."""

    home = Path(gemma_home).resolve()
    runtime = home / "runtime"
    models = home / "models"
    executable_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    executable = _first_file(runtime, executable_name, "llama-server*")
    model = _first_file(
        models, "gemma-4-E4B_q4_0-it.gguf", "*gemma*E4B*q4*it*.gguf"
    )
    projector = _first_file(
        models, "gemma-4-E4B-it-mmproj.gguf", "*gemma*E4B*mmproj*.gguf"
    )
    return [
        str(executable),
        "--model",
        str(model),
        "--mmproj",
        str(projector),
        "--gpu-layers",
        "all",
        "--ctx-size",
        str(context_size),
        "--reasoning",
        "off",
        "--host",
        HOST,
        "--port",
        str(LLAMA_PORT),
        "--jinja",
    ]


def build_streamlit_command(
    root: Path, env: Mapping[str, str] | None = None
) -> list[str]:
    """Build the command that serves ProofMode only on loopback."""

    python = find_project_python(Path(root), env)
    app = find_streamlit_entrypoint(Path(root), env)
    return [
        str(python),
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        HOST,
        "--server.port",
        str(STREAMLIT_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    """Return whether a TCP listener is accepting connections."""

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_ready(url: str, timeout: float = 0.75) -> bool:
    """Return whether a service health URL responds successfully."""

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ProofModeLauncher/1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (OSError, ValueError, urllib.error.URLError):
        return False


def wait_until_ready(
    url: str,
    timeout: float,
    process: subprocess.Popen[bytes] | None = None,
    check: Callable[[str], bool] = http_ready,
    interval: float = 0.25,
) -> None:
    """Wait for a health endpoint, failing early when its process exits."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check(url):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Service exited before becoming ready (exit code {process.returncode}): {url}"
            )
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for {url}")


def _hidden_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )


def stop_process(process: subprocess.Popen[bytes], timeout: float = 5.0) -> None:
    """Stop one owned process, including its Windows child process tree."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "nt":
        flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return


@dataclass
class OwnedProcesses:
    """Tracks only processes launched by this instance."""

    children: list[subprocess.Popen[bytes]] = field(default_factory=list)

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        hidden: bool = True,
    ) -> subprocess.Popen[bytes]:
        creationflags = _hidden_creation_flags() if hidden else 0
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=None if env is None else dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self.children.append(process)
        return process

    def stop_all(self) -> None:
        while self.children:
            process = self.children.pop()
            stop_process(process)


def ensure_service(
    *,
    name: str,
    host: str,
    port: int,
    health_url: str,
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    processes: OwnedProcesses,
    startup_timeout: float,
) -> subprocess.Popen[bytes] | None:
    """Reuse a healthy service or start and track exactly one new instance."""

    if is_port_open(host, port):
        try:
            wait_until_ready(health_url, timeout=12.0)
            return None
        except TimeoutError as exc:
            raise RuntimeError(
                f"Port {host}:{port} is occupied, but it is not a healthy {name} service."
            ) from exc

    process = processes.start(command, cwd=cwd, env=env, hidden=True)
    wait_until_ready(health_url, timeout=startup_timeout, process=process)
    return process


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # Do not use os.kill(pid, 0) on Windows: unlike POSIX, unsupported
        # signals can be implemented with TerminateProcess.  A read-only
        # process handle gives us an existence/liveness check with no signal.
        try:
            import ctypes

            synchronize = 0x00100000
            query_limited_information = 0x00001000
            still_active = 259
            handle = ctypes.windll.kernel32.OpenProcess(
                synchronize | query_limited_information, False, pid
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                ):
                    return False
                return exit_code.value == still_active
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except (AttributeError, OSError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


@dataclass
class LauncherLock:
    """Small atomic PID lock preventing simultaneous startup races."""

    path: Path
    acquired: bool = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                try:
                    owner = int(self.path.read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    owner = -1
                if _pid_is_running(owner):
                    return False
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(str(os.getpid()))
            self.acquired = True
            return True
        return False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            owner = int(self.path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            owner = -1
        if owner == os.getpid():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self.acquired = False


def find_app_browser(env: Mapping[str, str] | None = None) -> Path | None:
    """Find Edge/Chrome for a controllable app-mode window."""

    environment = os.environ if env is None else env
    configured = environment.get("PROOFMODE_APP_BROWSER", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        return candidate.resolve() if candidate.is_file() else None

    roots = [
        environment.get("PROGRAMFILES(X86)", ""),
        environment.get("PROGRAMFILES", ""),
        environment.get("LOCALAPPDATA", ""),
    ]
    relative_paths = (
        Path("Microsoft/Edge/Application/msedge.exe"),
        Path("Google/Chrome/Application/chrome.exe"),
    )
    for base in roots:
        if not base:
            continue
        for relative in relative_paths:
            candidate = Path(base) / relative
            if candidate.is_file():
                return candidate.resolve()
    for name in ("msedge.exe", "chrome.exe", "msedge", "google-chrome"):
        located = shutil.which(name)
        if located:
            return Path(located).resolve()
    return None


@dataclass
class AppWindowController:
    """Show/hide ProofMode in a dedicated browser-app process."""

    url: str
    browser_path: Path | None
    profile_dir: Path
    process: subprocess.Popen[bytes] | None = None
    fallback_opened: bool = False

    def is_open(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def show(self) -> bool:
        if self.is_open():
            return True
        if self.browser_path is None:
            self.fallback_opened = bool(webbrowser.open(self.url, new=1))
            return self.fallback_opened

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        flags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [
                str(self.browser_path),
                f"--app={self.url}",
                f"--user-data-dir={self.profile_dir}",
                "--no-first-run",
                "--disable-default-apps",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        return True

    def hide(self) -> bool:
        if not self.is_open():
            return False
        assert self.process is not None
        stop_process(self.process, timeout=2.0)
        self.process = None
        return True

    def toggle(self) -> bool:
        if self.is_open():
            self.hide()
            return False
        self.show()
        return True


def run_bubble(controller: AppWindowController, on_quit: Callable[[], None]) -> None:
    """Run a draggable, always-on-top ProofMode bubble."""

    import tkinter as tk

    window = tk.Tk()
    window.title("ProofMode")
    window.overrideredirect(True)
    window.attributes("-topmost", True)
    window.geometry("64x64+24+180")
    transparent = "#010203"
    window.configure(bg=transparent)
    if os.name == "nt":
        try:
            window.wm_attributes("-transparentcolor", transparent)
        except tk.TclError:
            pass

    canvas = tk.Canvas(
        window,
        width=64,
        height=64,
        bg=transparent,
        highlightthickness=0,
        cursor="hand2",
    )
    canvas.pack()
    circle = canvas.create_oval(5, 5, 59, 59, fill="#6D5DFB", outline="#FFFFFF", width=2)
    canvas.create_text(32, 31, text="P", fill="white", font=("Segoe UI", 22, "bold"))
    canvas.create_oval(46, 45, 57, 56, fill="#35D07F", outline="#FFFFFF")

    drag = {"x": 0, "y": 0, "root_x": 0, "root_y": 0, "moved": False}

    def refresh() -> None:
        canvas.itemconfigure(circle, fill="#4338CA" if controller.is_open() else "#6D5DFB")

    def press(event: object) -> None:
        drag["x"] = int(getattr(event, "x"))
        drag["y"] = int(getattr(event, "y"))
        drag["root_x"] = int(getattr(event, "x_root"))
        drag["root_y"] = int(getattr(event, "y_root"))
        drag["moved"] = False

    def move(event: object) -> None:
        root_x = int(getattr(event, "x_root"))
        root_y = int(getattr(event, "y_root"))
        if abs(root_x - drag["root_x"]) + abs(root_y - drag["root_y"]) > 4:
            drag["moved"] = True
        window.geometry(f"+{root_x - drag['x']}+{root_y - drag['y']}")

    def release(_event: object) -> None:
        if not drag["moved"]:
            controller.toggle()
            refresh()

    def open_default() -> None:
        webbrowser.open(controller.url, new=1)

    def quit_all() -> None:
        controller.hide()
        window.destroy()
        on_quit()

    menu = tk.Menu(window, tearoff=False)
    menu.add_command(label="Open / hide ProofMode", command=lambda: (controller.toggle(), refresh()))
    menu.add_command(label="Open in default browser", command=open_default)
    menu.add_separator()
    menu.add_command(label="Quit ProofMode", command=quit_all)

    def show_menu(event: object) -> None:
        menu.tk_popup(int(getattr(event, "x_root")), int(getattr(event, "y_root")))

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", move)
    canvas.bind("<ButtonRelease-1>", release)
    canvas.bind("<Button-3>", show_menu)
    window.bind("<Escape>", lambda _event: quit_all())
    refresh()
    window.mainloop()


def build_service_environment(root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    # PROOFMODE_GEMMA_URL is the application's canonical setting.  The aliases
    # keep the launcher friendly to small standalone clients as well.
    environment.setdefault("PROOFMODE_GEMMA_URL", f"http://{HOST}:{LLAMA_PORT}/v1")
    environment.setdefault("PROOFMODE_GEMMA_BASE_URL", f"http://{HOST}:{LLAMA_PORT}/v1")
    environment.setdefault("LLAMA_BASE_URL", f"http://{HOST}:{LLAMA_PORT}/v1")
    environment.setdefault("OPENAI_BASE_URL", f"http://{HOST}:{LLAMA_PORT}/v1")
    environment.setdefault("PROOFMODE_PROJECT_ROOT", str(Path(root).resolve()))
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def _default_profile_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA", "").strip()
    if not base:
        base = tempfile.gettempdir()
    return Path(base) / "ProofMode" / "BrowserProfile"


def run_launcher(
    *,
    bubble: bool = True,
    open_window: bool = True,
    default_browser: bool = False,
    context_size: int = 8192,
) -> int:
    root = project_root()
    lock = LauncherLock(Path(tempfile.gettempdir()) / "ProofMode" / "launcher.lock")
    if not lock.acquire():
        # Another launcher may still be loading Gemma.  Reuse it instead of
        # racing to create duplicate server processes.
        wait_until_ready(APP_HEALTH_URL, timeout=90.0)
        webbrowser.open(APP_URL, new=1)
        return 0

    processes = OwnedProcesses()
    atexit.register(processes.stop_all)
    environment = build_service_environment(root)
    quit_event = threading.Event()
    controller: AppWindowController | None = None

    def request_quit(*_args: object) -> None:
        quit_event.set()

    try:
        gemma_home = find_gemma_home(root)
        ensure_service(
            name="Gemma",
            host=HOST,
            port=LLAMA_PORT,
            health_url=LLAMA_HEALTH_URL,
            command=build_llama_command(gemma_home, context_size),
            cwd=gemma_home,
            env=environment,
            processes=processes,
            startup_timeout=180.0,
        )
        ensure_service(
            name="ProofMode",
            host=HOST,
            port=STREAMLIT_PORT,
            health_url=APP_HEALTH_URL,
            command=build_streamlit_command(root, environment),
            cwd=root,
            env=environment,
            processes=processes,
            startup_timeout=60.0,
        )

        browser = None if default_browser else find_app_browser(environment)
        controller = AppWindowController(APP_URL, browser, _default_profile_dir())
        if open_window:
            controller.show()

        if bubble:
            run_bubble(controller, request_quit)
        else:
            for named_signal in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(named_signal, request_quit)
                except (ValueError, OSError):
                    pass
            while not quit_event.wait(0.5):
                continue
        return 0
    finally:
        if controller is not None:
            controller.hide()
        processes.stop_all()
        atexit.unregister(processes.stop_all)
        lock.release()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Gemma and the ProofMode desktop app.")
    parser.add_argument("--no-bubble", action="store_true", help="Run without the floating control.")
    parser.add_argument("--no-open", action="store_true", help="Do not open an app window at startup.")
    parser.add_argument(
        "--default-browser",
        action="store_true",
        help="Use the default browser instead of a controllable Edge/Chrome app window.",
    )
    parser.add_argument("--context-size", type=int, default=8192)
    args = parser.parse_args(argv)
    if not 1024 <= args.context_size <= 131072:
        parser.error("--context-size must be between 1024 and 131072")
    return args


def _show_startup_error(message: str) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "ProofMode could not start", 0x10)
    except Exception:
        return


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_launcher(
            bubble=not args.no_bubble,
            open_window=not args.no_open,
            default_browser=args.default_browser,
            context_size=args.context_size,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(message, file=sys.stderr)
        _show_startup_error(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

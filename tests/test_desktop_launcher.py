from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "desktop_launcher.py"
SPEC = importlib.util.spec_from_file_location("desktop_launcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_find_gemma_home_uses_override(tmp_path: Path) -> None:
    configured = tmp_path / "models elsewhere"
    assert launcher.find_gemma_home(tmp_path / "ProofMode", {"PROOFMODE_GEMMA_HOME": str(configured)}) == configured.resolve()


def test_find_gemma_home_defaults_to_sibling(tmp_path: Path) -> None:
    root = tmp_path / "ProofMode"
    assert launcher.find_gemma_home(root, {}) == tmp_path.resolve() / "gemma4"


def test_find_streamlit_entrypoint_and_build_command(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.touch()
    configured_python = tmp_path / "python.exe"
    command = launcher.build_streamlit_command(
        tmp_path,
        {"PROOFMODE_PYTHON": str(configured_python)},
    )
    assert command[:4] == [str(configured_python.resolve()), "-m", "streamlit", "run"]
    assert str(app.resolve()) in command
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert command[command.index("--server.port") + 1] == "8501"


def test_build_llama_command_requires_and_uses_multimodal_files(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    models = tmp_path / "models"
    runtime.mkdir()
    models.mkdir()
    executable_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    executable = runtime / executable_name
    model = models / "gemma-4-E4B_q4_0-it.gguf"
    projector = models / "gemma-4-E4B-it-mmproj.gguf"
    for path in (executable, model, projector):
        path.touch()

    command = launcher.build_llama_command(tmp_path, context_size=4096)
    assert command[0] == str(executable.resolve())
    assert command[command.index("--model") + 1] == str(model.resolve())
    assert command[command.index("--mmproj") + 1] == str(projector.resolve())
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_wait_until_ready_stops_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter((False, False, True))
    check = Mock(side_effect=lambda _url: next(outcomes))
    monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: None)
    launcher.wait_until_ready("http://example.test/health", 1, check=check)
    assert check.call_count == 3


def test_wait_until_ready_reports_early_process_exit() -> None:
    process = Mock()
    process.poll.return_value = 7
    process.returncode = 7
    with pytest.raises(RuntimeError, match="exit code 7"):
        launcher.wait_until_ready(
            "http://example.test/health",
            1,
            process=process,
            check=lambda _url: False,
        )


def test_ensure_service_reuses_existing_healthy_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "is_port_open", lambda _host, _port: True)
    monkeypatch.setattr(launcher, "wait_until_ready", lambda *_args, **_kwargs: None)
    processes = Mock()
    result = launcher.ensure_service(
        name="Gemma",
        host="127.0.0.1",
        port=8080,
        health_url="http://127.0.0.1:8080/health",
        command=["llama-server"],
        cwd=tmp_path,
        env={},
        processes=processes,
        startup_timeout=1,
    )
    assert result is None
    processes.start.assert_not_called()


def test_owned_processes_stops_only_registered_children(monkeypatch: pytest.MonkeyPatch) -> None:
    first = Mock()
    second = Mock()
    stopped = []
    monkeypatch.setattr(launcher, "stop_process", lambda process: stopped.append(process))
    owned = launcher.OwnedProcesses([first, second])
    owned.stop_all()
    assert stopped == [second, first]
    assert owned.children == []


def test_launcher_lock_reclaims_stale_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "launcher.lock"
    lock_path.write_text("999999", encoding="ascii")
    monkeypatch.setattr(launcher, "_pid_is_running", lambda _pid: False)
    lock = launcher.LauncherLock(lock_path)
    assert lock.acquire() is True
    assert lock_path.read_text(encoding="ascii") == str(os.getpid())
    lock.release()
    assert not lock_path.exists()


def test_pid_liveness_check_is_non_destructive() -> None:
    assert launcher._pid_is_running(os.getpid()) is True
    assert launcher._pid_is_running(-1) is False


def test_parse_args_validates_context_size() -> None:
    args = launcher.parse_args(["--no-bubble", "--context-size", "4096"])
    assert args.no_bubble is True
    assert args.context_size == 4096
    with pytest.raises(SystemExit):
        launcher.parse_args(["--context-size", "12"])

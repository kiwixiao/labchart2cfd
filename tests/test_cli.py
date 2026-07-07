"""Tests for CLI entry behavior.

Pins the GUI-first invocation contract: bare `labchart2cfd` launches the GUI,
while subcommands and --help continue to work unchanged. The gui command itself
must survive because the macOS relaunch re-invokes `-m labchart2cfd gui`.
"""
from typer.testing import CliRunner

from labchart2cfd.cli import main as cli_main

runner = CliRunner()


def test_bare_invocation_launches_gui(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_main, "gui", lambda: calls.append("gui"))

    result = runner.invoke(cli_main.app, [])

    assert result.exit_code == 0
    assert calls == ["gui"]


def test_subcommand_does_not_launch_gui(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_main, "gui", lambda: calls.append("gui"))

    result = runner.invoke(cli_main.app, ["version"])

    assert result.exit_code == 0
    assert "labchart2cfd version" in result.output
    assert calls == []


def test_help_still_shows_help_not_gui(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_main, "gui", lambda: calls.append("gui"))

    result = runner.invoke(cli_main.app, ["--help"])

    assert result.exit_code == 0
    assert "process" in result.output  # subcommands listed
    assert calls == []


def test_gui_subcommand_still_exists():
    # The macOS focus relaunch re-execs `-m labchart2cfd gui`; the explicit
    # subcommand must never be removed.
    result = runner.invoke(cli_main.app, ["gui", "--help"])
    assert result.exit_code == 0

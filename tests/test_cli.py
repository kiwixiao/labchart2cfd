"""Tests for CLI entry behavior.

Pins the GUI-first invocation contract: bare `labchart2cfd` launches the GUI,
while subcommands and --help continue to work unchanged. The gui command itself
must survive because the macOS relaunch re-invokes `-m labchart2cfd gui`.
"""
import numpy as np
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


def _mat_with_separated_channels(tmp_path):
    """Write a .mat whose trigger sits on ch1 and flow on ch3, NOT adjacent.

    Channel 2 holds a flat signal, so a reader that guesses "the row above the
    flow" lands on it and finds no steps. Only an exact row 1 works.
    """
    from scipy.io import savemat

    fs = 1000.0
    t = np.arange(0.0, 10.0, 1 / fs)
    trigger = np.zeros_like(t)
    trigger[(t >= 2.0) & (t < 5.0)] = 5.0        # one 3 s step on channel 1
    decoy = np.zeros_like(t)                      # channel 2: nothing to find
    flow = np.sin(2 * np.pi * 0.5 * t)            # channel 3

    path = tmp_path / "sep.mat"
    payload = {"titles": np.array(["trig", "decoy", "flow"], dtype=object)}
    for i, sig in enumerate([trigger, decoy, flow], start=1):
        payload[f"data_block{i - 1}"] = sig
    # LabChart-style flat layout the reader understands.
    payload = {
        "data": np.concatenate([trigger, decoy, flow]),
        "datastart": np.array([[1], [len(t) + 1], [2 * len(t) + 1]], dtype=float),
        "dataend": np.array([[len(t)], [2 * len(t)], [3 * len(t)]], dtype=float),
        "samplerate": np.array([[fs], [fs], [fs]]),
        "titles": np.array(["trig ", "decoy", "flow "]),
    }
    savemat(str(path), payload)
    return path


def test_detect_steps_reads_the_exact_trigger_row_given(tmp_path):
    """--trigger-row must be used verbatim, not guessed from the flow row."""
    mat = _mat_with_separated_channels(tmp_path)

    result = runner.invoke(
        cli_main.app,
        ["detect-steps", str(mat), "--row", "3", "--column", "1", "--trigger-row", "1"],
    )

    assert result.exit_code == 0, result.output
    assert "1 step trigger" in result.output


def test_detect_steps_reports_an_empty_trigger_row_instead_of_guessing(tmp_path):
    """Pointing at the flat decoy channel must say so, not silently fall back."""
    mat = _mat_with_separated_channels(tmp_path)

    result = runner.invoke(
        cli_main.app,
        ["detect-steps", str(mat), "--row", "3", "--column", "1", "--trigger-row", "2"],
    )

    # Channel 2 is flat: no steps, and the message must name row 2.
    assert "2" in result.output
    assert "1 step trigger" not in result.output


def test_validate_verbose_reports_corrupt_blocks_without_crashing(tmp_path):
    """Regression: the reader began emitting corrupt-block dicts without a
    'samples' key while validate --verbose still read it, so the command
    crashed with KeyError on exactly the file the change was written for."""
    from scipy.io import savemat

    n = 100
    path = tmp_path / "inv.mat"
    savemat(str(path), {
        "data": np.concatenate([np.arange(n, dtype=float),
                                np.arange(n, dtype=float) * 10]),
        "datastart": np.array([[80.0], [101.0]]),   # ch1 start beyond its end
        "dataend": np.array([[40.0], [200.0]]),
        "samplerate": np.array([[100.0], [100.0]]),
        "titles": np.array(["flow ", "press"]),
    })

    result = runner.invoke(cli_main.app, ["validate", str(path), "--verbose"])

    assert result.exit_code == 0, result.output
    assert "corrupt" in result.output.lower()
    assert "100" in result.output          # the healthy channel still reported


def test_gui_subcommand_still_exists():
    # The macOS focus relaunch re-execs `-m labchart2cfd gui`; the explicit
    # subcommand must never be removed.
    result = runner.invoke(cli_main.app, ["gui", "--help"])
    assert result.exit_code == 0

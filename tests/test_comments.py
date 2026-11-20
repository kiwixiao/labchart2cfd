"""LabChart comments (the annotation text typed during a recording).

LabChart's export stores them in three variables: ``com`` (one row per
comment: channel or -1 for all, block, position in ticks, type, row of the
text), ``comtext`` (the distinct texts) and ``tickrate`` (ticks per second in
each block). The native file stores them with each record. Both readers must
hand over the same ``Comment`` list.
"""
import numpy as np
import pytest
from scipy.io import loadmat, savemat

from labchart2cfd.io.labchart import Comment, load_labchart_mat
from labchart2cfd.io.recording import load_recording

_COM = np.array([
    [-1, 2, 1000, 1, 1],      # written out of time order on purpose
    [-1, 1, 2000, 1, 1],
    [2, 1, 5000, 1, 2],
], dtype=float)
_COMTEXT = np.array(["CT scan start", "cough        "])


def _with(mat_file, tmp_path, **variables):
    """A copy of the fixture export with extra variables added."""
    mat = {k: v for k, v in loadmat(str(mat_file)).items() if not k.startswith("__")}
    mat.update(variables)
    path = tmp_path / "with_comments.mat"
    savemat(str(path), mat)
    return path


@pytest.fixture
def commented(mat_file, tmp_path):
    return _with(mat_file, tmp_path, com=_COM, comtext=_COMTEXT,
                 tickrate=np.full((1, 3), 1000.0))


def test_comments_of_an_export_are_read_in_time_order(commented):
    data = load_labchart_mat(commented)

    assert data.comments == [
        Comment(block=1, channel=None, time=2.0, text="CT scan start"),
        Comment(block=1, channel=2, time=5.0, text="cough"),
        Comment(block=2, channel=None, time=1.0, text="CT scan start"),
    ]


def test_comments_can_be_asked_for_one_block(commented):
    data = load_labchart_mat(commented)

    assert [c.text for c in data.comments_in(1)] == ["CT scan start", "cough"]
    assert [c.time for c in data.comments_in(2)] == [1.0]
    assert data.comments_in(3) == []


def test_an_export_without_comments_has_none(mat_file):
    assert load_labchart_mat(mat_file).comments == []


def test_an_empty_comment_table_is_no_comments(mat_file, tmp_path):
    path = _with(mat_file, tmp_path, com=np.zeros((0, 5)),
                 tickrate=np.full((1, 3), 1000.0))

    notes = []
    assert load_labchart_mat(path, notes=notes).comments == []
    assert notes == []


def test_the_tick_rate_of_each_block_gives_the_time(mat_file, tmp_path):
    path = _with(mat_file, tmp_path, com=_COM, comtext=_COMTEXT,
                 tickrate=np.array([[1000.0, 500.0, 1000.0]]))

    assert [c.time for c in load_labchart_mat(path).comments_in(2)] == [2.0]


@pytest.mark.parametrize("variables, reason", [
    (dict(com=_COM, comtext=_COMTEXT), "tick rate"),
    (dict(com=_COM, tickrate=np.full((1, 3), 1000.0)), "text"),
    (dict(com=_COM, comtext=_COMTEXT[:1], tickrate=np.full((1, 3), 1000.0)), "text"),
    (dict(com=np.array([[-1.0, 9, 10, 1, 1]]), comtext=_COMTEXT,
          tickrate=np.full((1, 3), 1000.0)), "block"),
    (dict(com=np.array([[7.0, 1, 10, 1, 1]]), comtext=_COMTEXT,
          tickrate=np.full((1, 3), 1000.0)), "channel"),
    (dict(com=_COM[:, :3], comtext=_COMTEXT,
          tickrate=np.full((1, 3), 1000.0)), "shape"),
])
def test_comments_that_cannot_be_placed_are_dropped_and_named(
        mat_file, tmp_path, variables, reason):
    """A comment at a guessed time would be worse than none. The signals
    still load, and the note says why the comments are missing."""
    path = _with(mat_file, tmp_path, **variables)
    notes = []

    data = load_labchart_mat(path, notes=notes)

    assert data.comments == []
    assert data.num_channels == 4                  # the recording still opens
    assert len(notes) == 1
    assert "comment" in notes[0].lower() and reason in notes[0].lower()


def test_load_recording_passes_the_note_on(mat_file, tmp_path):
    path = _with(mat_file, tmp_path, com=_COM, comtext=_COMTEXT)
    notes = []

    load_recording(path, notes=notes)

    assert len(notes) == 1 and "comment" in notes[0].lower()


def test_comments_of_a_native_file_use_the_same_numbering(monkeypatch, tmp_path):
    """The native reader counts records and channels from 0 and uses -1 for
    all channels. The application counts from 1 and uses None."""
    from tests.test_recording_loading import _FakeAdicht

    class _WithComments(_FakeAdicht):
        comments = [
            {"record": 1, "channel": -1, "tick": 10, "time_s": 1.0, "number": 2,
             "text": "second record"},
            {"record": 0, "channel": 1, "tick": 2, "time_s": 0.2, "number": 1,
             "text": "on flow"},
        ]

    monkeypatch.setattr("labchart2cfd.io.recording.AdichtFile", _WithComments)
    monkeypatch.setattr("labchart2cfd.io.recording.plan_export", lambda adi: ([], []))
    path = tmp_path / "rec.adicht"
    path.write_bytes(b"")

    data = load_recording(path)

    assert data.comments == [
        Comment(block=1, channel=2, time=0.2, text="on flow"),
        Comment(block=2, channel=None, time=1.0, text="second record"),
    ]


# ------------------------------------ a broken table never stops the recording

@pytest.mark.parametrize("com, reason", [
    (np.array([[-1.0, np.nan, 2000, 1, 1]]), "block"),
    (np.array([[np.nan, 1, 2000, 1, 1]]), "channel"),
    (np.array([[-1.0, 1, 2000, 1, np.nan]]), "text"),
    (np.array([[-1.0, 1, np.nan, 1, 1]]), "position"),
    (np.array([[-1.0, 1, np.inf, 1, 1]]), "position"),
    (np.array([[-1.0, 1.7, 2000, 1, 1]]), "block"),
    (np.array([[2.5, 1, 2000, 1, 1]]), "channel"),
    (np.array(["not a table"]), "table"),
])
def test_a_comment_table_with_bad_values_never_stops_the_recording(
        mat_file, tmp_path, com, reason):
    """A NaN in the table raised inside the loader, so the whole recording
    failed to open over its comments."""
    path = _with(mat_file, tmp_path, com=com, comtext=_COMTEXT,
                 tickrate=np.full((1, 3), 1000.0))
    notes = []

    data = load_labchart_mat(path, notes=notes)

    assert data.num_channels == 4 and not data.is_block_empty(2, 1)
    assert data.comments == []
    assert len(notes) == 1 and reason in notes[0].lower()

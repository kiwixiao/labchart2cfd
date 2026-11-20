"""What the window remembers between sessions.

Kept small on purpose: the folder of the last recording and the window's
size and position. Channel roles are not stored. Restoring them would choose
channels without a click, and a wrong channel is the costliest mistake this
application can make.
"""

import os

from PyQt5.QtCore import QSettings

# Names a settings file to use in place of the user's own. The tests set it,
# so a test run neither reads nor changes what the application saved.
SETTINGS_FILE_VARIABLE = "LABCHART2CFD_SETTINGS"

LAST_FOLDER = "last_folder"
WINDOW_GEOMETRY = "window_geometry"


def open_settings() -> QSettings:
    """The settings store: an ini file in the user's configuration folder."""
    override = os.environ.get(SETTINGS_FILE_VARIABLE)
    if override:
        return QSettings(override, QSettings.IniFormat)
    return QSettings(QSettings.IniFormat, QSettings.UserScope, "labchart2cfd", "labchart2cfd")

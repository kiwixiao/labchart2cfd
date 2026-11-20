"""Dialogs of the labchart2cfd GUI: the assembly preview and the export result."""

import html
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QFrame, QLabel, QListWidget,
    QScrollArea, QVBoxLayout,
)


class AssemblyPreviewDialog(QDialog):
    """Shows the assembled synthetic flow profile and asks whether to export it.

    ``exec_()`` returns ``QDialog.Accepted`` only when Export is pressed, so
    the caller can leave the disk untouched after a Cancel.
    """

    def __init__(self, time, flow, n_images, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Synthetic Profile Preview")
        self.resize(640, 440)
        layout = QVBoxLayout(self)

        fig = Figure(figsize=(6, 3.5))
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        ax.plot(time, flow, "b-", linewidth=0.8)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Synthetic time (s)")
        ax.set_ylabel("Mass flow (kg/s)")
        ax.set_title(f"Assembled profile: {len(flow)} samples, {n_images} images")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        layout.addWidget(canvas)

        layout.addWidget(QLabel(
            "This is the profile that will be written. Export saves it, "
            "Cancel changes nothing."
        ))

        buttons = QDialogButtonBox()
        self.export_button = buttons.addButton("Export", QDialogButtonBox.AcceptRole)
        self.cancel_button = buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.export_button.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class QuickGuideDialog(QDialog):
    """The whole flow on one small page, opened from the Help menu."""

    GUIDE = (
        "<b>1. Recording.</b> Press Open recording and pick a LabChart .mat "
        "export or the native .adicht file, or drop the file on the window. "
        "A native file needs no export from LabChart first; it takes longer "
        "to open. Check that Flow, Trigger and "
        "Pressure name the right channels. Click a block to open it.<br><br>"
        "<b>2. Study type.</b> Choose MRI, CPAP, Phase contrast or CT. Only "
        "the controls of that type are shown.<br><br>"
        "<b>3. Time window.</b> Click the plot once for Start and once for "
        "End, or type the times.<br><br>"
        "<b>4. CT only.</b> Select the scan step, then choose how the breath "
        "is built: reorder image parts, landmarks, or cycle frames.<br><br>"
        "<b>Export.</b> Enter the Subject ID and press Export. The line above "
        "the button says what is still missing.<br><br>"
        "The coloured line above the plot always says what a click on the "
        "plot will do. If it turns red, the zoom or pan tool is on.<br><br>"
        "<b>Moving in time.</b> The mouse wheel over a plot zooms in time "
        "about the pointer. The window in force is shaded on both plots.<br><br>"
        "<b>Scales.</b> Flow Y and Trigger Y above the plot enlarge or "
        "shrink each scale. The mouse wheel over the numbers of an axis does "
        "the same. Fit sets the flow scale to the flow in view. Full shows "
        "the whole range of the trigger channel.<br><br>"
        "<b>Comments.</b> The comments typed in LabChart are drawn as dashed "
        "purple lines with their text. View, Show comments hides them."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick guide")
        self.setModal(False)
        layout = QVBoxLayout(self)
        self.guide_label = QLabel(self.GUIDE)
        self.guide_label.setTextFormat(Qt.RichText)
        self.guide_label.setWordWrap(True)
        self.guide_label.setMinimumWidth(self._TEXT_WIDTH)
        self.guide_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.guide_label.setContentsMargins(4, 4, 12, 4)
        # A wrapped label does not tell a dialog how tall it is, so the
        # window opened at 400 pixels for text that needs 640 and the rest
        # could not be reached. The text scrolls, and the window opens tall
        # enough for all of it where the screen allows.
        self.guide_scroll = QScrollArea()
        self.guide_scroll.setWidgetResizable(True)
        self.guide_scroll.setFrameShape(QFrame.NoFrame)
        self.guide_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.guide_scroll.setWidget(self.guide_label)
        layout.addWidget(self.guide_scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        width = self._TEXT_WIDTH + 16 + 40      # text, its margins, scroll bar and frame
        wanted = self.guide_label.heightForWidth(self._TEXT_WIDTH + 16) + 90
        screen = self.screen().availableGeometry().height() if self.screen() else 800
        self.resize(width, min(wanted, int(screen * 0.85)))

    _TEXT_WIDTH = 440


class ExportResultDialog(QDialog):
    """Lists what an export wrote, with any warning shown above the details.

    ``report`` holds ``folder`` (Path), ``files`` (list of Path), ``summary``
    (list of label and value pairs), ``warnings`` (list of str) and ``plot``
    (Path of a picture of the exported data, or None).
    """

    PLOT_WIDTH = 620

    def __init__(self, report: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export complete")
        self._folder = Path(report["folder"])
        files = list(report.get("files", []))
        warnings = list(report.get("warnings", []))
        layout = QVBoxLayout(self)

        heading = QLabel(f"Wrote {len(files)} file{'s' if len(files) != 1 else ''} to")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        folder_label = QLabel(str(self._folder))
        folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        folder_label.setWordWrap(True)
        layout.addWidget(folder_label)

        # Warnings come first: a missing pressure profile used to sit at the
        # far end of a long status bar line.
        self.warning_label = QLabel("\n".join(warnings))
        self.warning_label.setObjectName("warningBox")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(bool(warnings))
        layout.addWidget(self.warning_label)

        self.file_list = QListWidget()
        for path in files:
            self.file_list.addItem(Path(path).name)
        row_height = self.file_list.sizeHintForRow(0) if files else 18
        self.file_list.setFixedHeight(max(1, len(files)) * row_height + 8)
        layout.addWidget(self.file_list)

        # One wrapped label holding a two column table. Separate wrapped
        # labels in a form layout were given one line of height each, so a
        # long row (the image order, the conversion) was cut off mid text.
        rows = "".join(
            "<tr><td style='padding: 2px 12px 2px 0; white-space: nowrap;'>"
            f"{html.escape(str(label))}:</td>"
            f"<td style='padding: 2px 0;'>{html.escape(str(value))}</td></tr>"
            for label, value in report.get("summary", [])
        )
        self.summary_label = QLabel(f"<table cellspacing='0'>{rows}</table>")
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_label.setMinimumWidth(self.PLOT_WIDTH)
        layout.addWidget(self.summary_label)

        plot = report.get("plot")
        if plot is not None and Path(plot).exists():
            pixmap = QPixmap(str(plot))
            if not pixmap.isNull():
                picture = QLabel()
                picture.setPixmap(
                    pixmap.scaledToWidth(self.PLOT_WIDTH, Qt.SmoothTransformation)
                )
                layout.addWidget(picture)

        buttons = QDialogButtonBox()
        self.open_folder_button = buttons.addButton(
            "Open folder", QDialogButtonBox.ActionRole
        )
        self.close_button = buttons.addButton("Close", QDialogButtonBox.AcceptRole)
        self.close_button.setDefault(True)
        self.open_folder_button.clicked.connect(self._open_folder)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _open_folder(self) -> None:
        """Show the output folder in the system file browser."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._folder)))

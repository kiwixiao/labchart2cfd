"""Dialogs of the labchart2cfd GUI: the assembly preview and the export result."""

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QListWidget, QVBoxLayout,
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

        summary = QFormLayout()
        for label, value in report.get("summary", []):
            value_label = QLabel(str(value))
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            summary.addRow(f"{label}:", value_label)
        layout.addLayout(summary)

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

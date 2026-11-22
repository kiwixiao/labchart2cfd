"""Dialogs of the labchart2cfd GUI: the assembly preview and the export result."""

import html
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup, QDialog, QDialogButtonBox, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QListWidget, QRadioButton, QScrollArea, QVBoxLayout, QWidget,
)

from labchart2cfd.gui.frame_order_plot import SIDE_NAMES, balance_notes, draw_frame_order
from labchart2cfd.processing.volume_balance import (
    MIN_HALF_SHARE, half_volumes, mass_volume_to_ml,
)


class AssemblyPreviewDialog(QDialog):
    """The composed breath with its frame order, before anything is written.

    The curve is the array the export writes: the composed flow with one
    half scaled so in equals out (design 2026-11-22, 4.5 and 4.7). The user
    names the complete half; the curve and the notes follow the choice.
    ``exec_()`` returns ``QDialog.Accepted`` only when Export is pressed, so
    the caller can leave the disk untouched after a Cancel.

    Attributes:
        asm: The AssemblyResult shown.
        balances: The balance for each complete half, {} when none is possible.
        chosen_keep: "positive" or "negative", the half kept as recorded;
            None when no balance is possible.
        shown_notes: The notes under the plot now, for the saved picture.
    """

    # Above 10 percent the composed half differed that much from the
    # complete half, so the extra flow or the breath is probably off.
    FACTOR_CAUTION = (0.9, 1.1)

    def __init__(self, asm, balances, default_keep, notes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Composed Breath Preview")
        self.resize(820, 600)
        self.asm = asm
        self.balances = dict(balances)
        self.chosen_keep = default_keep if self.balances else None
        self._notes = list(notes)
        self.shown_notes = []
        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(8, 4.4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        row = QHBoxLayout()
        row.addWidget(QLabel("Complete half, kept as recorded:"))
        self.keep_buttons = {}
        group = QButtonGroup(self)
        for keep in ("positive", "negative"):
            button = QRadioButton(SIDE_NAMES[keep])
            button.setEnabled(bool(self.balances))
            button.setChecked(keep == self.chosen_keep)
            button.toggled.connect(lambda on, k=keep: on and self._choose(k))
            group.addButton(button)
            row.addWidget(button)
            self.keep_buttons[keep] = button
        row.addStretch(1)
        layout.addLayout(row)

        self.caution_label = QLabel("")
        self.caution_label.setObjectName("warningBox")
        self.caution_label.setWordWrap(True)
        layout.addWidget(self.caution_label)

        layout.addWidget(QLabel(
            "This is the profile that will be written. Export saves it, "
            "Cancel changes nothing."
        ))
        buttons = QDialogButtonBox()
        self.export_button = buttons.addButton("Export", QDialogButtonBox.AcceptRole)
        self.cancel_button = buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.export_button.setDefault(True)
        self.export_button.setEnabled(bool(self.balances))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._redraw()

    def _choose(self, keep: str) -> None:
        self.chosen_keep = keep
        self._redraw()

    def shown_flow(self):
        """The flow on screen now: the one Export writes."""
        if self.chosen_keep is None:
            return self.asm.flow
        return self.balances[self.chosen_keep].flow

    def _redraw(self) -> None:
        caution = ""
        if self.chosen_keep is None:
            notes = list(self._notes)
            above, below = (mass_volume_to_ml(v)
                            for v in half_volumes(self.asm.time, self.asm.flow))
            caution = (
                f"One half holds no volume or almost none: above zero {above:.1f} mL, "
                f"below zero {abs(below):.1f} mL. The smaller must be at least "
                f"{MIN_HALF_SHARE * 100:g} percent of the larger, so in and out "
                "cannot be matched and nothing can be written. Check the drift "
                "correction breath and the frames, then export again."
            )
        else:
            balance = self.balances[self.chosen_keep]
            notes = self._notes + balance_notes(balance, self.chosen_keep, self.asm.time)
            low, high = self.FACTOR_CAUTION
            if not low <= balance.factor <= high:
                caution = (
                    f"The scale factor is {balance.factor:.3f}, more than 10 percent "
                    "from 1. The composed half differed that much before "
                    "scaling, so check the extra flow and the drift correction breath."
                )
        self.shown_notes = notes
        draw_frame_order(self.figure, self.asm.time, self.shown_flow(),
                         self.asm.frames, self.asm.extra_spans, notes)
        self.canvas.draw_idle()
        self.caution_label.setText(caution)
        self.caution_label.setVisible(bool(caution))


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
        "End, or type the times. In CT, Select step sets them. CT also has "
        "the Drift correction breath here: one complete breath, typed as "
        "From and To or picked with two clicks. Its mean flow is the drift "
        "taken out of the export. Without one, the drift is measured on the "
        "image grid.<br><br>"
        "<b>4. Triggers</b> (MRI, CPAP and Phase contrast). Optional. Pick a "
        "run of trigger pulses, and Mark triggers sets Start and End to "
        "it.<br><br>"
        "<b>4. CT scan.</b> Select step sets Start and End to the scan step "
        "and draws the image grid. Temporal Res and Images in step are set "
        "separately. A typed image count moves End at once. A Start Shift "
        "waits for Apply shift.<br><br>"
        "<b>5. Build the breath</b> (CT). Choose Reorder image parts, Inhale "
        "and exhale landmarks, or Label cycle frames. Only the method on "
        "screen is exported. With Reorder image parts and Label cycle "
        "frames, Extra flow keeps the recorded flow after a frame up to the "
        "time you set, before the next frame starts. It is drawn "
        "hatched.<br><br>"
        "<b>Export.</b> Enter the Subject ID and press Export. The line above "
        "the button says what is still missing. With Reorder image parts and "
        "Label cycle frames, a preview first shows the breath that will be "
        "written, with the frame number above each image and the image "
        "number below. Under Complete half, choose the half kept as "
        "recorded, above zero or below zero. The other half is scaled so "
        "the volume in equals the volume out. The written flow file is read "
        "back, and a net volume above 0.01 mL is shown in red.<br><br>"
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
    (list of label and value pairs), ``errors`` (list of str, optional: a file
    that was written but must not be used, shown in red), ``warnings`` (list
    of str) and ``plot`` (Path of a picture of the exported data, or None).
    """

    PLOT_WIDTH = 620

    def __init__(self, report: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export complete")
        self._folder = Path(report["folder"])
        files = list(report.get("files", []))
        errors = list(report.get("errors", []))
        warnings = list(report.get("warnings", []))
        layout = QVBoxLayout(self)

        heading = QLabel(f"Wrote {len(files)} file{'s' if len(files) != 1 else ''} to")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        self.folder_label = QLabel(str(self._folder))
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)

        # Errors, then warnings, come first: a missing pressure profile used
        # to sit at the far end of a long status bar line.
        self.error_label = QLabel("\n".join(errors))
        self.error_label.setObjectName("errorBox")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(bool(errors))
        layout.addWidget(self.error_label)
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

        # The long parts scroll; the folder, the warnings and the buttons stay
        # in view. The composed breath panel (a picture and the balance rows)
        # was taller than a laptop screen. It opened at its layout minimum,
        # which counts a wrapped folder path as one line, and the picture
        # covered the last summary row.
        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.file_list)

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
        # A wrapped rich text label reports too small a height to the layout,
        # so the last row was sliced through. Its height at this width is
        # given to it outright; a wider dialog only needs less.
        self.summary_label.setMinimumHeight(
            self.summary_label.heightForWidth(self.PLOT_WIDTH) + 4)
        body.addWidget(self.summary_label)

        self.plot_label = None
        plot = report.get("plot")
        if plot is not None and Path(plot).exists():
            pixmap = QPixmap(str(plot))
            if not pixmap.isNull():
                self.plot_label = QLabel()
                self.plot_label.setPixmap(
                    pixmap.scaledToWidth(self.PLOT_WIDTH, Qt.SmoothTransformation)
                )
                body.addWidget(self.plot_label)
        body.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(page)
        layout.addWidget(self.scroll, 1)

        buttons = QDialogButtonBox()
        self.open_folder_button = buttons.addButton(
            "Open folder", QDialogButtonBox.ActionRole
        )
        self.close_button = buttons.addButton("Close", QDialogButtonBox.AcceptRole)
        self.close_button.setDefault(True)
        self.open_folder_button.clicked.connect(self._open_folder)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self._fit_to_screen(page)

    def _fit_to_screen(self, page: QWidget) -> None:
        """Open tall enough for every part where the screen allows.

        The width leaves room for the scroll bar, so a page that scrolls is
        not cut on the right. The bar's own width is used: the style metric
        gave 14 px for a bar drawn 19 px wide.
        """
        margins = self.layout().contentsMargins()
        width = (self.PLOT_WIDTH + margins.left() + margins.right()
                 + self.scroll.verticalScrollBar().sizeHint().width())
        self.scroll.setMinimumHeight(page.layout().totalHeightForWidth(self.PLOT_WIDTH))
        self.layout().invalidate()
        wanted = self.layout().totalHeightForWidth(width)
        self.scroll.setMinimumHeight(0)
        screen = self.screen().availableGeometry().height() if self.screen() else 800
        self.resize(width, min(wanted, int(screen * 0.9)))

    def _open_folder(self) -> None:
        """Show the output folder in the system file browser."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._folder)))

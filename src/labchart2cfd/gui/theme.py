"""Visual theme of the labchart2cfd GUI.

One place for colours and the style sheet, so the window looks the same on
macOS and Windows. Standard controls are drawn by Qt's Fusion style; the style
sheet only dresses the widgets that carry meaning: section headers, the click
mode banner, hints, warnings and the Export button.
"""

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

ACCENT = "#1f6feb"
ACCENT_DARK = "#1a5fd0"
TEXT = "#1f2933"
MUTED_TEXT = "#5b6470"
PANEL = "#eceef2"
BORDER = "#cdd2da"

# Banner colours per kind of click mode. "idle" is the resting state, "active"
# means a tool is armed and waiting for a click, "blocked" means clicks are
# being swallowed (zoom or pan is on).
BANNER_COLORS = {
    "idle": {"background": "#e8f0fe", "text": "#1a3d7c", "edge": ACCENT},
    "active": {"background": "#fff4d6", "text": "#6b4e00", "edge": "#e0a100"},
    "blocked": {"background": "#fde2e1", "text": "#8a1c14", "edge": "#d93025"},
}


def _banner_rules() -> str:
    rules = []
    for kind, colors in BANNER_COLORS.items():
        rules.append(
            f'QLabel#modeBanner[kind="{kind}"] {{'
            f' background: {colors["background"]}; color: {colors["text"]};'
            f' border-left: 4px solid {colors["edge"]}; }}'
        )
    return "\n".join(rules)


STYLE_SHEET = f"""
QWidget#sidebar {{ background: {PANEL}; }}

QToolButton#sectionHeader {{
    background: #dde2ea; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 4px;
    padding: 6px 8px; font-size: 13px; font-weight: bold; text-align: left;
}}
QToolButton#sectionHeader:hover {{ background: #d3d9e3; }}
QWidget#sectionBody {{
    background: #ffffff;
    border: 1px solid {BORDER}; border-top: none;
    border-bottom-left-radius: 4px; border-bottom-right-radius: 4px;
}}

QLabel#modeBanner {{ padding: 7px 10px; border-radius: 3px; font-size: 13px; }}
{_banner_rules()}

QLabel#hintLabel {{ color: {MUTED_TEXT}; }}
QLabel#captureHint {{ color: #b3261e; font-weight: bold; }}
QLabel#readout {{ background: #f7f8fa; color: {TEXT}; padding: 2px 6px; }}
QLabel#dialogHeading {{ font-size: 15px; font-weight: bold; }}
QLabel#warningBox {{
    background: #fff4d6; color: #6b4e00;
    border: 1px solid #e0a100; border-radius: 4px; padding: 8px;
}}

QFrame#exportFooter {{ background: #e1e5ec; border-top: 1px solid {BORDER}; }}
QPushButton#primaryButton {{
    background: {ACCENT}; color: #ffffff;
    border: none; border-radius: 5px;
    padding: 9px 12px; font-size: 14px; font-weight: bold;
}}
QPushButton#primaryButton:hover {{ background: {ACCENT_DARK}; }}
QPushButton#primaryButton:pressed {{ background: #174ea6; }}
QPushButton#primaryButton:disabled {{ background: #9db8e8; }}
"""


def apply_theme(app: QApplication) -> None:
    """Give the application the Fusion style and a light palette.

    Fusion looks the same on every platform, which matters because the
    application is developed on macOS and used mostly on Windows.
    """
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#f4f5f7"))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f0f2f5"))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor("#f7f8fa"))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

"""
Theme Manager for interactive visualizations.

Provides light/dark theme presets for Plotly charts.
"""

from typing import Dict, Any


LIGHT_THEME: Dict[str, Any] = {
    "template": "plotly_white",
    "bg_color": "#ffffff",
    "text_color": "#333333",
    "grid_color": "#e5e5e5",
    "up_color": "#26a69a",
    "down_color": "#ef5350",
    "line_colors": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ],
}

DARK_THEME: Dict[str, Any] = {
    "template": "plotly_dark",
    "bg_color": "#1e1e2f",
    "text_color": "#e0e0e0",
    "grid_color": "#3a3a5c",
    "up_color": "#00e676",
    "down_color": "#ff5252",
    "line_colors": [
        "#42a5f5", "#ffca28", "#66bb6a", "#ef5350",
        "#ab47bc", "#ff7043", "#ec407a", "#78909c",
    ],
}

_THEMES: Dict[str, Dict[str, Any]] = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}


class ThemeManager:
    """Manage chart themes for interactive visualizations."""

    def __init__(self, theme: str = "light"):
        self.set_theme(theme)

    def set_theme(self, theme: str) -> None:
        if theme not in _THEMES:
            raise ValueError(f"Unknown theme '{theme}'. Choose from: {list(_THEMES.keys())}")
        self._name = theme
        self._theme = _THEMES[theme]

    @property
    def name(self) -> str:
        return self._name

    @property
    def template(self) -> str:
        return self._theme["template"]

    @property
    def bg_color(self) -> str:
        return self._theme["bg_color"]

    @property
    def text_color(self) -> str:
        return self._theme["text_color"]

    @property
    def grid_color(self) -> str:
        return self._theme["grid_color"]

    @property
    def up_color(self) -> str:
        return self._theme["up_color"]

    @property
    def down_color(self) -> str:
        return self._theme["down_color"]

    def line_color(self, index: int) -> str:
        colors = self._theme["line_colors"]
        return colors[index % len(colors)]

    def layout_defaults(self) -> Dict[str, Any]:
        return {
            "template": self.template,
            "paper_bgcolor": self.bg_color,
            "plot_bgcolor": self.bg_color,
            "font": {"color": self.text_color},
            "xaxis": {"gridcolor": self.grid_color},
            "yaxis": {"gridcolor": self.grid_color},
        }

    @staticmethod
    def available_themes() -> list:
        return list(_THEMES.keys())

import sys
import os
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
DEFAULT_APP_ICON = project_root / "src" / "frontend" / "icons" / "cuba3.ico"

import flet as ft
from src.frontend.theme import get_dark_theme
from src.frontend.views.search_page import SearchPage


def _resolve_host() -> str:
    return os.getenv("FLET_HOST", "0.0.0.0")


def _resolve_port() -> int:
    value = os.getenv("FLET_PORT", "8550").strip()
    try:
        port = int(value)
    except ValueError:
        return 8550
    return port if port > 0 else 8550

def _resolve_view() -> ft.AppView:
    value = os.getenv("FLET_VIEW", "").strip().lower()

    if value in {"desktop", "flet_app"}:
        return ft.AppView.FLET_APP

    if value in {"web", "web_browser", "flet_app_web"}:
        return ft.AppView.WEB_BROWSER

    if value in {"hidden", "flet_app_hidden"}:
        return ft.AppView.FLET_APP_HIDDEN

    # fallback inteligente
    return ft.AppView.FLET_APP

def main(page: ft.Page):
    page.title = "SRI - Sistema de Recuperacion de Turismo en Cuba"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = get_dark_theme()
    page.bgcolor = "#0f0f0f"
    page.scroll = ft.ScrollMode.AUTO
    if DEFAULT_APP_ICON.exists():
        page.window.icon = str(DEFAULT_APP_ICON)

    page.add(SearchPage(page))


if __name__ == "__main__":
    ft.run(
        main, 
        host=_resolve_host(),
        port=_resolve_port(), 
        view=_resolve_view())

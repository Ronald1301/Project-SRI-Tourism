import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
DEFAULT_APP_ICON = project_root / "src" / "frontend" / "icons" / "cuba3.ico"

import flet as ft
from src.frontend.theme import get_dark_theme
from src.frontend.views.search_page import SearchPage

def main(page: ft.Page):
    page.title = "SRI - Sistema de Recuperacion de Turismo"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = get_dark_theme()
    page.bgcolor = "#0f0f0f"
    page.scroll = ft.ScrollMode.AUTO
    if DEFAULT_APP_ICON.exists():
        page.window.icon = str(DEFAULT_APP_ICON)

    page.add(SearchPage(page))

ft.run(main)

import flet as ft
from views.search_page import SearchPage
from theme import get_dark_theme

def main(page: ft.Page):
    # page.title = "SRI - Tourism Recommender System"
    page.title= "SRI - Sistema de Recomendação de Turismo"
    page.theme = get_dark_theme()
    page.bgcolor = "#0f0f0f"

    page.add(SearchPage(page))

ft.app(target=main)
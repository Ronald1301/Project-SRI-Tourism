import flet as ft
from state import AppState
from api.client import search
from components.search_bar import SearchBar
from components.result_card import ResultCard
from components.status_banner import StatusBanner

def SearchPage(page: ft.Page):

    state = AppState()

    results_column = ft.Column()

    def update_ui():
        results_column.controls.clear()

        banner = StatusBanner(state)
        if banner:
            results_column.controls.append(banner)

        if state.ui_state == state.ui_state.SUCCESS:
            for i, doc in enumerate(state.results):
                results_column.controls.append(ResultCard(doc, i))

        page.update()

    def handle_search(query, mode, top_k):

        if not query.strip():
            state.set_error("La consulta no puede estar vacía")
            update_ui()
            return

        state.set_loading()
        update_ui()

        data = search(query, mode, top_k)

        if "error" in data:
            state.set_error(data["error"])
        else:
            state.set_success(
                data.get("results", []),
                data.get("answer_rag", "")
            )

        update_ui()

    search_bar = SearchBar(handle_search)

    return ft.Column(
        controls=[
            ft.Text("Sistema de Recuperación", size=24, weight="bold"),
            search_bar,
            results_column
        ]
    )
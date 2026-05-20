import flet as ft
from src.frontend.api.client import search
from src.frontend.components.rag_answer_card import RagAnswerCard
from src.frontend.components.result_card import ResultCard
from src.frontend.components.search_bar import SearchBar
from src.frontend.components.status_banner import StatusBanner
from src.frontend.state import AppState, UIState

def SearchPage(page: ft.Page):

    state = AppState()

    results_column = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=10,
    )
    load_more_button = ft.TextButton("Cargar más")

    def update_ui():
        results_column.controls.clear()

        banner = StatusBanner(state)
        if banner:
            results_column.controls.append(banner)
            
        if state.answer_rag:
            results_column.controls.append(RagAnswerCard(state.answer_rag))
            
        for i, doc in enumerate(state.results):
            results_column.controls.append(
                ResultCard(doc, i, state.query)
            )

        if state.ui_state in (UIState.SUCCESS, UIState.LOADING) and state.results and state.has_more:
            results_column.controls.append(load_more_button)

        page.update()

    def handle_search(query, mode, top_k):

        if not query.strip():
            state.set_error("Consulta vacía")
            update_ui()
            return

        state.reset_search(query, mode, top_k)
        state.set_loading()
        update_ui()

        data = search(query, mode, top_k, state.page)

        if data.get("error"):
            state.set_error(data["error"])
        else:
            state.set_success(
                data.get("results", []),
                answer_rag=data.get("answer"),
                prompt=data.get("prompt"),
                has_more=data.get("has_more", False),
            )

        update_ui()

    def load_more(e):
        state.page += 1
        state.set_loading()
        update_ui()

        data = search(state.query, state.mode, state.top_k, state.page)

        if data.get("error"):
            state.set_error(data["error"])
        else:
            state.append_results(data.get("results", []))
            if data.get("answer"):
                state.answer_rag = data.get("answer")
            if data.get("prompt"):
                state.prompt = data.get("prompt")
            state.has_more = data.get("has_more", False)
            state.ui_state = UIState.SUCCESS if state.results else UIState.EMPTY

        update_ui()

    load_more_button.on_click = load_more
    search_bar = SearchBar(handle_search)

    return ft.Column(
        expand=True,
        spacing=16,
        controls=[
            ft.Text("Sistema de Recuperación", size=24, weight="bold"),
            search_bar,
            ft.Container(
                content=results_column,
                expand=True,
            ),
        ]
    )

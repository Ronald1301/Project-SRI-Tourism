import flet as ft
from src.frontend.api.client import search
from src.frontend.components.rag_answer_card import RagAnswerCard
from src.frontend.components.result_card import ResultCard
from src.frontend.components.search_bar import SearchBar
from src.frontend.components.status_banner import StatusBanner
from src.frontend.state import AppState, UIState

def _mode_label(mode: str) -> str:
    labels = {
        "vectorial": "Vectorial",
        "lsi": "LSI",
        "hybrid_search": "Hibrido",
    }
    return labels.get(mode, mode or "N/D")


def _loading_placeholder() -> ft.Container:
    return ft.Container(
        padding=20,
        margin=ft.Margin(left=0, top=6, right=0, bottom=6),
        bgcolor="#151515",
        border_radius=18,
        border=ft.Border(
            left=ft.BorderSide(1, "#262626"),
            top=ft.BorderSide(1, "#262626"),
            right=ft.BorderSide(1, "#262626"),
            bottom=ft.BorderSide(1, "#262626"),
        ),
        content=ft.Column(
            spacing=12,
            controls=[
                ft.Container(height=14, width=90, bgcolor="#2a2a2a", border_radius=999),
                ft.Container(height=22, width=420, bgcolor="#303030", border_radius=10),
                ft.Container(height=14, width=620, bgcolor="#242424", border_radius=10),
                ft.Container(height=14, width=560, bgcolor="#242424", border_radius=10),
                ft.Container(height=14, width=280, bgcolor="#242424", border_radius=10),
            ],
        ),
    )


def SearchPage(page: ft.Page):

    state = AppState()
    feedback_bar = ft.SnackBar(
        ft.Text(""),
        open=False,
        bgcolor="#1f2937",
        duration=1800,
        show_close_icon=False,
    )
    page.overlay.append(feedback_bar)

    results_column = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=14,
    )
    load_more_button = ft.TextButton("Cargar más")

    def show_feedback(message: str):
        feedback_bar.content = ft.Text(message, color="#f8fafc")
        feedback_bar.open = True
        page.update()

    def update_ui():
        results_column.controls.clear()

        banner = StatusBanner(state)
        if banner:
            results_column.controls.append(banner)

        if state.answer_rag:
            results_column.controls.append(
                RagAnswerCard(
                    state.answer_rag,
                    query=state.query,
                    mode=_mode_label(state.mode),
                    result_count=len(state.results),
                    prompt=state.prompt,
                )
            )

        if state.ui_state == UIState.LOADING and not state.results:
            results_column.controls.extend(
                [_loading_placeholder(), _loading_placeholder(), _loading_placeholder()]
            )

        for i, doc in enumerate(state.results):
            results_column.controls.append(
                ResultCard(doc, i, state.query, page, show_feedback)
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

    return ft.Row(
        expand=True,
        alignment=ft.MainAxisAlignment.CENTER,
        controls=[
            ft.Container(
                width=1040,
                expand=True,
                padding=ft.Padding(left=24, top=24, right=24, bottom=20),
                content=ft.Column(
                    expand=True,
                    spacing=18,
                    controls=[
                        ft.Container(
                            padding=24,
                            border_radius=24,
                            gradient=ft.LinearGradient(colors=["#101820", "#0f1c2d"]),
                            shadow=[
                                ft.BoxShadow(
                                    spread_radius=0,
                                    blur_radius=26,
                                    color="#02061744",
                                    offset=ft.Offset(0, 12),
                                )
                            ],
                            border=ft.Border(
                                left=ft.BorderSide(1, "#24364d"),
                                top=ft.BorderSide(1, "#24364d"),
                                right=ft.BorderSide(1, "#24364d"),
                                bottom=ft.BorderSide(1, "#24364d"),
                            ),
                            content=ft.Column(
                                spacing=14,
                                controls=[
                                    ft.Text(
                                        "Sistema de Recuperacion Turistica",
                                        size=28,
                                        weight="bold",
                                        color="#f8fafc",
                                    ),
                                    ft.Text(
                                        "Explora resultados con recuperacion vectorial, LSI y busqueda hibrida desde una sola interfaz.",
                                        size=14,
                                        color="#cbd5e1",
                                    ),
                                    ft.Container(
                                        padding=18,
                                        border_radius=18,
                                        bgcolor="#0f1725",
                                        border=ft.Border(
                                            left=ft.BorderSide(1, "#22314a"),
                                            top=ft.BorderSide(1, "#22314a"),
                                            right=ft.BorderSide(1, "#22314a"),
                                            bottom=ft.BorderSide(1, "#22314a"),
                                        ),
                                        content=search_bar,
                                    ),
                                ],
                            ),
                        ),
                        ft.Container(
                            content=results_column,
                            expand=True,
                        ),
                    ],
                ),
            ),
        ]
    )

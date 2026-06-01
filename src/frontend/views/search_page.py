import asyncio

import flet as ft

from src.frontend.api.client import search, send_explicit_feedback, send_implicit_feedback
from src.frontend.components.rag_answer_card import RagAnswerCard
from src.frontend.components.result_card import ResultCard
from src.frontend.components.search_bar import SearchBar
from src.frontend.components.status_banner import StatusBanner
from src.frontend.state import AppState, UIState


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
    loading_token = {"value": None}

    def show_feedback(message: str):
        feedback_bar.content = ft.Text(message, color="#f8fafc")
        feedback_bar.open = True
        page.update()

    async def animate_loading(token):
        loading_token["dots"] = 0
        while loading_token["value"] is token and state.ui_state == UIState.LOADING:
            loading_token["dots"] = (int(loading_token.get("dots", 0)) % 3) + 1
            dots = "." * loading_token["dots"]
            state.loading_stage = "checking_domain"
            state.loading_label = f"Analizando consulta{dots}"
            state.loading_detail = "Verificando el dominio antes de buscar."
            update_ui()
            await asyncio.sleep(0.45)

    async def reveal_cards(cards):
        for card in cards:
            await asyncio.sleep(0.05)
            card.opacity = 1
            card.scale = 1
            card.offset = ft.Offset(0, 0)
            try:
                card.update()
            except AssertionError:
                return

    def start_loading_animation():
        token = object()
        loading_token["value"] = token
        loading_token["dots"] = 0
        page.run_task(animate_loading, token)

    def stop_loading_animation():
        loading_token["value"] = None

    async def submit_feedback(doc_id: str, choice: str):
        relevance = 1 if choice == "like" else 0
        return await asyncio.to_thread(
            send_explicit_feedback,
            state.query,
            doc_id,
            relevance,
            expanded_query=(state.expansion_info or {}).get("expanded_query"),
        )

    def persist_feedback(doc_id: str, choice: str | None):
        state.set_feedback_choice(state.query, doc_id, choice)

    def confirm_feedback(choice: str):
        if choice == "like":
            show_feedback("Marcado como relevante.")
        elif choice == "dislike":
            show_feedback("Marcado como no relevante.")

    def update_ui(*, reveal_results: bool = False):
        results_column.controls.clear()
        cards_to_reveal = []

        banner = StatusBanner(state)
        if banner:
            results_column.controls.append(banner)

        if state.ui_state == UIState.LOADING and not state.results:
            results_column.controls.extend(
                [_loading_placeholder(), _loading_placeholder(), _loading_placeholder()]
            )

        if state.ui_state == UIState.SHOWING_RESULTS and state.answer_rag:
            results_column.controls.append(
                RagAnswerCard(
                    state.answer_rag,
                    query=state.query,
                    result_count=len(state.results),
                    prompt=state.prompt,
                    expansion=state.expansion_info,
                )
            )

        for i, doc in enumerate(state.results):
            doc_id = str(doc.get("doc_id") or "")
            card = ResultCard(
                doc,
                i,
                state.query,
                page,
                show_feedback,
                handle_implicit_feedback,
                feedback_value=state.get_feedback_choice(state.query, doc_id),
                on_feedback_submit=lambda choice, current_doc_id=doc_id: submit_feedback(current_doc_id, choice),
                on_feedback_persist=lambda choice, current_doc_id=doc_id: persist_feedback(current_doc_id, choice),
                on_feedback_success=confirm_feedback,
                on_feedback_error=show_feedback,
                initially_hidden=reveal_results,
            )
            results_column.controls.append(card)
            if reveal_results:
                cards_to_reveal.append(card)

        page.update()
        return cards_to_reveal

    async def handle_search(query, top_k):
        if not query.strip():
            state.set_error("Consulta vacía")
            update_ui()
            return

        state.reset_search(query, top_k)
        state.set_loading()
        state.loading_detail = "Verificando el dominio antes de buscar."
        update_ui()
        start_loading_animation()

        try:
            data = await asyncio.to_thread(search, query, top_k)
        finally:
            stop_loading_animation()

        if data.get("error"):
            state.set_error(data["error"])
            update_ui()
            return

        domain = data.get("domain") or {}
        state.domain_info = domain
        events = data.get("events") or []
        if events:
            last_event = events[-1]
            state.loading_stage = str(last_event.get("stage") or "done")
            state.loading_label = str(last_event.get("message") or "Listo.")
            state.loading_detail = str(last_event.get("message") or "La respuesta ya fue recibida.")

        if domain.get("status") == "OUT_OF_DOMAIN":
            state.set_out_of_domain(domain.get("message"), domain_info=domain)
            update_ui()
            return

        state.set_success(
            data.get("results", []),
            answer_rag=data.get("answer"),
            prompt=data.get("prompt"),
            expansion_info=data.get("expansion"),
            domain_info=domain,
        )

        cards = update_ui(reveal_results=True)
        await reveal_cards(cards)

    def handle_implicit_feedback(doc_id, event):
        if not doc_id:
            return
        send_implicit_feedback(
            state.query,
            doc_id,
            event,
        )

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
                                        "Explora resultados con recuperacion hibrida y respuestas generadas a partir del contexto recuperado.",
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
        ],
    )

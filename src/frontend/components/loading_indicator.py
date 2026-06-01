import flet as ft

from src.frontend.components.status_text import StatusText


def _step_icon(step: str) -> str:
    mapping = {
        "validating_domain": ft.Icons.SEARCH,
        "local_search": ft.Icons.FOLDER_OPEN,
        "web_search": ft.Icons.PUBLIC,
        "generating_answer": ft.Icons.AUTO_AWESOME,
        "done": ft.Icons.CHECK_CIRCLE,
        "out_of_domain": ft.Icons.SEARCH_OFF,
    }
    return mapping.get(step, ft.Icons.CIRCLE_OUTLINED)


def LoadingIndicator(
    title: str,
    subtitle: str,
    *,
    progress: float | None = None,
    history: list[dict[str, object]] | None = None,
) -> ft.Container:
    history_rows: list[ft.Control] = []
    for item in (history or [])[-4:]:
        step = str(item.get("step") or "")
        message = str(item.get("message") or "")
        history_rows.append(
            ft.Row(
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(_step_icon(step), size=14, color="#93c5fd"),
                    ft.Text(message, size=11, color="#dbeafe", max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                ],
            )
        )

    return ft.Container(
        padding=18,
        border_radius=16,
        bgcolor="#132238",
        border=ft.Border(
            left=ft.BorderSide(1, "#24456a"),
            top=ft.BorderSide(1, "#24456a"),
            right=ft.BorderSide(1, "#24456a"),
            bottom=ft.BorderSide(1, "#24456a"),
        ),
        content=ft.Row(
            spacing=14,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.ProgressRing(width=18, height=18, stroke_width=2, color="#93c5fd"),
                ft.Column(
                    expand=True,
                    spacing=8,
                    controls=[
                        StatusText(title, subtitle),
                        ft.ProgressBar(
                            value=progress if progress is not None else None,
                            bgcolor="#1f3147",
                            color="#60a5fa",
                            bar_height=4,
                        ),
                        *history_rows,
                    ],
                ),
            ],
        ),
    )

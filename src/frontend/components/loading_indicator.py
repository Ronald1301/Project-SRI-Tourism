import flet as ft

from src.frontend.components.status_text import StatusText


def LoadingIndicator(title: str, subtitle: str) -> ft.Container:
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
                StatusText(title, subtitle),
            ],
        ),
    )

import flet as ft


def StatusText(title: str, subtitle: str | None = None) -> ft.Column:
    controls = [
        ft.Text(title, size=14, weight="bold", color="#f8fafc"),
    ]
    if subtitle:
        controls.append(ft.Text(subtitle, size=12, color="#cbd5e1"))
    return ft.Column(spacing=4, controls=controls)

import flet as ft

def get_dark_theme():
    return ft.Theme(
        color_scheme=ft.ColorScheme(
            background="#0f0f0f",
            surface="#1a1a1a",
            primary="#3b82f6",
            on_surface="#ffffff",
        )
    )
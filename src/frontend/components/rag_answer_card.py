import flet as ft

def RagAnswerCard(answer):

    if not answer:
        return None

    return ft.Container(
        padding=20,
        bgcolor="#1e293b",
        border_radius=12,
        content=ft.Column(
            controls=[
                ft.Text("Respuesta generada", size=16, weight="bold"),
                ft.Text(answer, size=16),
            ]
        )
    )
